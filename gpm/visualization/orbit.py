# -----------------------------------------------------------------------------.
# MIT License

# Copyright (c) 2024 GPM-API developers
#
# This file is part of GPM-API.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -----------------------------------------------------------------------------.
"""This module contains functions to visualize GPM-API ORBIT data."""
import functools

import cartopy.crs as ccrs
import numpy as np

from gpm import get_plot_kwargs
from gpm.checks import check_has_spatial_dim
from gpm.utils.checks import (
    get_slices_contiguous_scans,
)
from gpm.visualization.facetgrid import (
    CartopyFacetGrid,
    sanitize_facetgrid_plot_kwargs,
)
from gpm.visualization.plot import (
    add_optimize_layout_method,
    check_object_format,
    infer_map_xy_coords,
    infill_invalid_coords,
    initialize_cartopy_plot,
    plot_cartopy_pcolormesh,
    plot_sides,
    #  plot_mpl_imshow,
    preprocess_figure_args,
    preprocess_subplot_kwargs,
)

####----------------------------------------------------------------------------
#### ORBIT utilities


def infer_orbit_xy_dim(da, x, y):
    """
    Infer possible along-track and cross-track dimensions for the given DataArray.

    Parameters
    ----------
    da : xarray.DataArray
        The input DataArray.
    x : str
        The name of the x coordinate.
    y : str
        The name of the y coordinate.

    Returns
    -------
    tuple
        The inferred (along_track_dim, cross_track_dim) dimensions.
    """
    possible_along_track_dim = ["x", "along_track"]
    possible_cross_track_dim = ["y", "cross_track"]
    coordinates_dims = np.unique(list(da[x].dims) + list(da[y].dims)).tolist()

    # Retrieve n_dims
    n_dims = len(coordinates_dims)  # nadir-only vs 2D
    # Check for cross_track_dim
    cross_track_dim = None
    for dim in coordinates_dims:
        if dim in possible_cross_track_dim:
            cross_track_dim = dim
            break

    # Check for along_track_dim
    along_track_dim = None
    for dim in coordinates_dims:
        if dim in possible_along_track_dim:
            along_track_dim = dim
            break
    if n_dims > 1:
        if cross_track_dim is None:
            raise ValueError(f"Cross-track dimension could not be identified across {coordinates_dims}.")
        if along_track_dim is None:
            raise ValueError(f"Along-track dimension could not be identified across {coordinates_dims}.")
    return along_track_dim, cross_track_dim


def remove_invalid_outer_cross_track(
    xr_obj,
    coord="lon",
    cross_track_dim="cross_track",
    along_track_dim="along_track",
    alpha=None,
):
    """Remove outer cross-track scans if geolocation is always missing."""
    if cross_track_dim not in xr_obj.dims:
        return xr_obj, alpha
    if along_track_dim not in xr_obj.dims:
        coord_arr = np.asanyarray(xr_obj[coord])
        isna = np.isnan(coord_arr)
    else:
        coord_arr = np.asanyarray(xr_obj[coord].transpose(cross_track_dim, along_track_dim))
        isna = np.all(np.isnan(coord_arr), axis=1)
    if isna[0] or isna[-1]:
        is_valid = ~isna
        # Find the index where coordinates start to be valid
        start_index = np.argmax(is_valid)
        # Find the index where the first False value occurs (from the end)
        end_index = len(is_valid) - np.argmax(is_valid[::-1])
        # Define slice
        slc = slice(start_index, end_index)
        # Subset object
        xr_obj = xr_obj.isel({cross_track_dim: slc})
        if alpha is not None:
            alpha = alpha[slc, :]
    return xr_obj, alpha


def _get_contiguous_slices(da, x="lon", y="lat", along_track_dim="along_track", cross_track_dim="cross_track"):
    # NOTE: Using get_slices_regular would split when there is any NaN coordinate
    if along_track_dim not in da.dims:
        list_slices = [None]  # case: cross-track nadir-view / cross-section
    else:
        list_slices = get_slices_contiguous_scans(
            da,
            min_size=2,
            min_n_scans=2,
            x=x,
            y=y,
            along_track_dim=along_track_dim,
            cross_track_dim=cross_track_dim,
        )

    # Check there are scans to plot
    if len(list_slices) == 0:
        raise ValueError("No regular scans available. Impossible to plot.")
    return list_slices


def call_over_contiguous_scans(function):
    """Decorator to call the plotting function multiple times only over contiguous scans intervals."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        # Assumption: only da and ax are passed as args

        # Get data array (first position)
        da = args[0] if len(args) > 0 else kwargs.get("da")

        # Get axis
        ax = args[1] if len(args) > 1 else kwargs.get("ax")

        # Get DataArray in memory
        da = da.compute()

        # Define dimensions and coordinates
        x, y = infer_map_xy_coords(da, x=kwargs.get("x"), y=kwargs.get("y"))
        along_track_dim, cross_track_dim = infer_orbit_xy_dim(da, x=x, y=y)

        # Define kwargs
        user_kwargs = kwargs.copy()
        user_kwargs["x"] = x
        user_kwargs["y"] = y
        p = None
        is_facetgrid = user_kwargs.get("_is_facetgrid", False)
        alpha = user_kwargs.get("alpha", None)
        alpha_2darray_provided = isinstance(alpha, np.ndarray)

        # Get slices with contiguous scans if along_track dimension is available
        list_slices = _get_contiguous_slices(
            da,
            x=x,
            y=y,
            along_track_dim=along_track_dim,
            cross_track_dim=cross_track_dim,
        )

        # - Call the function over each slice
        for i, slc in enumerate(list_slices):
            # Retrieve contiguous data array
            # - slc=None when cross-track cross-section
            tmp_da = da.isel({along_track_dim: slc}) if slc is not None else da

            # Adapt for alpha
            tmp_alpha = alpha[:, slc].copy() if alpha_2darray_provided else None

            # Remove outer cross-track indices if all without coordinates
            # - Infill of coordinates is done separately with infill_invalid_coordins
            # - If along_track cross-section (or nadir-view), return as it is
            tmp_da, tmp_alpha = remove_invalid_outer_cross_track(
                tmp_da,
                alpha=tmp_alpha,
                coord=x,
                cross_track_dim=cross_track_dim,
                along_track_dim=along_track_dim,
            )
            tmp_da, _ = remove_invalid_outer_cross_track(
                tmp_da,
                alpha=None,
                coord=y,
                cross_track_dim=cross_track_dim,
                along_track_dim=along_track_dim,
            )

            # Define temporary kwargs
            tmp_kwargs = user_kwargs.copy()
            tmp_kwargs["da"] = tmp_da
            if alpha_2darray_provided:
                tmp_kwargs["alpha"] = tmp_alpha
            if i == 0:
                tmp_kwargs["ax"] = ax
            else:
                tmp_kwargs["ax"] = p.axes  # Pass previous iteration axis
                tmp_kwargs["add_background"] = False
                tmp_kwargs["add_gridlines"] = False
                tmp_kwargs["add_labels"] = False

            # Set colorbar to False for all except last iteration
            # --> Avoid drawing multiple colorbars (one at each iteration)
            if i != len(list_slices) - 1 and "add_colorbar" in user_kwargs:
                tmp_kwargs["add_colorbar"] = False

            # Before function call
            p = function(**tmp_kwargs)

        # Monkey patch the mappable instance to add optimize_layout
        if not is_facetgrid:
            p = add_optimize_layout_method(p)

        return p

    return wrapper


####----------------------------------------------------------------------------
#### Swath plotting utilities


def _get_swath_line_sides(lon, lat):
    """Compute the top and bottom swath sides."""
    from gpm.utils.area import get_lonlat_corners_from_centroids

    lon_top, lat_top = get_lonlat_corners_from_centroids(lon[0:2], lat[0:2])
    lon_top = lon_top[0, :]
    lat_top = lat_top[0, :]

    lon_bottom, lat_bottom = get_lonlat_corners_from_centroids(lon[-2:], lat[-2:])
    lon_bottom = lon_bottom[-1, :]
    lat_bottom = lat_bottom[-1, :]
    return (lon_top, lat_top), (lon_bottom, lat_bottom)


@call_over_contiguous_scans
def plot_swath_lines(
    da,
    ax=None,
    x="lon",
    y="lat",
    linestyle="--",
    color="k",
    add_background=True,
    add_gridlines=True,
    add_labels=True,
    subplot_kwargs=None,
    fig_kwargs=None,
    **plot_kwargs,
):
    """Plot GPM orbit granule swath lines."""
    # - Initialize figure if necessary
    ax = initialize_cartopy_plot(
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
        add_background=add_background,
        add_gridlines=add_gridlines,
        add_labels=add_labels,
    )

    # - Sanitize coordinates
    da = infill_invalid_coords(da, x=x, y=y)

    # - Retrieve swath sides
    lon = da[x].transpose("cross_track", "along_track").data
    lat = da[y].transpose("cross_track", "along_track").data

    # - Compute the bottom and top swath sides
    side_top, side_bottom = _get_swath_line_sides(lon, lat)

    # - Plot swath lines
    return plot_sides(
        sides=[side_top, side_bottom],
        ax=ax,
        linestyle=linestyle,
        color=color,
        **plot_kwargs,
    )


def plot_swath(
    ds,
    ax=None,
    facecolor="orange",
    edgecolor="black",
    alpha=0.4,
    add_background=True,
    add_gridlines=True,
    add_labels=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    **plot_kwargs,
):
    """Plot GPM orbit granule."""
    from shapely import Polygon

    # TODO: pyresample swath_def.plot() in future
    # - ensure ccw boundary
    # - iterate by descending/ascending blocks
    # --> da.gpm.pyresample_area.boundary
    # --> da.gpm.pyresample_area.outer_boundary.polygon
    # --> da.gpm.pyresample_area.outer_boundary.sides ..

    # - Initialize figure if necessary
    ax = initialize_cartopy_plot(
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
        add_background=add_background,
        add_gridlines=add_gridlines,
        add_labels=add_labels,
    )

    # Retrieve polygon
    swath_def = ds.gpm.pyresample_area
    boundary = swath_def.boundary(force_clockwise=True)
    polygon = Polygon(boundary.vertices[::-1])

    # Plot the swath polygon
    return ax.add_geometries(
        [polygon],
        crs=ccrs.Geodetic(),
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        **plot_kwargs,
    )


####----------------------------------------------------------------------------
#### Low-level Function


@call_over_contiguous_scans
def _plot_orbit_map_cartopy(
    da,
    ax=None,
    x=None,
    y=None,
    add_colorbar=True,
    add_swath_lines=True,
    add_background=True,
    add_labels=True,
    add_gridlines=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot GPM orbit granule in a cartographic map."""
    # Initialize figure if necessary
    ax = initialize_cartopy_plot(
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
        add_background=add_background,
        add_gridlines=add_gridlines,
        add_labels=add_labels,
    )

    # Sanitize plot_kwargs set by by xarray FacetGrid.map_dataarray
    plot_kwargs = sanitize_facetgrid_plot_kwargs(plot_kwargs)

    # If not specified, retrieve/update plot_kwargs and cbar_kwargs as function of variable name
    variable = da.name
    plot_kwargs, cbar_kwargs = get_plot_kwargs(
        name=variable,
        user_plot_kwargs=plot_kwargs,
        user_cbar_kwargs=cbar_kwargs,
    )
    # Specify colorbar label
    if cbar_kwargs.get("label", None) is None:
        unit = da.attrs.get("units", "-")
        cbar_kwargs["label"] = f"{variable} [{unit}]"

    # Add variable field with cartopy
    p = plot_cartopy_pcolormesh(
        ax=ax,
        da=da,
        x=x,
        y=y,
        plot_kwargs=plot_kwargs,
        cbar_kwargs=cbar_kwargs,
        add_colorbar=add_colorbar,
        add_swath_lines=add_swath_lines,
    )
    # Return mappable
    return p


####----------------------------------------------------------------------------
#### FacetGrid Wrapper


def _plot_orbit_map_facetgrid(
    da,
    x=None,
    y=None,
    ax=None,
    add_colorbar=True,
    add_swath_lines=True,
    add_background=True,
    add_gridlines=True,
    add_labels=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot 2D fields with FacetGrid."""
    # Check inputs
    fig_kwargs = preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs, subplot_kwargs=subplot_kwargs, is_facetgrid=True)
    subplot_kwargs = preprocess_subplot_kwargs(subplot_kwargs)

    # Retrieve GPM-API defaults cmap and cbar kwargs
    variable = da.name
    plot_kwargs, cbar_kwargs = get_plot_kwargs(
        name=variable,
        user_plot_kwargs=plot_kwargs,
        user_cbar_kwargs=cbar_kwargs,
    )
    # Retrieve Cartopy projection
    projection = subplot_kwargs.get("projection", None)
    if projection is None:  # _preprocess_subplots_kwargs should set a default projection
        raise ValueError("Please specify a Cartopy projection in subplot_kwargs['projection'].")

    # Disable colorbar if rgb
    if plot_kwargs.get("rgb", False):
        add_colorbar = False
        cbar_kwargs = {}

    # Create FacetGrid
    da = da.compute()
    optimize_layout = plot_kwargs.pop("optimize_layout", True)
    fc = CartopyFacetGrid(
        data=da,
        projection=projection,
        col=plot_kwargs.pop("col", None),
        row=plot_kwargs.pop("row", None),
        col_wrap=plot_kwargs.pop("col_wrap", None),
        axes_pad=plot_kwargs.pop("axes_pad", None),
        add_colorbar=add_colorbar,
        cbar_kwargs=cbar_kwargs,
        fig_kwargs=fig_kwargs,
        facet_height=plot_kwargs.pop("facet_height", 3),
        facet_aspect=plot_kwargs.pop("facet_aspect", 1),
    )

    # Plot the maps
    x, y = infer_map_xy_coords(da, x=x, y=y)
    fc = fc.map_dataarray(
        _plot_orbit_map_cartopy,
        x=x,
        y=y,
        add_colorbar=False,
        add_swath_lines=add_swath_lines,
        add_background=add_background,
        add_gridlines=add_gridlines,
        add_labels=add_labels,
        cbar_kwargs=cbar_kwargs,
        **plot_kwargs,
    )

    # Remove duplicated gridline labels
    fc.remove_duplicated_axis_labels()

    # Add colorbar
    if add_colorbar:
        fc.add_colorbar(**cbar_kwargs)

    # Optimize layout
    if optimize_layout:
        fc.optimize_layout()

    return fc


####----------------------------------------------------------------------------
#### High-level Wrappers


def plot_orbit_map(
    da,
    ax=None,
    x=None,
    y=None,
    add_colorbar=True,
    add_swath_lines=True,
    add_background=True,
    add_gridlines=True,
    add_labels=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot xarray.DataArray 2D field with cartopy."""
    # Check inputs
    da = check_object_format(da, plot_kwargs=plot_kwargs, check_function=check_has_spatial_dim, strict=True)
    # Plot FacetGrid
    if "col" in plot_kwargs or "row" in plot_kwargs:
        x, y = infer_map_xy_coords(da, x=x, y=y)
        p = _plot_orbit_map_facetgrid(
            da=da,
            x=x,
            y=y,
            ax=ax,
            add_colorbar=add_colorbar,
            add_swath_lines=add_swath_lines,
            add_background=add_background,
            add_gridlines=add_gridlines,
            add_labels=add_labels,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    # Plot with cartopy imshow
    else:
        p = _plot_orbit_map_cartopy(
            da=da,
            x=x,
            y=y,
            ax=ax,
            add_colorbar=add_colorbar,
            add_swath_lines=add_swath_lines,
            add_background=add_background,
            add_gridlines=add_gridlines,
            add_labels=add_labels,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    # Return mappable
    return p


@call_over_contiguous_scans
def plot_orbit_mesh(
    da,
    ax=None,
    x=None,
    y=None,
    edgecolors="k",
    linewidth=0.1,
    add_background=True,
    add_gridlines=True,
    add_labels=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    **plot_kwargs,
):
    """Plot GPM orbit granule mesh in a cartographic map."""
    # Initialize figure if necessary
    ax = initialize_cartopy_plot(
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
        add_background=add_background,
        add_gridlines=add_gridlines,
        add_labels=add_labels,
    )

    # Define plot_kwargs to display only the mesh
    plot_kwargs["facecolor"] = "none"
    plot_kwargs["alpha"] = 1
    plot_kwargs["edgecolors"] = (edgecolors,)  # Introduce bugs in Cartopy !
    plot_kwargs["linewidth"] = (linewidth,)
    plot_kwargs["antialiased"] = True

    # Add variable field with cartopy
    p = plot_cartopy_pcolormesh(
        da=da,
        ax=ax,
        x=x,
        y=y,
        plot_kwargs=plot_kwargs,
        add_colorbar=False,
    )
    # Return mappable
    return p
