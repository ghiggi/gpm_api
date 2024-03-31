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
import matplotlib.pyplot as plt
import numpy as np

from gpm import get_plot_kwargs
from gpm.checks import check_is_spatial_2d
from gpm.utils.checks import (
    check_contiguous_scans,
    get_slices_contiguous_scans,
)
from gpm.visualization.facetgrid import (
    CartopyFacetGrid,
    ImageFacetGrid,
    sanitize_facetgrid_plot_kwargs,
)
from gpm.visualization.plot import (
    _plot_cartopy_pcolormesh,
    #  _plot_mpl_imshow,
    _plot_xr_imshow,
    add_optimize_layout_method,
    infill_invalid_coords,
    initialize_cartopy_plot,
    plot_sides,
    preprocess_figure_args,
    preprocess_subplot_kwargs,
)


def plot_swath(
    ds,
    ax=None,
    facecolor="orange",
    edgecolor="black",
    alpha=0.4,
    add_background=True,
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


def _remove_invalid_outer_cross_track(xr_obj, coord="lon"):
    """Remove outer crosstrack scans if geolocation is always missing."""
    coord_arr = np.asanyarray(xr_obj[coord].transpose("cross_track", "along_track"))
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
        xr_obj = xr_obj.isel({"cross_track": slc})
    return xr_obj


def _call_over_contiguous_scans(function):
    """Decorator to call the plotting function multiple times only over contiguous scans intervals."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        # Assumption: only da and ax are passed as args

        # Get data array (first position)
        da = args[0] if len(args) > 0 else kwargs.get("da")

        # Get axis
        ax = args[1] if len(args) > 1 else kwargs.get("ax")

        # Check data validity
        rgb = kwargs.get("rgb", False)
        if not rgb:
            # - Check data array
            check_is_spatial_2d(da)
            # - Get slices with contiguous scans
            # --> get_slices_regular would split when there is any NaN coordinate
            list_slices = get_slices_contiguous_scans(da, min_size=2, min_n_scans=2)
            if len(list_slices) == 0:
                raise ValueError("No regular scans available. Impossible to plot.")
        else:
            list_slices = [slice(0, None)]

        # - Define kwargs
        user_kwargs = kwargs.copy()
        p = None
        x = user_kwargs.get("x", "lon")
        y = user_kwargs.get("y", "lat")
        is_facetgrid = user_kwargs.get("_is_facetgrid", False)
        # - Call the function over each slice
        for i, slc in enumerate(list_slices):
            # Retrieve contiguous data array
            tmp_da = da.isel({"along_track": slc})

            # Remove outer cross-track indices if all without coordinates
            # - Infill of coordinates is done separately with infill_invalid_coordins
            tmp_da = _remove_invalid_outer_cross_track(tmp_da, coord=x)
            tmp_da = _remove_invalid_outer_cross_track(tmp_da, coord=y)

            # Define temporary kwargs
            tmp_kwargs = user_kwargs.copy()
            tmp_kwargs["da"] = tmp_da
            if i == 0:
                tmp_kwargs["ax"] = ax

            else:
                tmp_kwargs["ax"] = p.axes
                tmp_kwargs["add_background"] = False

            # Set colorbar to False for all except last iteration
            # --> Avoid drawing multiple colorbars
            if i != len(list_slices) - 1 and "add_colorbar" in user_kwargs:
                tmp_kwargs["add_colorbar"] = False

            # Before function call
            p = function(**tmp_kwargs)
            # p.set_alpha(alpha)

        # Monkey patch the mappable instance to add optimize_layout
        if not is_facetgrid:
            p = add_optimize_layout_method(p)

        return p

    return wrapper


####----------------------------------------------------------------------------
def _get_swath_line_sides(lon, lat):
    """Compute the top and bottom swath sides."""
    from gpm.utils.area import _get_lonlat_corners

    lon_top, lat_top = _get_lonlat_corners(lon[0:2], lat[0:2])
    lon_top = lon_top[0, :]
    lat_top = lat_top[0, :]

    lon_bottom, lat_bottom = _get_lonlat_corners(lon[-2:], lat[-2:])
    lon_bottom = lon_bottom[-1, :]
    lat_bottom = lat_bottom[-1, :]
    return (lon_top, lat_top), (lon_bottom, lat_bottom)


@_call_over_contiguous_scans
def plot_swath_lines(
    da,
    ax=None,
    x="lon",
    y="lat",
    linestyle="--",
    color="k",
    add_background=True,
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


####----------------------------------------------------------------------------


@_call_over_contiguous_scans
def _plot_orbit_map_cartopy(
    da,
    ax=None,
    x="lon",
    y="lat",
    add_colorbar=True,
    add_swath_lines=True,
    add_background=True,
    rgb=False,
    fig_kwargs=None,
    subplot_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot GPM orbit granule in a cartographic map."""
    # - Check inputs
    if not rgb:
        check_is_spatial_2d(da)

    # - Initialize figure if necessary
    ax = initialize_cartopy_plot(
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
        add_background=add_background,
    )

    # - Sanitize plot_kwargs set by by xarray FacetGrid.map_datarray
    plot_kwargs = sanitize_facetgrid_plot_kwargs(plot_kwargs)

    # - If not specified, retrieve/update plot_kwargs and cbar_kwargs as function of variable name
    variable = da.name
    plot_kwargs, cbar_kwargs = get_plot_kwargs(
        name=variable,
        user_plot_kwargs=plot_kwargs,
        user_cbar_kwargs=cbar_kwargs,
    )
    # - Specify colorbar label
    if "label" not in cbar_kwargs:
        unit = da.attrs.get("units", "-")
        cbar_kwargs["label"] = f"{variable} [{unit}]"

    # - Add variable field with cartopy
    return _plot_cartopy_pcolormesh(
        ax=ax,
        da=da,
        x=x,
        y=y,
        plot_kwargs=plot_kwargs,
        cbar_kwargs=cbar_kwargs,
        add_colorbar=add_colorbar,
        add_swath_lines=add_swath_lines,
        rgb=rgb,
    )
    # - Return mappable


def _plot_orbit_image(
    da,
    x=None,
    y=None,
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    fig_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot GPM orbit granule as in image."""
    # ----------------------------------------------
    # - Check inputs
    check_contiguous_scans(da)
    fig_kwargs = preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs)

    # - Initialize figure
    if ax is None:
        if "rgb" not in plot_kwargs:
            check_is_spatial_2d(da)
        _, ax = plt.subplots(**fig_kwargs)

    # - Sanitize plot_kwargs set by by xarray FacetGrid.map_datarray
    is_facetgrid = plot_kwargs.get("_is_facetgrid", False)
    plot_kwargs = sanitize_facetgrid_plot_kwargs(plot_kwargs)

    # - If not specified, retrieve/update plot_kwargs and cbar_kwargs as function of product name
    plot_kwargs, cbar_kwargs = get_plot_kwargs(
        name=da.name,
        user_plot_kwargs=plot_kwargs,
        user_cbar_kwargs=cbar_kwargs,
    )

    # - Plot with xarray
    p = _plot_xr_imshow(
        ax=ax,
        da=da,
        x=x,
        y=y,
        interpolation=interpolation,
        add_colorbar=add_colorbar,
        plot_kwargs=plot_kwargs,
        cbar_kwargs=cbar_kwargs,
    )

    # - Add axis labels
    p.axes.set_xlabel("Along-Track")
    p.axes.set_ylabel("Cross-Track")

    # - Monkey patch the mappable instance to add optimize_layout
    if not is_facetgrid:
        p = add_optimize_layout_method(p)

    # - Return mappable
    return p


@_call_over_contiguous_scans
def plot_orbit_mesh(
    da,
    ax=None,
    x="lon",
    y="lat",
    edgecolors="k",
    linewidth=0.1,
    add_background=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    **plot_kwargs,
):
    """Plot GPM orbit granule mesh in a cartographic map."""
    # - Initialize figure if necessary
    ax = initialize_cartopy_plot(
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
        add_background=add_background,
    )

    # - Define plot_kwargs to display only the mesh
    plot_kwargs["facecolor"] = "none"
    plot_kwargs["alpha"] = 1
    plot_kwargs["edgecolors"] = (edgecolors,)  # Introduce bugs in Cartopy !
    plot_kwargs["linewidth"] = (linewidth,)
    plot_kwargs["antialiased"] = True

    # - Add variable field with cartopy
    return _plot_cartopy_pcolormesh(
        da=da,
        ax=ax,
        x=x,
        y=y,
        plot_kwargs=plot_kwargs,
        add_colorbar=False,
    )
    # - Return mappable


####----------------------------------------------------------------------------


def _plot_orbit_map_facetgrid(
    da,
    x="lon",
    y="lat",
    ax=None,
    add_colorbar=True,
    add_swath_lines=True,
    add_background=True,
    rgb=False,
    fig_kwargs=None,
    subplot_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot 2D fields with FacetGrid."""
    # - Check inputs
    if ax is not None:
        raise ValueError("When plotting with FacetGrid, do not specify the 'ax'.")
    fig_kwargs = preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs, subplot_kwargs=subplot_kwargs)
    subplot_kwargs = preprocess_subplot_kwargs(subplot_kwargs)

    # Retrieve GPM-API defaults cmap and cbar kwargs
    variable = da.name
    plot_kwargs, cbar_kwargs = get_plot_kwargs(
        name=variable,
        user_plot_kwargs=plot_kwargs,
        user_cbar_kwargs=cbar_kwargs,
    )
    # Retrieve projection
    projection = subplot_kwargs.get("projection", None)
    if projection is None:  # _preprocess_subplots_kwargs should set a default projection
        raise ValueError("Please specify a Cartopy projection in subplot_kwargs['projection'].")

    # Create FacetGrid
    optimize_layout = plot_kwargs.pop("optimize_layout", True)
    fc = CartopyFacetGrid(
        data=da.compute(),
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
    fc = fc.map_dataarray(
        _plot_orbit_map_cartopy,
        x=x,
        y=y,
        add_colorbar=False,
        add_background=add_background,
        add_swath_lines=add_swath_lines,
        rgb=rgb,
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


def _plot_orbit_image_facetgrid(
    da,
    x="lon",
    y="lat",
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    fig_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot 2D fields with FacetGrid."""
    # - Check inputs
    if ax is not None:
        raise ValueError("When plotting with FacetGrid, do not specify the 'ax'.")
    fig_kwargs = preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs)

    # Retrieve GPM-API defaults cmap and cbar kwargs
    variable = da.name
    plot_kwargs, cbar_kwargs = get_plot_kwargs(
        name=variable,
        user_plot_kwargs=plot_kwargs,
        user_cbar_kwargs=cbar_kwargs,
    )

    # Create FacetGrid
    fc = ImageFacetGrid(
        data=da.compute(),
        col=plot_kwargs.pop("col", None),
        row=plot_kwargs.pop("row", None),
        col_wrap=plot_kwargs.pop("col_wrap", None),
        axes_pad=plot_kwargs.pop("axes_pad", None),
        fig_kwargs=fig_kwargs,
        cbar_kwargs=cbar_kwargs,
        add_colorbar=add_colorbar,
        aspect=plot_kwargs.pop("aspect", False),
        facet_height=plot_kwargs.pop("facet_height", 3),
        facet_aspect=plot_kwargs.pop("facet_aspect", 1),
    )

    # Plot the maps
    fc = fc.map_dataarray(
        _plot_orbit_image,
        x=x,
        y=y,
        add_colorbar=False,
        interpolation=interpolation,
        cbar_kwargs=cbar_kwargs,
        **plot_kwargs,
    )

    fc.remove_duplicated_axis_labels()

    # Add colorbar
    if add_colorbar:
        fc.add_colorbar(**cbar_kwargs)

    return fc


####----------------------------------------------------------------------------


def plot_orbit_map(
    da,
    ax=None,
    x="lon",
    y="lat",
    add_colorbar=True,
    add_swath_lines=True,
    add_background=True,
    rgb=False,
    fig_kwargs=None,
    subplot_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot DataArray 2D field with cartopy."""
    # Plot FacetGrid
    if "col" in plot_kwargs or "row" in plot_kwargs:
        p = _plot_orbit_map_facetgrid(
            da=da,
            x=x,
            y=y,
            ax=ax,
            add_colorbar=add_colorbar,
            add_swath_lines=add_swath_lines,
            add_background=add_background,
            rgb=rgb,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    # Plot with cartopy imshow
    else:
        da = da.squeeze()  # remove time if dim=1
        p = _plot_orbit_map_cartopy(
            da=da,
            x=x,
            y=y,
            ax=ax,
            add_colorbar=add_colorbar,
            add_swath_lines=add_swath_lines,
            add_background=add_background,
            rgb=rgb,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    # - Return mappable
    return p


def plot_orbit_image(
    da,
    x="lon",
    y="lat",
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    fig_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot DataArray 2D field with cartopy."""
    # Plot FacetGrid with xarray imshow
    if "col" in plot_kwargs or "row" in plot_kwargs:
        p = _plot_orbit_image_facetgrid(
            da=da,
            x=x,
            y=y,
            ax=ax,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )

    # Plot with cartopy imshow
    else:
        da = da.squeeze()  # remove time if dim=1
        p = _plot_orbit_image(
            da=da,
            x=x,
            y=y,
            ax=ax,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    # - Return imagepable
    return p
