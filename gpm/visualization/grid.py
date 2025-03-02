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
"""This module contains functions to visualize GPM-API GRID data."""

from gpm import get_plot_kwargs
from gpm.checks import check_is_spatial_2d
from gpm.visualization.facetgrid import (
    CartopyFacetGrid,
    sanitize_facetgrid_plot_kwargs,
)
from gpm.visualization.plot import (
    add_optimize_layout_method,
    check_object_format,
    create_grid_mesh_data_array,
    infer_map_xy_coords,
    infer_xy_labels,
    initialize_cartopy_plot,
    plot_cartopy_imshow,
    #  plot_mpl_imshow,
    preprocess_figure_args,
    preprocess_subplot_kwargs,
)

####----------------------------------------------------------------------------
#### Low-level Function


def _plot_grid_map_cartopy(
    da,
    x=None,
    y=None,
    ax=None,
    interpolation="nearest",
    add_colorbar=True,
    add_background=True,
    add_gridlines=True,
    add_labels=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot xarray.DataArray 2D field (with optional RGB dimension) with cartopy."""
    # - Initialize figure if necessary
    ax = initialize_cartopy_plot(
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
        add_background=add_background,
        add_gridlines=add_gridlines,
        add_labels=add_labels,
    )

    # - Sanitize plot_kwargs set by by xarray FacetGrid.map_dataarray
    is_facetgrid = plot_kwargs.get("_is_facetgrid", False)
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

    # - Add variable field with matplotlib
    p = plot_cartopy_imshow(
        ax=ax,
        da=da,
        x=x,
        y=y,
        interpolation=interpolation,
        plot_kwargs=plot_kwargs,
        cbar_kwargs=cbar_kwargs,
        add_colorbar=add_colorbar,
    )

    # - Monkey patch the mappable instance to add optimize_layout
    if not is_facetgrid:
        p = add_optimize_layout_method(p)

    # - Return mappable
    return p


####----------------------------------------------------------------------------
#### FacetGrid Wrapper


def _plot_grid_map_facetgrid(
    da,
    x=None,
    y=None,
    ax=None,
    interpolation="nearest",
    add_colorbar=True,
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
    if projection is None:
        raise ValueError("Please specify a Cartopy projection in subplot_kwargs['projection'].")

    # Disable colorbar if rgb
    if plot_kwargs.get("rgb", False):
        add_colorbar = False
        cbar_kwargs = {}

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
        _plot_grid_map_cartopy,
        x=x,
        y=y,
        add_colorbar=False,
        add_background=add_background,
        add_gridlines=add_gridlines,
        add_labels=add_labels,
        interpolation=interpolation,
        cbar_kwargs=cbar_kwargs,
        **plot_kwargs,
    )

    # Remove duplicated axis labels
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


def plot_grid_map(
    da,
    x=None,
    y=None,
    ax=None,
    interpolation="nearest",
    add_colorbar=True,
    add_background=True,
    add_labels=True,
    add_gridlines=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot xarray.DataArray 2D field with cartopy."""
    # Check inputs
    da = check_object_format(da, plot_kwargs=plot_kwargs, check_function=check_is_spatial_2d, strict=True)
    x, y = infer_map_xy_coords(da, x=x, y=y)
    # Plot FacetGrid with xarray imshow
    if "col" in plot_kwargs or "row" in plot_kwargs:
        p = _plot_grid_map_facetgrid(
            da=da,
            x=x,
            y=y,
            ax=ax,
            interpolation=interpolation,
            add_colorbar=add_colorbar,
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
        p = _plot_grid_map_cartopy(
            da=da,
            x=x,
            y=y,
            ax=ax,
            interpolation=interpolation,
            add_colorbar=add_colorbar,
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


def plot_grid_mesh(
    xr_obj,
    x=None,
    y=None,
    ax=None,
    edgecolors="k",
    linewidth=0.1,
    add_background=True,
    add_gridlines=True,
    add_labels=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    **plot_kwargs,
):
    """Plot GPM grid mesh in a cartographic map."""
    from gpm.visualization.orbit import plot_cartopy_pcolormesh

    # Initialize figure if necessary
    ax = initialize_cartopy_plot(
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
        add_background=add_background,
        add_gridlines=add_gridlines,
        add_labels=add_labels,
    )
    # Infer x and y
    x, y = infer_xy_labels(xr_obj, x=x, y=y, rgb=plot_kwargs.get("rgb"))

    # Create 2D mesh xarray.DataArray
    da = create_grid_mesh_data_array(xr_obj, x=x, y=y)

    # Define plot_kwargs to display only the mesh
    plot_kwargs["facecolor"] = "none"
    plot_kwargs["alpha"] = 1
    plot_kwargs["edgecolors"] = (edgecolors,)
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

    # Monkey patch the mappable instance to add optimize_layout
    p = add_optimize_layout_method(p)

    # Return mappable
    return p
