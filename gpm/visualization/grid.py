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
import matplotlib.pyplot as plt

from gpm import get_plot_kwargs
from gpm.checks import check_is_spatial_2d
from gpm.visualization.facetgrid import (
    CartopyFacetGrid,
    ImageFacetGrid,
    sanitize_facetgrid_plot_kwargs,
)
from gpm.visualization.plot import (
    _plot_cartopy_imshow,
    #  _plot_mpl_imshow,
    _plot_xr_imshow,
    add_optimize_layout_method,
    create_grid_mesh_data_array,
    initialize_cartopy_plot,
    preprocess_figure_args,
    preprocess_subplot_kwargs,
)


def _plot_grid_map_cartopy(
    da,
    x="lon",
    y="lat",
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    add_background=True,
    rgb=False,
    fig_kwargs=None,
    subplot_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot DataArray 2D field with cartopy."""
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
    p = _plot_cartopy_imshow(
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


def _plot_grid_map_facetgrid(
    da,
    x="lon",
    y="lat",
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    add_background=True,
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

    projection = subplot_kwargs.get("projection", None)
    if projection is None:
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
        _plot_grid_map_cartopy,
        x=x,
        y=y,
        add_colorbar=False,
        add_background=add_background,
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


def plot_grid_map(
    da,
    x="lon",
    y="lat",
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    add_background=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot DataArray 2D field with cartopy."""
    # Plot FacetGrid with xarray imshow
    if "col" in plot_kwargs or "row" in plot_kwargs:
        p = _plot_grid_map_facetgrid(
            da=da,
            x=x,
            y=y,
            ax=ax,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            add_background=add_background,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    # Plot with cartopy imshow
    else:
        da = da.squeeze()  # remove time if dim=1
        p = _plot_grid_map_cartopy(
            da=da,
            x=x,
            y=y,
            ax=ax,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            add_background=add_background,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    # - Return mappable
    return p


def _plot_grid_image(
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
    """Plot DataArray 2D image."""
    # - Check inputs
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
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # - Monkey patch the mappable instance to add optimize_layout
    if not is_facetgrid:
        p = add_optimize_layout_method(p)

    # - Return mappable
    return p


def plot_grid_mesh(
    xr_obj,
    x="lon",
    y="lat",
    ax=None,
    edgecolors="k",
    linewidth=0.1,
    add_background=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    **plot_kwargs,
):
    """Plot GPM grid mesh in a cartographic map."""
    from gpm.visualization.orbit import _plot_cartopy_pcolormesh

    # - Initialize figure if necessary
    ax = initialize_cartopy_plot(
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
        add_background=add_background,
    )

    # - Create 2D mesh DataArray
    da = create_grid_mesh_data_array(xr_obj, x=x, y=y)

    # - Define plot_kwargs to display only the mesh
    plot_kwargs["facecolor"] = "none"
    plot_kwargs["alpha"] = 1
    plot_kwargs["edgecolors"] = (edgecolors,)
    plot_kwargs["linewidth"] = (linewidth,)
    plot_kwargs["antialiased"] = True

    # - Add variable field with cartopy
    p = _plot_cartopy_pcolormesh(
        da=da,
        ax=ax,
        x=x,
        y=y,
        plot_kwargs=plot_kwargs,
        add_colorbar=False,
    )

    # - Monkey patch the mappable instance to add optimize_layout
    return add_optimize_layout_method(p)

    # - Return mappable


def plot_grid_image(
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
        p = _plot_grid_image_facetgrid(
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
        p = _plot_grid_image(
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


def _plot_grid_image_facetgrid(
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
        _plot_grid_image,
        x=x,
        y=y,
        add_colorbar=False,
        interpolation=interpolation,
        cbar_kwargs=cbar_kwargs,
        **plot_kwargs,
    )

    # Remove duplicated axis labels
    fc.remove_duplicated_axis_labels()

    # Add colorbar
    if add_colorbar:
        fc.add_colorbar(**cbar_kwargs)

    return fc
