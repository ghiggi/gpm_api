#!/usr/bin/env python3
"""
Created on Sat Dec 10 19:13:34 2022

@author: ghiggi
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from gpm_api.checks import check_is_spatial_2d
from gpm_api.utils.utils_cmap import get_colorbar_settings
from gpm_api.visualization.plot import (
    _plot_cartopy_imshow,
    #  _plot_mpl_imshow,
    _plot_xr_imshow,
    _preprocess_figure_args,
    _preprocess_subplot_kwargs,
    plot_cartopy_background,
)


def _plot_grid_map_cartopy(
    da,
    x="lon",
    y="lat",
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    add_background=True,
    fig_kwargs={},
    subplot_kwargs={},
    cbar_kwargs={},
    **plot_kwargs,
):
    """Plot DataArray 2D field with cartopy."""
    # TODO: allow PlateeCarree subset (--> update _plot_cartopy_imshow)

    # - Check inputs
    check_is_spatial_2d(da)  # TODO: allow RGB !
    _preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs, subplot_kwargs=subplot_kwargs)

    # - Initialize figure
    if ax is None:
        subplot_kwargs = _preprocess_subplot_kwargs(subplot_kwargs)
        fig, ax = plt.subplots(subplot_kw=subplot_kwargs, **fig_kwargs)

    # - Add cartopy background
    if add_background:
        ax = plot_cartopy_background(ax)

    # - If not specified, retrieve/update plot_kwargs and cbar_kwargs as function of variable name
    variable = da.name
    plot_kwargs, cbar_kwargs = get_colorbar_settings(
        name=variable, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs
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
    fig_kwargs={},
    subplot_kwargs={},
    cbar_kwargs={},
    **plot_kwargs,
):
    """Plot DataArray 2D field with xarray."""
    # - Check inputs
    if ax is not None:
        raise ValueError("When plotting with FacetGrid, do not specify the 'ax'.")
    _preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs, subplot_kwargs=subplot_kwargs)
    subplot_kwargs = _preprocess_subplot_kwargs(subplot_kwargs)
    # - Add info required to plot on cartopy axes within FacetGrid
    plot_kwargs.update({"subplot_kws": subplot_kwargs})
    plot_kwargs.update({"transform": ccrs.PlateCarree()})

    # - Plot with FacetGrid
    p = plot_grid_image(
        da=da,
        x=x,
        y=y,
        ax=None,
        add_colorbar=add_colorbar,
        interpolation=interpolation,
        fig_kwargs={},
        cbar_kwargs={},
        **plot_kwargs,
    )

    # - Add cartopy background to each subplot
    if add_background:
        for ax in p.axs.flatten():
            plot_cartopy_background(ax)

    # - Return mappable
    return p


def plot_grid_map(
    da,
    x="lon",
    y="lat",
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    add_background=True,
    fig_kwargs={},
    subplot_kwargs={},
    cbar_kwargs={},
    **plot_kwargs,
):
    """Plot DataArray 2D field with cartopy."""
    # Plot FacetGrid with xarray imshow
    # - TODO: add supertitle, better scale colorbar if cartopy axes !
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


def plot_grid_image(
    da,
    x=None,
    y=None,
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    fig_kwargs={},
    cbar_kwargs={},
    **plot_kwargs,
):
    """Plot DataArray 2D image."""
    # Check inputs
    # check_is_spatial_2d(da)
    _preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs)

    # - Define default x and y
    # if x is None:
    #     x = "lon"
    # if y is None:
    #     y = "lat"

    # Initialize figure
    if ax is None:
        # - If col and row are not provided (not FacetedGrid), initialize
        if "col" not in plot_kwargs and "row" not in plot_kwargs:
            fig, ax = plt.subplots(**fig_kwargs)
        # Add fig_kwargs to plot_kwargs for FacetGrid initialization
        else:
            plot_kwargs.update(fig_kwargs)

    # - If not specified, retrieve/update plot_kwargs and cbar_kwargs as function of product name
    plot_kwargs, cbar_kwargs = get_colorbar_settings(
        name=da.name, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs
    )

    # # - Plot with matplotlib
    # p = _plot_mpl_imshow(ax=ax,
    #                      da=da,
    #                      x="lon",
    #                      y="lat",
    #                      interpolation=interpolation,
    #                      add_colorbar=add_colorbar,
    #                      plot_kwargs=plot_kwargs,
    #                      cbar_kwargs=cbar_kwargs,
    #                      ticklabels=ticklabels,
    # )

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
    print(p.axes)
    if ax is not None:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

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
    fig_kwargs={},
    subplot_kwargs={},
    **plot_kwargs,
):
    """Plot GPM grid mesh in a cartographic map."""
    # - Check inputs
    _preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs, subplot_kwargs=subplot_kwargs)

    # - Initialize figure
    if ax is None:
        subplot_kwargs = _preprocess_subplot_kwargs(subplot_kwargs)
        fig, ax = plt.subplots(subplot_kw=subplot_kwargs, **fig_kwargs)

    # - Add cartopy background
    if add_background:
        ax = plot_cartopy_background(ax)

    # - Select lat coordinate for plotting
    da = xr_obj[y]

    # - Define plot_kwargs to display only the mesh
    plot_kwargs["facecolor"] = "none"
    plot_kwargs["alpha"] = 1
    plot_kwargs["edgecolors"] = (edgecolors,)
    plot_kwargs["linewidth"] = (linewidth,)
    plot_kwargs["antialiased"] = True

    # - Add variable field with cartopy
    p = _plot_grid_map_cartopy(
        da=da,
        ax=ax,
        x=x,
        y=y,
        plot_kwargs=plot_kwargs,
        add_colorbar=False,
    )
    # - Return mappable
    return p
