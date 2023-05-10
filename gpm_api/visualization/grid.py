#!/usr/bin/env python3
"""
Created on Sat Dec 10 19:13:34 2022

@author: ghiggi
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from gpm_api.utils.checks import check_is_spatial_2d
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
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    fig_kwargs={},
    subplot_kwargs={},
    cbar_kwargs={},
    **plot_kwargs,
):
    """Plot DataArray 2D field with cartopy."""
    # - Check inputs
    check_is_spatial_2d(da)
    _preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs, subplot_kwargs=subplot_kwargs)

    # - Initialize figure
    if ax is None:
        subplot_kwargs = _preprocess_subplot_kwargs(subplot_kwargs)
        fig, ax = plt.subplots(subplot_kw=subplot_kwargs, **fig_kwargs)
        # - Add cartopy background
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
        x="lon",
        y="lat",
        interpolation=interpolation,
        plot_kwargs=plot_kwargs,
        cbar_kwargs=cbar_kwargs,
        add_colorbar=add_colorbar,
    )
    # - Return mappable
    return p


def _plot_grid_map_facetgrid(
    da,
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
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
        ax=None,
        add_colorbar=add_colorbar,
        interpolation=interpolation,
        fig_kwargs={},
        cbar_kwargs={},
        **plot_kwargs,
    )

    # - Add cartopy background to each subplot
    for ax in p.axs.flatten():
        plot_cartopy_background(ax)

    # - Return mappable
    return p


def plot_grid_map(
    da,
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    fig_kwargs={},
    subplot_kwargs={},
    cbar_kwargs={},
    **plot_kwargs,
):
    """Plot DataArray 2D field with cartopy."""
    # Plot FacetGrid with xarray imshow
    if "col" in plot_kwargs or "row" in plot_kwargs:
        p = _plot_grid_map_facetgrid(
            da=da,
            ax=ax,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    # Plot with cartopy imshow
    else:
        p = _plot_grid_map_cartopy(
            da=da,
            ax=ax,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    # - Return mappable
    return p


def plot_grid_image(
    da,
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
        x="lon",
        y="lat",
        interpolation=interpolation,
        add_colorbar=add_colorbar,
        plot_kwargs=plot_kwargs,
        cbar_kwargs=cbar_kwargs,
    )
    # - Return mappable
    return p
