#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:13:34 2022

@author: ghiggi
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from gpm_api.utils.checks import check_is_spatial_2D_field
from gpm_api.visualization.plot import (
    plot_cartopy_background,
    _plot_cartopy_imshow,
    #  _plot_mpl_imshow,
    _plot_xr_imshow,
    _preprocess_figure_args,
    get_colorbar_settings,
)


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
    # - Check inputs
    check_is_spatial_2D_field(da)
    _preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs, subplot_kwargs=subplot_kwargs)

    # - Initialize figure
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=subplot_kwargs, **fig_kwargs)
        # - Add cartopy background
        ax = plot_cartopy_background(ax)
  
    # - If not specified, retrieve/update plot_kwargs and cbar_kwargs as function of product name
    plot_kwargs, cbar_kwargs = get_colorbar_settings(name=da.name,
                                                     plot_kwargs=plot_kwargs, 
                                                     cbar_kwargs=cbar_kwargs)

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


def plot_grid_image(
    da, ax=None,
    add_colorbar=True,
    interpolation="nearest", 
    fig_kwargs={}, 
    cbar_kwargs={},
    **plot_kwargs,
):
    """Plot DataArray 2D image."""
    # Check inputs
    check_is_spatial_2D_field(da)
    _preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs)
    
    # Initialize figure
    if ax is None:
        fig, ax = plt.subplots(**fig_kwargs)

    # - If not specified, retrieve/update plot_kwargs and cbar_kwargs as function of product name
    plot_kwargs, cbar_kwargs = get_colorbar_settings(name=da.name,
                                                     plot_kwargs=plot_kwargs, 
                                                     cbar_kwargs=cbar_kwargs)

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
