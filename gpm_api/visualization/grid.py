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
    _plot_cartopy_background,
    _plot_cartopy_imshow,
    #  _plot_mpl_imshow,
    _plot_xr_imshow,
    get_colorbar_settings,
)


def plot_grid_map(
    da,
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    subplot_kw=None,
    figsize=(12, 10),
    dpi=100,
):
    """Plot DataArray 2D field with cartopy.

    figsize, dpi, subplot_kw used only if ax is None.
    """
    # - Check is 2D array ... without time dimension
    check_is_spatial_2D_field(da)

    # - Initialize cartopy projection
    if subplot_kw is None:
        subplot_kw = {"projection": ccrs.PlateCarree()}

    # - Initialize figure
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=subplot_kw, figsize=figsize, dpi=dpi)

    # - Get colorbar settings as function of product name
    plot_kwargs, cbar_kwargs, ticklabels = get_colorbar_settings(name=da.name)

    # - Add cartopy background
    ax = _plot_cartopy_background(ax)

    # - Add variable field with matplotlib
    p = _plot_cartopy_imshow(
        ax=ax,
        da=da,
        x="lon",
        y="lat",
        interpolation=interpolation,
        ticklabels=ticklabels,
        plot_kwargs=plot_kwargs,
        cbar_kwargs=cbar_kwargs,
        add_colorbar=add_colorbar,
    )
    # - Return mappable
    return p


def plot_grid_image(
    da, ax=None, add_colorbar=True, interpolation="nearest", figsize=(12, 10), dpi=100
):
    """Plot DataArray 2D image

    figsize, dpi used only if ax is None.
    """
    # Check inputs
    check_is_spatial_2D_field(da)

    # Initialize figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # - Get colorbar settings as function of product name
    plot_kwargs, cbar_kwargs, ticklabels = get_colorbar_settings(name=da.name)

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
        ticklabels=ticklabels,
    )
    # - Return mappable
    return p
