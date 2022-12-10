#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:06:20 2022

@author: ghiggi
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from gpm_api.utils.utils_cmap import get_colormap_setting
from gpm_api.utils.checks import check_is_spatial_2D_field
from gpm_api.visualization.plot import (
    _plot_cartopy_background,
    _plot_cartopy_pcolormesh,
    #  _plot_mpl_imshow,
    _plot_xr_imshow,
    get_colorbar_settings,
)


def plot_swath_lines(ds, ax=None, **kwargs):
    """Plot GPM orbit granule swath lines."""
    # - 0.0485 to account for 2.5 km from pixel center
    # TODO: adapt based on bin length (changing for each sensor) --> FUNCTION
    lon = ds["lon"].transpose("cross_track", "along_track").data
    lat = ds["lat"].transpose("cross_track", "along_track").data
    ax.plot(lon[0, :] + 0.0485, lat[0, :], transform=ccrs.Geodetic(), **kwargs)
    ax.plot(lon[-1, :] - 0.0485, lat[-1, :], transform=ccrs.Geodetic(), **kwargs)

    # ax.plot(da['lon'][:, 0] + 0.0485, da['lat'][:,0],'--k')
    # ax.plot(da['lon'][:,-1] - 0.0485, da['lat'][:,-1],'--k')


# TODO: plot swath polygon
# def plot_swath(ds, ax=None):


def plot_orbit_map(
    da, ax=None, add_colorbar=True, subplot_kw=None, figsize=(12, 10), dpi=100
):
    """Plot GPM orbit granule in a cartographic map."""
    # - Check inputs
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

    # - Add swath lines
    plot_swath_lines(da, ax=ax, linestyle="--", color="black")

    # - Add variable field with matplotlib
    p = _plot_cartopy_pcolormesh(
        ax=ax,
        da=da,
        x="lon",
        y="lat",
        ticklabels=ticklabels,
        plot_kwargs=plot_kwargs,
        cbar_kwargs=cbar_kwargs,
        add_colorbar=add_colorbar,
    )
    # - Return mappable
    return p


def plot_orbit_image(
    da, ax=None, add_colorbar=True, interpolation="nearest", figsize=(12, 10), dpi=100
):
    """Plot GPM orbit granule as in image.

    figsize, dpi used only if ax is None.
    """
    # - Check inputs
    check_is_spatial_2D_field(da)

    # - Initialize figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # - Get colorbar settings as function of product name
    plot_kwargs, cbar_kwargs, ticklabels = get_colorbar_settings(name=da.name)

    # # - Plot with matplotlib
    # p = _plot_mpl_imshow(ax=ax,
    #                      da=da,
    #                      x="along_track",
    #                      y="cross_track",
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
        x="along_track",
        y="cross_track",
        interpolation=interpolation,
        add_colorbar=add_colorbar,
        plot_kwargs=plot_kwargs,
        cbar_kwargs=cbar_kwargs,
        ticklabels=ticklabels,
    )
    # - Return mappable
    return p
