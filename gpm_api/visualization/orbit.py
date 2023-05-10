#!/usr/bin/env python3
"""
Created on Sat Dec 10 19:06:20 2022

@author: ghiggi
"""
import functools

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from gpm_api.utils.checks import (
    check_contiguous_scans,
    check_is_spatial_2d,
    get_slices_regular,
)
from gpm_api.utils.utils_cmap import get_colorbar_settings
from gpm_api.visualization.plot import (
    _plot_cartopy_pcolormesh,
    #  _plot_mpl_imshow,
    _plot_xr_imshow,
    _preprocess_figure_args,
    _preprocess_subplot_kwargs,
    plot_cartopy_background,
)


def plot_swath_lines(ds, ax=None, **kwargs):
    """Plot GPM orbit granule swath lines."""
    # - 0.0485 to account for 2.5 km from pixel center
    # TODO: adapt based on bin length (changing for each sensor) --> FUNCTION

    # - Initialize figure
    subplot_kwargs = kwargs.get("subplot_kwargs", {})
    fig_kwargs = kwargs.get("fig_kwargs", {})
    if ax is None:
        subplot_kwargs = _preprocess_subplot_kwargs(subplot_kwargs)
        fig, ax = plt.subplots(subplot_kw=subplot_kwargs, **fig_kwargs)
        # - Add cartopy background
        ax = plot_cartopy_background(ax)

    # - Plot swath line
    lon = ds["lon"].transpose("cross_track", "along_track").data
    lat = ds["lat"].transpose("cross_track", "along_track").data
    ax.plot(lon[0, :] + 0.0485, lat[0, :], transform=ccrs.Geodetic(), **kwargs)
    ax.plot(lon[-1, :] - 0.0485, lat[-1, :], transform=ccrs.Geodetic(), **kwargs)

    # ax.plot(da['lon'][:, 0] + 0.0485, da['lat'][:,0],'--k')
    # ax.plot(da['lon'][:,-1] - 0.0485, da['lat'][:,-1],'--k')


def infill_invalid_coords(xr_obj):
    """Replace invalid coordinates with closer valid location.

    It assumes that the invalid pixel variables are already masked to NaN.
    """
    # TODO: unvalid pixel coordinates should be masked by full transparency !

    from gpm_api.utils.checks import _is_non_valid_geolocation

    # Retrieve array indicated unvalid geolocation
    mask = _is_non_valid_geolocation(xr_obj).data
    # Retrieve lon and lat array
    lon = xr_obj["lon"].data
    lat = xr_obj["lat"].data
    # Replace unvalid coordinates with closer valid values
    lon_dummy = lon.copy()
    lon_dummy[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), lon[~mask])
    lat_dummy = lat.copy()
    lat_dummy[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), lat[~mask])
    xr_obj["lon"].data = lon_dummy
    xr_obj["lat"].data = lat_dummy
    return xr_obj


# TODO: plot swath polygon
# def plot_swath(ds, ax=None):

# da.gpm_api.pyresample_area.boundary
# da.gpm_api.pyresample_area.outer_boundary.polygon
# da.gpm_api.pyresample_area.outer_boundary.sides ..


def _call_over_contiguous_scans(function):
    """Decorator to call the plotting function multiple times only over contiguous scans intervals."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        # Assumption: only da and ax are passed as args

        # Get data array (first position)
        da = args[0] if len(args) > 0 else kwargs.get("da")
        # Get axis
        ax = args[1] if len(args) > 1 else kwargs.get("ax")

        # - Check data array
        check_is_spatial_2d(da)

        # - Get slices with contiguous scans and valid geolocation
        list_slices = get_slices_regular(da)
        if len(list_slices) == 0:
            return ValueError("No regular scans available. Impossible to plot.")

        # - Define kwargs
        user_kwargs = kwargs.copy()
        p = None

        # - Call the function over each slice
        for i, slc in enumerate(list_slices):

            # Retrive contiguous data array
            tmp_da = da.isel(along_track=slc)

            # Replace invalid coordinate with closer value
            # - This might be necessary for some products
            #   having all the outer swath invalid coordinates
            # tmp_da = infill_invalid_coords(tmp_da)

            # Define  temporary kwargs
            tmp_kwargs = user_kwargs.copy()
            tmp_kwargs["da"] = tmp_da
            if i == 0:
                tmp_kwargs["ax"] = ax
            else:
                tmp_kwargs["ax"] = p.axes

            # Set colorbar to False for all except last iteration
            # --> Avoid drawing multiple colorbars
            if i != len(list_slices) - 1 and "add_colorbar" in user_kwargs:
                tmp_kwargs["add_colorbar"] = False

            # Before function call
            p = function(**tmp_kwargs)

        return p

    return wrapper


@_call_over_contiguous_scans
def plot_orbit_map(
    da,
    ax=None,
    add_colorbar=True,
    add_swath_lines=True,
    fig_kwargs={},
    subplot_kwargs={},
    cbar_kwargs={},
    **plot_kwargs,
):
    """Plot GPM orbit granule in a cartographic map."""
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

    # - Add swath lines
    if add_swath_lines:
        plot_swath_lines(da, ax=ax, linestyle="--", color="black")

    # - Add variable field with cartopy
    p = _plot_cartopy_pcolormesh(
        ax=ax,
        da=da,
        x="lon",
        y="lat",
        plot_kwargs=plot_kwargs,
        cbar_kwargs=cbar_kwargs,
        add_colorbar=add_colorbar,
    )
    # - Return mappable
    return p


@_call_over_contiguous_scans
def plot_orbit_mesh(
    da,
    ax=None,
    edgecolors="k",
    linewidth=0.1,
    fig_kwargs={},
    subplot_kwargs={},
    **plot_kwargs,
):
    """Plot GPM orbit granule mesh in a cartographic map."""
    # - Check inputs
    check_is_spatial_2d(da)
    _preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs, subplot_kwargs=subplot_kwargs)

    # - Initialize figure
    if ax is None:
        subplot_kwargs = _preprocess_subplot_kwargs(subplot_kwargs)
        fig, ax = plt.subplots(subplot_kw=subplot_kwargs, **fig_kwargs)
        # - Add cartopy background
        ax = plot_cartopy_background(ax)

    # - Define plot_kwargs to display only the mesh
    plot_kwargs["facecolor"] = "none"
    plot_kwargs["alpha"] = 1
    plot_kwargs["edgecolors"] = (edgecolors,)
    plot_kwargs["linewidth"] = (linewidth,)
    plot_kwargs["antialiased"] = True
    print(plot_kwargs)
    # - Add variable field with cartopy
    p = _plot_cartopy_pcolormesh(
        ax=ax,
        da=da,
        x="lon",
        y="lat",
        plot_kwargs=plot_kwargs,
        add_colorbar=False,
    )
    # - Return mappable
    return p


def plot_orbit_image(
    da,
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    fig_kwargs={},
    cbar_kwargs={},
    **plot_kwargs,
):
    """Plot GPM orbit granule as in image."""
    # - Check inputs
    check_is_spatial_2d(da)
    check_contiguous_scans(da)
    _preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs)

    # - Initialize figure
    if ax is None:
        fig, ax = plt.subplots(**fig_kwargs)

    # - If not specified, retrieve/update plot_kwargs and cbar_kwargs as function of product name
    plot_kwargs, cbar_kwargs = get_colorbar_settings(
        name=da.name, plot_kwargs=plot_kwargs, cbar_kwargs=cbar_kwargs
    )

    # # - Plot with matplotlib
    # p = _plot_mpl_imshow(ax=ax,
    #                      da=da,
    #                      x="along_track",
    #                      y="cross_track",
    #                      interpolation=interpolation,
    #                      add_colorbar=add_colorbar,
    #                      plot_kwargs=plot_kwargs,
    #                      cbar_kwargs=cbar_kwargs,
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
    )
    # - Return mappable
    return p
