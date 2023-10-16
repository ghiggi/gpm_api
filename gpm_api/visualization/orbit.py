#!/usr/bin/env python3
"""
Created on Sat Dec 10 19:06:20 2022

@author: ghiggi
"""
import functools

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from gpm_api.checks import check_is_spatial_2d
from gpm_api.utils.checks import (
    check_contiguous_scans,
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


def plot_swath(
    ds, ax=None, facecolor="orange", edgecolor="black", alpha=0.4, add_background=True, **kwargs
):
    """Plot GPM orbit granule."""
    from shapely import Polygon

    # TODO: swath_def.plot() one day ...
    # - iterate by descending/ascending blocks
    # - ensure ccw boundary

    # Retrieve polygon
    swath_def = ds.gpm_api.pyresample_area
    boundary = swath_def.boundary(force_clockwise=True)
    polygon = Polygon(boundary.vertices[::-1])

    # - Initialize figure
    subplot_kwargs = kwargs.get("subplot_kwargs", {})
    fig_kwargs = kwargs.get("fig_kwargs", {})
    if ax is None:
        subplot_kwargs = _preprocess_subplot_kwargs(subplot_kwargs)
        fig, ax = plt.subplots(subplot_kw=subplot_kwargs, **fig_kwargs)

    # - Add cartopy background
    if add_background:
        ax = plot_cartopy_background(ax)

    p = ax.add_geometries(
        [polygon],
        crs=ccrs.Geodetic(),
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        **kwargs,
    )
    return p


def plot_swath_lines(ds, ax=None, x="lon", y="lat", linestyle="--", color="k", **kwargs):
    """Plot GPM orbit granule swath lines."""
    # - 0.0485 to account for 2.5 km from pixel center
    # TODO: adapt based on bin length (changing for each sensor) --> FUNCTION
    # ds.gpm_api.pyresample_area ...

    # - Initialize figure
    subplot_kwargs = kwargs.get("subplot_kwargs", {})
    fig_kwargs = kwargs.get("fig_kwargs", {})
    if ax is None:
        subplot_kwargs = _preprocess_subplot_kwargs(subplot_kwargs)
        fig, ax = plt.subplots(subplot_kw=subplot_kwargs, **fig_kwargs)
        # - Add cartopy background
        ax = plot_cartopy_background(ax)

    # - Plot swath line
    lon = ds[x].transpose("cross_track", "along_track").data
    lat = ds[y].transpose("cross_track", "along_track").data
    p = ax.plot(
        lon[0, :] + 0.0485,
        lat[0, :],
        transform=ccrs.Geodetic(),
        linestyle=linestyle,
        color=color,
        **kwargs,
    )
    p = ax.plot(
        lon[-1, :] - 0.0485,
        lat[-1, :],
        transform=ccrs.Geodetic(),
        linestyle=linestyle,
        color=color,
        **kwargs,
    )
    return p[0]


def infill_invalid_coords(xr_obj, x="lon", y="lat", mask_variables=True):
    """Replace invalid coordinates with closer valid location.

    This operation is required to plot with pcolormesh.
    If mask_variables is True (the default) sets invalid pixel variables to NaN.

    Return tuple with 'sanitized' xr_obj and a mask with the valid coordinates.
    """
    from gpm_api.utils.checks import _is_valid_geolocation

    # Copy object
    xr_obj = xr_obj.copy()

    # Retrieve pixel with valid/invalid geolocation
    xr_valid_mask = _is_valid_geolocation(xr_obj, x=x)  # True=Valid, False=Invalid
    xr_valid_mask.name = "valid_geolocation_mask"

    np_valid_mask = xr_valid_mask.data  # True=Valid, False=Invalid
    np_unvalid_mask = ~np_valid_mask  # True=Invalid, False=Valid

    # If there are invalid pixels, replace invalid coordinates with closer valid values
    if np.any(np_unvalid_mask):
        lon = np.asanyarray(xr_obj[x].data)
        lat = np.asanyarray(xr_obj[y].data)
        lon_dummy = lon.copy()
        lon_dummy[np_unvalid_mask] = np.interp(
            np.flatnonzero(np_unvalid_mask), np.flatnonzero(np_valid_mask), lon[np_valid_mask]
        )
        lat_dummy = lat.copy()
        lat_dummy[np_unvalid_mask] = np.interp(
            np.flatnonzero(np_unvalid_mask), np.flatnonzero(np_valid_mask), lat[np_valid_mask]
        )
        xr_obj[x].data = lon_dummy
        xr_obj[y].data = lat_dummy

    # Mask variables if asked
    if mask_variables:
        xr_obj = xr_obj.where(xr_valid_mask)

    return xr_obj, xr_valid_mask


# TODO: plot swath polygon
# def plot_swath(ds, ax=None):

# da.gpm_api.pyresample_area.boundary
# da.gpm_api.pyresample_area.outer_boundary.polygon
# da.gpm_api.pyresample_area.outer_boundary.sides ..


def _remove_invalid_outer_cross_track(xr_obj, x="lon"):
    """Remove outer crosstrack scans if geolocation is always missing."""
    lon = np.asanyarray(xr_obj[x].transpose("cross_track", "along_track"))
    isna = np.all(np.isnan(lon), axis=1)
    if isna[0] or isna[-1]:
        # Find the index where the first False value occurs
        start_index = np.argmax(isna is False)
        # Find the index where the first False value occurs (from the end)
        end_index = len(isna) - np.argmax(isna[::-1] is False)
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
        # TODO: improve for rgb=True
        # TODO: Make independent of name of lon/lat, cross_track, along_track
        rgb = kwargs.get("rgb", False)
        if not rgb:
            # - Check data array
            check_is_spatial_2d(da)

            # - Get slices with contiguous scans and valid geolocation
            list_slices = get_slices_regular(da)
            if len(list_slices) == 0:
                return ValueError("No regular scans available. Impossible to plot.")
        else:
            list_slices = [slice(0, None)]

        # - Define kwargs
        user_kwargs = kwargs.copy()
        p = None
        x = user_kwargs["x"]
        y = user_kwargs["y"]

        # - Call the function over each slice
        for i, slc in enumerate(list_slices):
            if not rgb:
                # Retrieve contiguous data array
                tmp_da = da.isel({"along_track": slc})

                # Remove outer cross-track indices if all without coordinates
                tmp_da = _remove_invalid_outer_cross_track(tmp_da, x=x)
            else:
                tmp_da = da

            # Replace invalid coordinate with closer value
            # - This might be necessary for some products
            #   having all the outer swath invalid coordinates
            # - An example is the 2B-GPM-CORRA
            tmp_da, tmp_da_valid_mask = infill_invalid_coords(tmp_da, x=x, y=y, mask_variables=True)

            # Define temporary kwargs
            tmp_kwargs = user_kwargs.copy()
            tmp_kwargs["da"] = tmp_da
            if i == 0:
                tmp_kwargs["ax"] = ax

            else:
                tmp_kwargs["ax"] = p.axes
                tmp_kwargs["add_background"] = False

            # Define alpha to make invalid coordinates transparent
            # --> cartopy.pcolormesh currently bug when providing a alpha array :(
            # TODO: Open an issue on that !

            # tmp_valid_mask = tmp_da_valid_mask.data
            # if not np.all(tmp_valid_mask):
            #     alpha = tmp_valid_mask.astype(int)  # 0s and 1s
            #     if "alpha" in tmp_kwargs:
            #         alpha = tmp_kwargs["alpha"] * alpha
            #     tmp_kwargs["alpha"] = alpha

            # Set colorbar to False for all except last iteration
            # --> Avoid drawing multiple colorbars
            if i != len(list_slices) - 1 and "add_colorbar" in user_kwargs:
                tmp_kwargs["add_colorbar"] = False

            # Before function call
            p = function(**tmp_kwargs)
            # p.set_alpha(alpha)

        return p

    return wrapper


@_call_over_contiguous_scans
def plot_orbit_map(
    da,
    ax=None,
    x="lon",
    y="lat",
    add_colorbar=True,
    add_swath_lines=True,
    add_background=True,
    rgb=False,
    fig_kwargs={},
    subplot_kwargs={},
    cbar_kwargs={},
    **plot_kwargs,
):
    """Plot GPM orbit granule in a cartographic map."""
    # - Check inputs
    if not rgb:
        check_is_spatial_2d(da)
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

    # - Add swath lines
    # --> Fail if not cross_track, along_track dimension currently
    if add_swath_lines and not rgb:
        p = plot_swath_lines(da, ax=ax, linestyle="--", color="black")

    # - Add variable field with cartopy
    p = _plot_cartopy_pcolormesh(
        ax=ax,
        da=da,
        x=x,
        y=y,
        plot_kwargs=plot_kwargs,
        cbar_kwargs=cbar_kwargs,
        add_colorbar=add_colorbar,
        rgb=rgb,
    )
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
    fig_kwargs={},
    subplot_kwargs={},
    **plot_kwargs,
):
    """Plot GPM orbit granule mesh in a cartographic map."""
    # - Check inputs
    _preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs, subplot_kwargs=subplot_kwargs)

    # - Initialize figure
    if ax is None:
        subplot_kwargs = _preprocess_subplot_kwargs(subplot_kwargs)
        fig, ax = plt.subplots(subplot_kw=subplot_kwargs, **fig_kwargs)

    # - Add cartopy background
    if add_background:
        ax = plot_cartopy_background(ax)

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
    # - Return mappable
    return p


def plot_orbit_image(
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
    """Plot GPM orbit granule as in image."""
    # - Check inputs
    check_is_spatial_2d(da)
    check_contiguous_scans(da)
    _preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs)

    # - Define default x and y
    # if x is None:
    #     x = "along_track"
    # if y is None:
    #     y = "cross_track"

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
        x=x,
        y=y,
        interpolation=interpolation,
        add_colorbar=add_colorbar,
        plot_kwargs=plot_kwargs,
        cbar_kwargs=cbar_kwargs,
    )
    p.axes.set_xlabel("Along-Track")
    p.axes.set_ylabel("Cross-Track")

    # - Return mappable
    return p
