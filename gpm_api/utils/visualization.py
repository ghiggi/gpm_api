#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:45:28 2022

@author: ghiggi
"""
import pyproj
import cartopy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Geod
from mpl_toolkits.axes_grid1 import make_axes_locatable
from gpm_api.utils.utils_cmap import get_colormap_setting
from gpm_api.utils.geospatial import is_spatial_2D_field


def get_dataset_title(
    ds,
    time_idx=None,
    resolution="m",
    timezone="UTC",
    prefix_product=True,
    add_timestep=True,
):
    product = ds.attrs.get("gpm_api_product", "")
    if prefix_product:
        title_str = product + " " + ds.name
    else:
        title_str = ds.name
    # Make title in Capital Case
    title_str = " ".join([word[0].upper() + word[1:] for word in title_str.split(" ")])
    # Add time
    if add_timestep:
        # Parse time
        if ds["time"].size == 1:
            timestep = ds["time"].data
        else:
            if time_idx is None:
                timesteps = ds["time"].data
                timestep = timesteps[int(len(timesteps) / 2)]
            else:
                timestep = ds["time"].data[time_idx]
        time_str = np.datetime_as_string(timestep, unit=resolution, timezone=timezone)
        time_str = time_str.replace("T", " ").replace("Z", "")
        title_str = title_str + " (" + time_str + ")"
    return title_str


def check_is_spatial_2D_field(da):
    if not is_spatial_2D_field(da):
        raise ValueError("Expecting a 2D GPM field.")


# -----------------------------------------------------------------------------.
#################
#### Profile ####
#################
def xr_exclude_variables_without(ds, dim):
    # ds.filter.variables_without_dims()
    dataset_vars = list(ds)
    valid_vars = [var for var, da in ds.items() if dim in list(da.dims)]
    if len(valid_vars) == 0:
        raise ValueError(f"No dataset variables with dimension {dim}")
    ds_subset = ds[valid_vars]
    return ds_subset


def ensure_is_slice(slc):
    if isinstance(slc, slice):
        return slc
    else:
        if isinstance(slc, int):
            slc = slice(slc, slc + 1)
        elif isinstance(slc, (list, tuple)) and len(slc) == 1:
            slc = slice(slc[0], slc[0] + 1)
        elif isinstance(slc, np.ndarray) and slc.size == 1:
            slc = slice(slc.item(), slc.item() + 1)
        else:
            # TODO: check if continuous
            raise ValueError("Impossibile to convert to a slice object.")
    return slc


def get_slice_size(slc):
    if not isinstance(slc, slice):
        raise TypeError("Expecting slice object")
    size = slc.stop - slc.start
    return size


def optimize_transect_slices(
    obj, transect_slices, trim_threshold, variable=None, left_pad=0, right_pad=0
):
    # --------------------------------------------------------------------------.
    # Check variable
    # --> TODO: This can be removed under certain conditions
    # --> When trim_threshold become facultative
    # --> Left and right padding make sense only when trim_threshold is provided
    # TODO: Min_length, max_length arguments
    # --------------------------------------------------------------------------.
    if isinstance(obj, xr.Dataset):
        if variable is None:
            raise ValueError("If providing a xr.Dataset, 'variable' must be specified.")

    # --------------------------------------------------------------------------.
    # Check profile slice validity
    along_track_slice = ensure_is_slice(transect_slices["along_track"])
    cross_track_slice = ensure_is_slice(transect_slices["cross_track"])
    along_track_size = get_slice_size(along_track_slice)
    cross_track_size = get_slice_size(cross_track_slice)
    if along_track_size == 1 and cross_track_size == 1:
        raise ValueError("Both 'along_track' and 'cross_track' slices have size 1.")
    if along_track_size != 1 and cross_track_size != 1:
        raise ValueError(
            "Either 'along_track' or 'cross_track' must have a slice of size 1."
        )
    # --------------------------------------------------------------------------.
    # Get object transect
    obj_transect = obj.isel(transect_slices)

    # Retrieve transect dimension name
    if along_track_size == 1:
        transect_dim_name = "cross_track"
    else:
        transect_dim_name = "along_track"

    # Transpose transect dimension to first dimension
    obj_transect = obj_transect.transpose(transect_dim_name, ...)

    # --------------------------------------------------------------------------.
    # If xr.Dataset, get a DataArray
    if isinstance(obj_transect, xr.Dataset):
        obj_transect = obj_transect[variable]

    # --------------------------------------------------------------------------.
    # Get transect info
    len_transect = len(obj_transect[transect_dim_name])
    ndim_transect = obj_transect.ndim

    # --------------------------------------------------------------------------.
    # Identify transect extent including all pixel/profiles with value above threshold
    # - Transect line
    if ndim_transect == 1:
        idx_above_thr = np.where(obj_transect.data > trim_threshold)[0]
    else:  # 2D case (profile) or more ... (i.e. time or ensemble simulations)
        any_axis = tuple(np.arange(1, ndim_transect))
        idx_above_thr = np.where(
            np.any(obj_transect.data > trim_threshold, axis=any_axis)
        )[0]

    # --------------------------------------------------------------------------.
    # Check there are residual data along the transect
    if len(idx_above_thr) == 0:
        trim_variable = obj_transect.name
        raise ValueError(
            "No {trim_variable} value above trim_threshold={trim_threshold}. Try to decrease it."
        )
    valid_idx = np.unique(idx_above_thr[[0, -1]])

    # --------------------------------------------------------------------------.
    # Padding the transect extent
    if left_pad != 0:
        valid_idx[0] = max(0, valid_idx[0] - left_pad)
    if right_pad != 0:
        valid_idx[1] = min(len_transect - 1, valid_idx[1] + right_pad)

    # --------------------------------------------------------------------------.
    # TODO: Ensure minimum transect size

    # TODO: Ensure maximum transect size (centered around max?)

    # --------------------------------------------------------------------------.
    # Retrieve obj_transect slices
    if len(valid_idx) == 1:
        print(
            "Printing a single profile! To plot a longer profile transect increase `trim_threshold`."
        )
        transect_slice = slice(valid_idx, valid_idx + 1)
    else:
        transect_slice = slice(valid_idx[0], valid_idx[1] + 1)

    # --------------------------------------------------------------------------.
    # Update transect_slices
    original_slice = transect_slices[transect_dim_name]
    start = transect_slice.start + original_slice.start
    stop = transect_slice.stop + original_slice.start
    transect_slices[transect_dim_name] = slice(start, stop)

    ###-----------------------------------------------------------------------.
    # Return transect_slices
    return transect_slices


def get_transect_slices(
    obj, direction="cross_track", lon=None, lat=None, variable=None, transect_kwargs={}
):
    # TODO: add lon, lat argument ... --> variable and argmax used only if not specified
    # Variable need to be specified for xr.Dataset
    # -------------------------------------------------------------------------.
    # Checks
    # - Object type
    if not isinstance(obj, (xr.DataArray, xr.Dataset)):
        raise TypeError("Expecting xr.DataArray or xr.Dataset object.")
    # - Valid dimensions # --> TODO: check for each Datarray
    dims = set(list(obj.dims))
    required_dims = set(["along_track", "cross_track", "range"])
    if not dims.issuperset(required_dims):
        raise ValueError(f"Requires xarray object with dimensions {required_dims}")
    # - Verifiy valid input combination
    # --> If input xr.Dataset and variable, lat and lon not specified, raise Error
    if isinstance(obj, xr.Dataset):
        if lat is None and lon is None and variable is None:
            raise ValueError(
                "Need to provide 'variable' if passing a xr.Dataset and not specifying 'lat' / 'lon'."
            )

    # -------------------------------------------------------------------------.
    # If lon and lat are provided, derive center idx
    if lon is not None and lat is not None:
        # TODO: Get closest idx
        # idx_along_track
        # idx_cross_track
        raise NotImplementedError()

    # Else derive center locating the maximum intensity
    else:
        if isinstance(obj, xr.Dataset):
            if variable is None:
                raise ValueError(
                    "If providing a xr.Dataset, 'variable' must be specified."
                )
            da_variable = obj[variable].compute()
            obj[variable] = da_variable
        else:
            da_variable = obj.compute()

        dict_argmax = da_variable.argmax(da_variable.dims)
        idx_along_track = dict_argmax["along_track"]
        idx_cross_track = dict_argmax["cross_track"]

    # -------------------------------------------------------------------------.
    # Get transect slices based on direction
    if direction == "along_track":
        transect_slices = {"cross_track": int(idx_cross_track.data)}
        transect_slices["along_track"] = slice(0, len(obj["along_track"]))
    elif direction == "cross_track":
        transect_slices = {"along_track": int(idx_along_track.data)}
        transect_slices["cross_track"] = slice(0, len(obj["cross_track"]))

    else:  # TODO: longest, or most_max
        raise NotImplementedError()

    # -------------------------------------------------------------------------.
    # Optimize transect extent
    if len(transect_kwargs) != 0:
        transect_slices = optimize_transect_slices(
            obj, transect_slices, variable=variable, **transect_kwargs
        )
    # -------------------------------------------------------------------------.
    # Return transect slices
    return transect_slices


def plot_profile(da_profile, colorscale=None, ylim=None, ax=None):
    x_direction = da_profile["lon"].dims[0]
    # Retrieve title
    title = da_profile.gpm_api.title(
        time_idx=0, prefix_product=False, add_timestep=False
    )
    # Retrieve colormap configs
    plot_kwargs, cbar_kwargs, ticklabels = get_colormap_setting(colorscale)
    # Plot
    p = da_profile.plot.pcolormesh(
        x=x_direction, y="height", ax=ax, cbar_kwargs=cbar_kwargs, **plot_kwargs
    )
    p.axes.set_title(title)
    if ylim is not None:
        p.axes.set_ylim(ylim)
    return p


def plot_transect_line(ds, ax, color="black"):
    # Check is a profile (lon and lat are 1D coords)
    if len(ds["lon"].shape) != 1:
        raise ValueError("The xr.Dataset/xr.DataArray is not a profile.")

    # Retrieve start and end coordinates
    start_lonlat = (ds["lon"].data[0], ds["lat"].data[0])
    end_lonlat = (ds["lon"].data[-1], ds["lat"].data[-1])
    lon_startend = (start_lonlat[0], end_lonlat[0])
    lat_startend = (start_lonlat[1], end_lonlat[1])

    # Draw line
    ax.plot(lon_startend, lat_startend, transform=ccrs.Geodetic(), color=color)

    # Add transect left and right side (when plotting profile)
    g = Geod(ellps="WGS84")
    fwd_az, back_az, dist = g.inv(*start_lonlat, *end_lonlat, radians=False)
    lon_r, lat_r, _ = g.fwd(*start_lonlat, az=fwd_az, dist=dist + 50000)  # dist in m
    fwd_az, back_az, dist = g.inv(*end_lonlat, *start_lonlat, radians=False)
    lon_l, lat_l, _ = g.fwd(*end_lonlat, az=fwd_az, dist=dist + 50000)  # dist in m
    ax.text(lon_r, lat_r, "R")
    ax.text(lon_l, lat_l, "L")


# -----------------------------------------------------------------------------.
###############
#### Swath ####
###############
# TODO: adapt based on bin length (changing for each sensor) --> FUNCTION
# ax.plot(da['lon'][:, 0] + 0.0485, da['lat'][:,0],'--k')
# ax.plot(da['lon'][:,-1] - 0.0485, da['lat'][:,-1],'--k')


# -----------------------------------------------------------------------------.
##############
#### Map #####
##############
def _plot_swath_lines(ds, ax=None, **kwargs):
    # - 0.0485 to account for 2.5 km from pixel center
    # TODO: adapt based on bin length (changing for each sensor) --> FUNCTION
    lon = ds["lon"].transpose("cross_track", "along_track").data
    lat = ds["lat"].transpose("cross_track", "along_track").data
    ax.plot(lon[0, :] + 0.0485, lat[0, :], **kwargs)
    ax.plot(lon[-1, :] - 0.0485, lat[-1, :], **kwargs)


# def plot_swath(ds, ax=None):
#    plot swath polygon


def _plot_map_orbit(da, ax=None, add_colorbar=True):
    # Check inputs
    check_is_spatial_2D_field(da)
    # Initialize figure
    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(12, 10), dpi=100
        )

    # Get colorbar settings
    # TODO: to customize as function of da.name
    plot_kwargs, cbar_kwargs, ticklabels = get_colormap_setting("pysteps_mm/hr")

    # Add coastlines
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, facecolor=[0.9, 0.9, 0.9])
    ax.add_feature(cartopy.feature.OCEAN, alpha=0.6)
    ax.add_feature(cartopy.feature.STATES)

    # - Add swath lines
    _plot_swath_lines(da, ax=ax, linestyle="--", color="black")

    # - Add grid lines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.1,
        linestyle="-",
    )
    gl.top_labels = False  # gl.xlabels_top = False
    gl.right_labels = False  # gl.ylabels_right = False
    gl.xlines = True
    gl.ylines = True

    # - Add variable field with matplotlib
    p = ax.pcolormesh(
        da.lon.data,
        da.lat.data,
        da.data,
        transform=ccrs.PlateCarree(),
        **plot_kwargs,
    )

    # Add colorbar
    if add_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
        p.figure.add_axes(cax)
        cbar = plt.colorbar(p, cax=cax, ax=ax, **cbar_kwargs)
        _ = cbar.ax.set_yticklabels(ticklabels)

    # Return mappable
    return p


def plot_image(da, ax=None, add_colorbar=True, interpolation="nearest"):
    # Check inputs
    check_is_spatial_2D_field(da)

    # Initialize figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10), dpi=100)

    # Get colorbar settings
    # TODO: to customize as function of da.name
    plot_kwargs, cbar_kwargs, ticklabels = get_colormap_setting("pysteps_mm/hr")
    ##------------------------------------------------------------------------.
    ### Plot with xarray
    # --> BUG with colorbar: https://github.com/pydata/xarray/issues/7014
    # Add variable field with matplotlib
    p = da.plot.imshow(
        x="along_track",
        y="cross_track",
        ax=ax,
        interpolation=interpolation,
        add_colorbar=add_colorbar,
        cbar_kwargs=cbar_kwargs,
        **plot_kwargs,
    )
    if add_colorbar:
        p.colorbar.ax.set_yticklabels(ticklabels)
    ##------------------------------------------------------------------------.
    ### Plot with matplotlib
    # --> TODO: set axis proportion in a meaningful way ...
    # arr = da.transpose("cross_track", "along_track").data.compute()
    # p  = ax.imshow(arr,
    #                 **plot_kwargs)
    # # Add colorbar
    # if add_colorbar:
    #     divider = make_axes_locatable(ax)
    #     cax = divider.new_horizontal(size="5%", pad=0.1, axes_class=plt.Axes)
    #     p.figure.add_axes(cax)
    #     cbar = plt.colorbar(p, cax=cax, ax=ax, **cbar_kwargs)
    #     _ = cbar.ax.set_yticklabels(ticklabels)

    # Return mappable
    return p


def _plot_map(da, ax=None, add_colorbar=True):
    # Plot orbit
    if len(da["lon"].shape):
        p = _plot_map_orbit(da=da, ax=ax, add_colorbar=add_colorbar)
    # Plot grid
    else:
        raise NotImplementedError
    return p


####--------------------------------------------------------------------------.
#################
#### Patches ####
#################
def check_is_xarray(x):
    if not isinstance(x, (xr.DataArray, xr.Dataset)):
        raise TypeError("Expecting a xr.Dataset or xr.DataArray.")


def check_is_xarray_dataarray(x):
    if not isinstance(x, xr.DataArray):
        raise TypeError("Expecting a xr.DataArray.")


def check_is_xarray_dataset(x):
    if not isinstance(x, xr.Dataset):
        raise TypeError("Expecting a xr.Dataset.")


def plot_patches(
    data_array,
    min_value_threshold=-np.inf,
    max_value_threshold=np.inf,
    min_area_threshold=1,
    max_area_threshold=np.inf,
    footprint_buffer=None,
    sort_by="area",
    sort_decreasing=True,
    label_name="label",
    n_patches=None,
    patch_margin=None,
    interpolation="nearest",
):

    from gpm_api.patch.generator import get_da_patch_generator

    check_is_xarray_dataarray(data_array)

    # Define generator
    gpm_da_patch_gen = get_da_patch_generator(
        data_array=data_array,
        min_value_threshold=min_value_threshold,
        max_value_threshold=max_value_threshold,
        min_area_threshold=min_area_threshold,
        max_area_threshold=max_area_threshold,
        footprint_buffer=footprint_buffer,
        sort_by=sort_by,
        sort_decreasing=sort_decreasing,
        label_name=label_name,
        n_patches=n_patches,
        patch_margin=patch_margin,
    )
    # Plot patches
    for da in gpm_da_patch_gen:
        plot_image(da, interpolation=interpolation, add_colorbar=True)
        plt.show()

    return None
