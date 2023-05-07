#!/usr/bin/env python3
"""
Created on Sat Dec 10 18:44:25 2022

@author: ghiggi
"""
import cartopy.crs as ccrs
import numpy as np
import pyproj
import xarray as xr

from gpm_api.utils.slices import ensure_is_slice, get_slice_size
from gpm_api.utils.utils_cmap import get_colormap_setting


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
    if isinstance(obj, xr.Dataset) and variable is None:
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
        raise ValueError("Either 'along_track' or 'cross_track' must have a slice of size 1.")
    # --------------------------------------------------------------------------.
    # Get object transect
    obj_transect = obj.isel(transect_slices)

    # Retrieve transect dimension name
    transect_dim_name = "cross_track" if along_track_size == 1 else "along_track"

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
        idx_above_thr = np.where(np.any(obj_transect.data > trim_threshold, axis=any_axis))[0]

    # --------------------------------------------------------------------------.
    # Check there are residual data along the transect
    if len(idx_above_thr) == 0:
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
    if isinstance(obj, xr.Dataset) and lat is None and lon is None and variable is None:
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
                raise ValueError("If providing a xr.Dataset, 'variable' must be specified.")
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
    title = da_profile.gpm_api.title(time_idx=0, prefix_product=False, add_timestep=False)
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
    g = pyproj.Geod(ellps="WGS84")
    fwd_az, back_az, dist = g.inv(*start_lonlat, *end_lonlat, radians=False)
    lon_r, lat_r, _ = g.fwd(*start_lonlat, az=fwd_az, dist=dist + 50000)  # dist in m
    fwd_az, back_az, dist = g.inv(*end_lonlat, *start_lonlat, radians=False)
    lon_l, lat_l, _ = g.fwd(*end_lonlat, az=fwd_az, dist=dist + 50000)  # dist in m
    ax.text(lon_r, lat_r, "R")
    ax.text(lon_l, lat_l, "L")
