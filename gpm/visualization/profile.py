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
"""This module contains functions to visualize radar transects."""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr

from gpm import get_plot_kwargs
from gpm.checks import check_is_transect
from gpm.utils.slices import ensure_is_slice, get_slice_size
from gpm.visualization.plot import (
    _plot_xr_pcolormesh,
    preprocess_figure_args,
)


def _optimize_transect_slices(
    xr_obj,
    transect_slices,
    trim_threshold,
    variable=None,
    left_pad=0,
    right_pad=0,
):
    # --------------------------------------------------------------------------.
    # Check variable
    # --> TODO: This can be removed under certain conditions
    # --> When trim_threshold become facultative
    # --> Left and right padding make sense only when trim_threshold is provided
    # TODO: Min_length, max_length arguments
    # --------------------------------------------------------------------------.
    if isinstance(xr_obj, xr.Dataset) and variable is None:
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
    # Get xr_object transect
    xr_obj_transect = xr_obj.isel(transect_slices)

    # Retrieve transect dimension name
    transect_dim_name = "cross_track" if along_track_size == 1 else "along_track"

    # Transpose transect dimension to first dimension
    xr_obj_transect = xr_obj_transect.transpose(transect_dim_name, ...)

    # --------------------------------------------------------------------------.
    # If xr.Dataset, get a DataArray
    if isinstance(xr_obj_transect, xr.Dataset):
        xr_obj_transect = xr_obj_transect[variable]

    # --------------------------------------------------------------------------.
    # Get transect info
    len_transect = len(xr_obj_transect[transect_dim_name])
    ndim_transect = xr_obj_transect.ndim

    # --------------------------------------------------------------------------.
    # Identify transect extent including all pixel/profiles with value above threshold
    # - Transect line
    if ndim_transect == 1:
        idx_above_thr = np.where(xr_obj_transect.data > trim_threshold)[0]
    else:  # 2D case (profile) or more ... (i.e. time or ensemble simulations)
        any_axis = tuple(np.arange(1, ndim_transect))
        idx_above_thr = np.where(np.any(xr_obj_transect.data > trim_threshold, axis=any_axis))[0]

    # --------------------------------------------------------------------------.
    # Check there are residual data along the transect
    if len(idx_above_thr) == 0:
        raise ValueError(
            "No {trim_variable} value above trim_threshold={trim_threshold}. Try to decrease it.",
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
    # Retrieve xr_obj_transect slices
    if len(valid_idx) == 1:
        print(
            "Printing a single profile! To plot a longer profile transect increase `trim_threshold`.",
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
    xr_obj,
    direction="cross_track",
    lon=None,
    lat=None,
    variable=None,
    transect_kwargs={},
):
    """Define transect isel dictionary slices.

    If lon and lat are provided, it returns the transect closest to such point.
    Otherwise, it returns the transect passing through the maximum value of 'variable'

    Parameters
    ----------
    xr_obj : TYPE
        DESCRIPTION.
    direction : TYPE, optional
        DESCRIPTION. The default is "cross_track".
    lon : TYPE, optional
        DESCRIPTION. The default is ``None``.
    lat : TYPE, optional
        DESCRIPTION. The default is ``None``.
    variable : TYPE, optional
        DESCRIPTION. The default is ``None``.
    transect_kwargs : TYPE, optional
        DESCRIPTION. The default is ``None``.

    Returns
    -------
    transect_slices : TYPE
        DESCRIPTION.

    """
    # TODO: add lon, lat argument ... --> variable and argmax used only if not specified
    # TODO: implement diagonal transect ?
    # TODO: enable a curvilinear track / trajectory

    # Variable need to be specified for xr.Dataset
    # -------------------------------------------------------------------------.
    # Checks
    # - xr_object type
    if not isinstance(xr_obj, (xr.DataArray, xr.Dataset)):
        raise TypeError("Expecting xr.DataArray or xr.Dataset xr_object.")
    # - Valid dimensions # --> TODO: check for each Datarray
    dims = set(xr_obj.dims)
    required_dims = {"along_track", "cross_track", "range"}
    if not dims.issuperset(required_dims):
        raise ValueError(f"Requires xarray xr_object with dimensions {required_dims}")
    # - Verify valid input combination
    # --> If input xr.Dataset and variable, lat and lon not specified, raise Error
    if isinstance(xr_obj, xr.Dataset) and lat is None and lon is None and variable is None:
        raise ValueError(
            "Need to provide 'variable' if passing a xr.Dataset and not specifying 'lat' / 'lon'.",
        )

    # -------------------------------------------------------------------------.
    # If lon and lat are provided, derive center idx
    if lon is not None and lat is not None:
        # TODO: Get closest idx
        # idx_along_track
        # idx_cross_track
        raise NotImplementedError

    # Else derive center locating the maximum intensity
    if isinstance(xr_obj, xr.Dataset):
        if variable is None:
            raise ValueError("If providing a xr.Dataset, 'variable' must be specified.")
        da_variable = xr_obj[variable].compute()
        xr_obj[variable] = da_variable
    else:
        da_variable = xr_obj.compute()

    dict_argmax = da_variable.argmax(da_variable.dims)
    idx_along_track = dict_argmax["along_track"]
    idx_cross_track = dict_argmax["cross_track"]

    # -------------------------------------------------------------------------.
    # Get transect slices based on direction
    if direction == "along_track":
        transect_slices = {"cross_track": int(idx_cross_track.data)}
        transect_slices["along_track"] = slice(0, len(xr_obj["along_track"]))
    elif direction == "cross_track":
        transect_slices = {"along_track": int(idx_along_track.data)}
        transect_slices["cross_track"] = slice(0, len(xr_obj["cross_track"]))
    else:  # TODO: longest, or most_max
        raise NotImplementedError

    # -------------------------------------------------------------------------.
    # Optimize transect extent
    if len(transect_kwargs) != 0:
        transect_slices = _optimize_transect_slices(
            xr_obj,
            transect_slices,
            variable=variable,
            **transect_kwargs,
        )
    # -------------------------------------------------------------------------.
    # Return transect slices
    return transect_slices


def select_transect(
    xr_obj,
    direction="cross_track",
    lon=None,
    lat=None,
    variable=None,
    transect_kwargs={},
    keep_only_valid_variables=True,
):
    # Identify transect isel_dict
    transect_slices = get_transect_slices(
        xr_obj,
        direction=direction,
        variable=variable,
        lon=lon,
        lat=lat,
        transect_kwargs=transect_kwargs,
    )
    # Extract the transect dataset
    if isinstance(xr_obj, xr.Dataset) and keep_only_valid_variables:
        xr_obj = xr_obj.gpm.select_spatial_3d_variables()
    return xr_obj.isel(transect_slices)


def plot_transect_line(
    ds,
    ax,
    add_direction=True,
    text_kwargs={},
    line_kwargs={},
    **common_kwargs,
):
    # Check is a profile (lon and lat are 1D coords)
    if len(ds["lon"].shape) != 1:
        raise ValueError("The xr.Dataset/xr.DataArray is not a profile.")

    # Retrieve start and end coordinates
    start_lonlat = (ds["lon"].data[0], ds["lat"].data[0])
    end_lonlat = (ds["lon"].data[-1], ds["lat"].data[-1])
    lon_startend = (start_lonlat[0], end_lonlat[0])
    lat_startend = (start_lonlat[1], end_lonlat[1])

    # Draw line
    ax.plot(lon_startend, lat_startend, transform=ccrs.Geodetic(), **line_kwargs, **common_kwargs)

    # Add transect left and right side (when plotting transect)
    if add_direction:
        g = pyproj.Geod(ellps="WGS84")
        fwd_az, back_az, dist = g.inv(*start_lonlat, *end_lonlat, radians=False)
        lon_r, lat_r, _ = g.fwd(*start_lonlat, az=fwd_az, dist=dist + 50000)  # dist in m
        fwd_az, back_az, dist = g.inv(*end_lonlat, *start_lonlat, radians=False)
        lon_l, lat_l, _ = g.fwd(*end_lonlat, az=fwd_az, dist=dist + 50000)  # dist in m
        ax.text(lon_r, lat_r, "R", **text_kwargs, **common_kwargs)
        ax.text(lon_l, lat_l, "L", **text_kwargs, **common_kwargs)


def plot_transect(
    da,
    ax=None,
    add_colorbar=True,
    zoom=True,
    fig_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot GPM transect."""
    # - Check inputs
    check_is_transect(da)
    fig_kwargs = preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs)

    # - Initialize figure
    if ax is None:
        _, ax = plt.subplots(**fig_kwargs)

    # - If not specified, retrieve/update plot_kwargs and cbar_kwargs as function of product name
    plot_kwargs, cbar_kwargs = get_plot_kwargs(
        name=da.name,
        user_plot_kwargs=plot_kwargs,
        user_cbar_kwargs=cbar_kwargs,
    )
    # - Select only vertical regions with data
    if zoom:
        da = da.gpm.slice_range_with_valid_data()

    # - Define xlabel
    spatial_dim = da.gpm.spatial_dimensions[0]
    xlabel_dicts = {"cross_track": "Cross-Track", "along_track": "Along-Track"}
    xlabel = xlabel_dicts[spatial_dim]

    # - Plot with xarray
    x_direction = da["lon"].dims[0]
    p = _plot_xr_pcolormesh(
        ax=ax,
        da=da,
        x=x_direction,
        y="height",
        add_colorbar=add_colorbar,
        plot_kwargs=plot_kwargs,
        cbar_kwargs=cbar_kwargs,
    )
    p.axes.set_xlabel(xlabel)
    p.axes.set_ylabel("Height [m]")
    # - Return mappable
    return p
