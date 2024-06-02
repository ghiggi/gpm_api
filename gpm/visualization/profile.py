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
from pyproj import Geod

from gpm import get_plot_kwargs
from gpm.checks import check_has_cross_track_dim, check_is_transect
from gpm.utils.checks import check_contiguous_scans
from gpm.utils.slices import ensure_is_slice, get_slice_size
from gpm.utils.xarray import get_dimensions_without
from gpm.visualization.plot import (
    check_object_format,
    get_valid_pcolormesh_inputs,
    initialize_cartopy_plot,
    plot_xr_imshow,
    plot_xr_pcolormesh,
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
        raise ValueError("If providing a xarray.Dataset, 'variable' must be specified.")

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
    # If xarray.Dataset, get a DataArray
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
            "Printing a single profile! To plot a longer profile transect increase 'trim_threshold'.",
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
    xr_obj : `xarray.DataArray` or `xarray.Dataset`
        DESCRIPTION.
    direction : str, optional
        DESCRIPTION. The default is "cross_track".
    lon : float, optional
        DESCRIPTION. The default is ``None``.
    lat : float, optional
        DESCRIPTION. The default is ``None``.
    variable : str, optional
        DESCRIPTION. The default is ``None``.
    transect_kwargs : dict, optional
        DESCRIPTION. The default is ``None``.

    Returns
    -------
    transect_slices : dict
        DESCRIPTION.

    """
    # TODO: add lon, lat argument ... --> variable and argmax used only if not specified
    # TODO: implement diagonal transect ?
    # TODO: enable a curvilinear track / trajectory

    # Variable need to be specified for xarray.Dataset
    # -------------------------------------------------------------------------.
    # Checks
    # - xr_object type
    if not isinstance(xr_obj, (xr.DataArray, xr.Dataset)):
        raise TypeError("Expecting xarray.DataArray or xarray.Dataset object.")
    # - Valid dimensions # --> TODO: check for each Datarray
    dims = set(xr_obj.dims)
    required_dims = {"along_track", "cross_track", "range"}
    if not dims.issuperset(required_dims):
        raise ValueError(f"Requires xarray xr_object with dimensions {required_dims}")
    # - Verify valid input combination
    # --> If input xarray.Dataset and variable, lat and lon not specified, raise Error
    if isinstance(xr_obj, xr.Dataset) and lat is None and lon is None and variable is None:
        raise ValueError(
            "Need to provide 'variable' if passing a xarray.Dataset and not specifying 'lat' / 'lon'.",
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
            raise ValueError("If providing a xarray.Dataset, 'variable' must be specified.")
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


####----------------------------------------------------------------------------------------------------------------.
#######################
#### Plot Transect ####
#######################


def get_cross_track_horizontal_distance(xr_obj):
    """Retrieve the horizontal_distance from the nadir.

    Requires a transect with cross_track dimension !
    """
    check_is_transect(xr_obj)
    check_has_cross_track_dim(xr_obj)

    # Retrieve required DataArrays
    lons = xr_obj["lon"].data
    lats = xr_obj["lat"].data
    idx = np.where(xr_obj["gpm_cross_track_id"] == 24)[0].item()
    start_lon = xr_obj["lon"].isel(cross_track=idx).data
    start_lat = xr_obj["lat"].isel(cross_track=idx).data

    geod = Geod(ellps="WGS84")
    distances = np.array([geod.inv(start_lon, start_lat, lon, lat)[2] for lon, lat in zip(lons, lats)])
    distances[:idx] = -distances[:idx]
    da_dist = xr.DataArray(distances, dims="cross_track")
    return da_dist


def _ensure_valid_pcolormesh_coords(da, x, y, rgb):
    # Get 2D x and y coordinates
    da = da.copy()
    da_template = da.isel({da.dims[-1]: 0}) if rgb else da
    da_x = da[x].broadcast_like(da_template)
    da_y = da[y].broadcast_like(da_template)
    # Get valid coordinates
    x_coord, y_coord, data = get_valid_pcolormesh_inputs(
        x=da_x.data,
        y=da_y.data,
        data=da.data,
        rgb=rgb,
        mask_data=True,
    )
    # Mask data
    da.data = data
    # Set back validated coordinates
    # - If x or y are dimension names without coordinates, nothing to be done
    if x in da.coords:
        if da[x].ndim == 1:
            dim_name = list(da[x].dims)[0]
            da_x.data = x_coord
            da_x_values = da_x.isel({dim: 0 for dim in get_dimensions_without(da_x, da[x].dims)}).data
            da = da.assign_coords({x: (dim_name, da_x_values)})
        else:
            da[x].data = x_coord
    if y in da.coords:
        if da[y].ndim == 1:
            dim_name = list(da[y].dims)[0]
            da_y.data = y_coord
            da_y_values = da_y.isel({dim: 0 for dim in get_dimensions_without(da_y, da[y].dims)}).data
            da = da.assign_coords({y: (dim_name, da_y_values)})
        else:
            da[y].data = y_coord
    return da


def _get_x_axis_options(da, x):
    # Define xlabels
    xlabel_dicts = {
        "cross_track": "Cross-Track",
        "along_track": "Along-Track",
        "horizontal_distance": "Distance from nadir [m]",
        "horizontal_distance_km": "Distance from nadir [km]",
        "lon": "Longitude [°]",
        "lat": "Latitude [°]",
    }

    # Define additional coordinates on the fly if asked
    if x in ["horizontal_distance", "horizontal_distance_km"]:
        scale_factor = 1000 if x == "horizontal_distance_km" else 1
        da_distance = get_cross_track_horizontal_distance(da) / scale_factor
        da = da.assign_coords({x: da_distance})
    # If x specified, check valid coordinate
    if x is not None:
        if x not in list(set(da.dims) | set(da.coords)):
            raise ValueError(f"'{x}' is not a DataArray coordinate. Specify a valid 'x' or compute '{x}'.")
    else:  # set default (cross_track or along_track)
        x = get_dimensions_without(da, da.gpm.vertical_dimension)[0]  # the dimension which is not vertical
    # Define xlabel
    xlabel = xlabel_dicts.get(x, x.title())
    # Return x, label and DataArray
    return x, xlabel, da


def _get_y_axis_options(da, y, origin):
    # Define ylabels
    # - Order of keys is the preferred y
    ylabel_dicts = {
        "height": "Height [m]",
        "height_km": "Height [km]",
        "range": "Range Index",  # Start at 1
        "gpm_range_id": "Range Index",  # Start at 0
        "range_distance_from_satellite": "Range Distance From Satellite [m]",
        "range_distance_from_ellipsoid": "Range Distance From Ellipsoid [m]",
        "range_distance_from_satellite_km": "Range Distance From Satellite [km]",
        "range_distance_from_ellipsoid_km": "Range Distance From Ellipsoid [km]",
    }

    # Check y and define default if None
    y = _get_default_y(y=y, da=da, possible_defaults=list(ylabel_dicts))

    # Define additional coordinates on the fly
    if y in ["range_distance_from_satellite_km", "range_distance_from_ellipsoid_km", "height_km"]:
        da = da.assign_coords({y: da[y[:-3]] / 1000})

    # Define origin for 1D y coordinate
    if origin is None:
        origin = "lower" if y in ["height", "height_km"] else "upper"  # range, gpm_range_id

    # Define ylabel
    ylabel = ylabel_dicts.get(y, y.title())

    # Return x, label and DataArray
    return y, ylabel, da, origin


def _get_default_y(y, da, possible_defaults):
    """Define default y."""
    # Define default "y" (at least "range" is available since check_is_transect() called before
    if y is None:
        candidate_y = list(set(da.dims) | set(da.coords))
        expected_y = np.array(possible_defaults)
        available_y = expected_y[np.isin(expected_y, candidate_y)]
        return available_y[0]
    if y in ["range_distance_from_satellite_km", "range_distance_from_ellipsoid_km", "height_km"]:
        if y[:-3] not in (da.coords):
            raise ValueError(f"'{y[:-3]}' is not a DataArray coordinate. Specify a valid 'y' or compute {y[:-3]}.")
        return y
    if y not in list(set(da.dims) | set(da.coords)):
        raise ValueError(f"'{y}' is not a DataArray coordinate. Specify a valid 'y' or compute '{y}'.")
    return y


def plot_transect(
    da,
    x=None,
    y=None,
    ax=None,
    add_colorbar=True,
    zoom=True,
    interpolation="nearest",
    fig_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot GPM transect.

    If RGB DataArray, all other plot_kwargs are ignored !
    """
    # Check inputs
    da = check_object_format(da, plot_kwargs=plot_kwargs, check_function=check_is_transect, strict=True)

    # - Check for contiguous along-track scans
    if "along_track" in da.dims:
        check_contiguous_scans(da)

    # - Initialize figure
    fig_kwargs = preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs)
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
        da = da.gpm.subset_range_with_valid_data()

    # - Check x and define x label
    x, xlabel, da = _get_x_axis_options(da, x=x)

    # - Check y and define ylabel
    y, ylabel, da, origin = _get_y_axis_options(da, y=y, origin=plot_kwargs.get("origin", None))

    # - Plot with xarray
    if da[y].ndim == 1 and da[x].ndim == 1:
        plot_kwargs["origin"] = origin
        p = plot_xr_imshow(
            ax=ax,
            da=da,
            x=x,
            y=y,
            interpolation=interpolation,
            add_colorbar=add_colorbar,
            cbar_kwargs=cbar_kwargs,
            visible_colorbar=True,
            **plot_kwargs,
        )
    else:

        # Infill invalid coordinates and add mask to data if necessary
        # - This occur when extracting L2 dataset from L1B and use y = "height/rangeDist"
        da = _ensure_valid_pcolormesh_coords(da, x=x, y=y, rgb=plot_kwargs.get("rgb", False))

        # Plot transect
        p = plot_xr_pcolormesh(
            ax=ax,
            da=da,
            x=x,
            y=y,
            add_colorbar=add_colorbar,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    p.axes.set_xlabel(xlabel)
    p.axes.set_ylabel(ylabel)
    # - Return mappable
    return p


def plot_transect_line(
    xr_obj,
    ax=None,
    add_direction=True,
    add_background=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    text_kwargs=None,
    line_kwargs=None,
    **common_kwargs,
):
    # - Check is transect
    check_is_transect(xr_obj, strict=False)  # allow i.e. radar_frequency

    # - Set defaults
    text_kwargs = {} if text_kwargs is None else text_kwargs
    line_kwargs = {} if line_kwargs is None else line_kwargs

    # - Initialize figure if necessary
    ax = initialize_cartopy_plot(
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
        add_background=add_background,
    )

    # Retrieve start and end coordinates
    start_lonlat = (xr_obj["lon"].data[0], xr_obj["lat"].data[0])
    end_lonlat = (xr_obj["lon"].data[-1], xr_obj["lat"].data[-1])
    lon_startend = (start_lonlat[0], end_lonlat[0])
    lat_startend = (start_lonlat[1], end_lonlat[1])

    # Draw line
    p = ax.plot(lon_startend, lat_startend, transform=ccrs.Geodetic(), **line_kwargs, **common_kwargs)

    # Add transect left and right side (when plotting transect)
    if add_direction:
        g = pyproj.Geod(ellps="WGS84")
        fwd_az, back_az, dist = g.inv(*start_lonlat, *end_lonlat, radians=False)
        lon_r, lat_r, _ = g.fwd(*start_lonlat, az=fwd_az, dist=dist + 50000)  # dist in m
        fwd_az, back_az, dist = g.inv(*end_lonlat, *start_lonlat, radians=False)
        lon_l, lat_l, _ = g.fwd(*end_lonlat, az=fwd_az, dist=dist + 50000)  # dist in m
        ax.text(lon_r, lat_r, "R", **text_kwargs, **common_kwargs)
        ax.text(lon_l, lat_l, "L", **text_kwargs, **common_kwargs)

    # - Return mappable
    return p[0]
