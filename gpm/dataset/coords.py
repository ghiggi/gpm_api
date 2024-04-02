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
"""This module contains functions to extract the coordinates from GPM files."""
import numpy as np
import pandas as pd
import xarray as xr

from gpm.dataset.attrs import decode_string


def _get_orbit_scan_time(dt, scan_mode):
    """Return timesteps array.

    dt must not decode_cf=True for this to work.
    """
    ds = dt[scan_mode]["ScanTime"].compute()
    dict_time = {
        "year": ds["Year"].data,
        "month": ds["Month"].data,
        "day": ds["DayOfMonth"].data,
        "hour": ds["Hour"].data,
        "minute": ds["Minute"].data,
        "second": ds["Second"].data,
    }
    return pd.to_datetime(dict_time).to_numpy()


def get_orbit_coords(dt, scan_mode):
    """Get coordinates from Orbit objects."""
    attrs = decode_string(dt.attrs["FileHeader"])
    granule_id = attrs["GranuleNumber"]

    ds = dt[scan_mode]
    time = _get_orbit_scan_time(dt, scan_mode)

    lon = ds["Longitude"].data
    lat = ds["Latitude"].data
    n_along_track, n_cross_track = lon.shape
    granule_id = np.repeat(granule_id, n_along_track)
    along_track_id = np.arange(n_along_track)
    cross_track_id = np.arange(n_cross_track)
    gpm_id = [str(g) + "-" + str(z) for g, z in zip(granule_id, along_track_id)]

    return {
        "lon": xr.DataArray(lon, dims=["along_track", "cross_track"]),
        "lat": xr.DataArray(lat, dims=["along_track", "cross_track"]),
        "time": xr.DataArray(time, dims="along_track"),
        "gpm_id": xr.DataArray(gpm_id, dims="along_track"),
        "gpm_granule_id": xr.DataArray(granule_id, dims="along_track"),
        "gpm_cross_track_id": xr.DataArray(cross_track_id, dims="cross_track"),
        "gpm_along_track_id": xr.DataArray(along_track_id, dims="along_track"),
    }


def get_time_delta_from_time_interval(time_interval):
    time_interval_dict = {
        "HALF_HOUR": np.timedelta64(30, "m"),
        "DAY": np.timedelta64(24, "h"),
    }
    return time_interval_dict[time_interval]


def get_grid_coords(dt, scan_mode):
    """Get coordinates from Grid objects.

    Set 'time' to the end of the accumulation period.
    Example: IMERG provide the average rain rate (mm/hr) over the half-hour period

    NOTE: IMERG and GRID products does not have GranuleNumber!
    """
    attrs = decode_string(dt.attrs["FileHeader"])
    start_time = attrs["StartGranuleDateTime"][:-1]  # 2016-03-09T10:30:00.000Z
    # end_time = attrs["StopGranuleDateTime"][:-1]    # 2003-05-01T23:59:59.999Z
    time_interval = attrs["TimeInterval"]
    time_delta = get_time_delta_from_time_interval(time_interval)
    start_time = np.array([start_time]).astype("M8[ns]")
    end_time = start_time + time_delta

    # Define time coordinate
    time = xr.DataArray(end_time, dims="time")
    time.attrs = {
        "axis": "T",
        "bounds": "time_bnds",
        "standard_name": "time",
        "description": "End time of the accumulation period",
    }
    # Define time bounds
    time_bnds = np.concatenate((start_time, end_time)).reshape(1, 2)
    time_bnds = xr.DataArray(time_bnds, dims=("time", "nv"))

    # Define dictionary with coordinates (DataArray)
    return {
        "time": time,
        "lon": dt[scan_mode]["lon"],
        "lat": dt[scan_mode]["lat"],
        "time_bnds": time_bnds,
    }


def get_coords(dt, scan_mode):
    """Get coordinates from GPM objects."""
    return get_grid_coords(dt, scan_mode) if scan_mode == "Grid" else get_orbit_coords(dt, scan_mode)


def _subset_dict_by_dataset(ds, dictionary):
    """Select the relevant dictionary key for a given dataset."""
    # Get dataset coords and variables
    names = list(ds.coords) + list(ds.data_vars)
    # Select valid keys
    valid_keys = [key for key in names if key in dictionary]
    return {k: dictionary[k] for k in valid_keys}


def get_coords_attrs_dict(ds):
    """Return relevant GPM coordinates attributes."""
    attrs_dict = {}
    # Define attributes for latitude and longitude
    attrs_dict["lat"] = {
        "name": "latitude",
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
        "valid_min": -90.0,
        "valid_max": 90.0,
        "comment": "Geographical coordinates, WGS84 datum",
        "coverage_content_type": "coordinate",
    }
    attrs_dict["lon"] = {
        "name": "longitude",
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
        "valid_min": -180.0,
        "valid_max": 180.0,
        "comment": "Geographical coordinates, WGS84 datum",
        "coverage_content_type": "coordinate",
    }

    attrs_dict["gpm_granule_id"] = {
        "long_name": "GPM Granule ID",
        "description": "ID number of the GPM Granule",
        "coverage_content_type": "auxiliaryInformation",
    }

    # Define general attributes for time coordinates
    attrs_dict["time"] = {
        "standard_name": "time",
        "coverage_content_type": "coordinate",
        "axis": "T",
    }

    # Add description of GPM ORBIT coordinates
    attrs_dict["gpm_cross_track_id"] = {
        "long_name": "Cross-Track ID",
        "description": "Cross-Track ID.",
        "coverage_content_type": "auxiliaryInformation",
    }

    attrs_dict["gpm_along_track_id"] = {
        "long_name": "Along-Track ID",
        "description": "Along-Track ID.",
        "coverage_content_type": "auxiliaryInformation",
    }

    attrs_dict["gpm_id"] = {
        "long_name": "Scan ID",
        "description": "Scan ID. Format: '{gpm_granule_id}-{gpm_along_track_id}'",
        "coverage_content_type": "auxiliaryInformation",
    }

    # Select required attributes
    return _subset_dict_by_dataset(ds, attrs_dict)


def _set_attrs_dict(ds, attrs_dict):
    """Set dataset attributes for each attrs_dict key."""
    for var in attrs_dict:
        ds[var].attrs.update(attrs_dict[var])


def set_coords_attrs(ds):
    """Set dataset coordinate attributes."""
    # Get attributes dictionary
    attrs_dict = get_coords_attrs_dict(ds)
    # Set attributes
    _set_attrs_dict(ds, attrs_dict)
    return ds
