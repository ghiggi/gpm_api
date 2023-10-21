#!/usr/bin/env python3
"""
Created on Tue Jul 18 17:08:55 2023

@author: ghiggi
"""
import numpy as np
import pandas as pd

from gpm_api.dataset.attrs import decode_string


def _get_orbit_scan_time(dt, scan_mode):
    """Return timesteps array."""
    ds = dt[scan_mode]["ScanTime"].compute()
    df = pd.DataFrame(
        {
            "year": ds["Year"].data,
            "month": ds["Month"].data,
            "day": ds["DayOfMonth"].data,
            "hour": ds["Hour"].data,
            "minute": ds["Minute"].data,
            "second": ds["Second"].data,
        }
    )
    return pd.to_datetime(df).to_numpy()


def get_orbit_coords(dt, scan_mode):
    """Get coordinates from Orbit objects."""
    attrs = decode_string(dt.attrs["FileHeader"])
    granule_id = attrs["GranuleNumber"]

    lon = np.asanyarray(dt[scan_mode]["Longitude"].data)
    lat = np.asanyarray(dt[scan_mode]["Latitude"].data)
    # lst = dt[scan_mode]["sunLocalTime"].data.compute()
    time = _get_orbit_scan_time(dt, scan_mode)

    n_along_track, n_cross_track = lon.shape
    granule_id = np.repeat(granule_id, n_along_track)
    along_track_id = np.arange(n_along_track)
    cross_track_id = np.arange(n_cross_track)
    gpm_id = [str(g) + "-" + str(z) for g, z in zip(granule_id, along_track_id)]
    coords = {
        "lon": (["along_track", "cross_track"], lon),
        "lat": (["along_track", "cross_track"], lat),
        "time": (["along_track"], time),
        "gpm_id": (["along_track"], gpm_id),
        "gpm_granule_id": (["along_track"], granule_id),
        "gpm_cross_track_id": (["cross_track"], cross_track_id),
        "gpm_along_track_id": (["along_track"], along_track_id),
    }
    return coords


def get_grid_coords(dt, scan_mode):
    """Get coordinates from Grid objects.

    IMERG and GRID products does not have GranuleNumber!
    """
    attrs = decode_string(dt.attrs["FileHeader"])
    lon = np.asanyarray(dt[scan_mode]["lon"])
    lat = np.asanyarray(dt[scan_mode]["lat"])
    time = attrs["StartGranuleDateTime"][:-1]
    time = np.array(
        np.datetime64(time) + np.timedelta64(30, "m"), ndmin=1
    )  # TODO: document why + 30 min
    coords = {
        "time": time,
        "lon": lon,
        "lat": lat,
    }
    return coords


def get_coords(dt, scan_mode):
    """Get coordinates from GPM objects."""
    coords = (
        get_grid_coords(dt, scan_mode) if scan_mode == "Grid" else get_orbit_coords(dt, scan_mode)
    )
    return coords


def _subset_dict_by_dataset(ds, dictionary):
    """Select the relevant dictionary key for a given dataset."""
    # Get dataset coords and variables
    names = list(ds.coords) + list(ds.data_vars)
    # Select valid keys
    valid_keys = [key for key in names if key in dictionary]
    dictionary = {k: dictionary[k] for k in valid_keys}
    return dictionary


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
    attrs_dict["time"] = {"standard_name": "time", "coverage_content_type": "coordinate"}

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
    attrs_dict = _subset_dict_by_dataset(ds, attrs_dict)
    return attrs_dict


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
