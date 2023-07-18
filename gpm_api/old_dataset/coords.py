#!/usr/bin/env python3
"""
Created on Thu Jun 22 14:57:11 2023

@author: ghiggi
"""
import numpy as np
import pandas as pd

from gpm_api.utils.utils_HDF5 import hdf5_file_attrs


def _parse_hdf_gpm_scantime(h):
    """Return timesteps array."""
    df = pd.DataFrame(
        {
            "year": h["Year"][:],
            "month": h["Month"][:],
            "day": h["DayOfMonth"][:],
            "hour": h["Hour"][:],
            "minute": h["Minute"][:],
            "second": h["Second"][:],
        }
    )
    return pd.to_datetime(df).to_numpy()


def get_orbit_coords(hdf, scan_mode):
    """Get coordinates from Swath objects."""
    hdf_attr = hdf5_file_attrs(hdf)
    granule_id = hdf_attr["FileHeader"]["GranuleNumber"]
    lon = hdf[scan_mode]["Longitude"][:]
    lat = hdf[scan_mode]["Latitude"][:]
    time = _parse_hdf_gpm_scantime(hdf[scan_mode]["ScanTime"])
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


def get_grid_coords(hdf, scan_mode):
    """Get coordinates from Grid objects.

    IMERG and GRID products does not have GranuleNumber!
    """
    hdf_attr = hdf5_file_attrs(hdf)
    lon = hdf[scan_mode]["lon"][:]
    lat = hdf[scan_mode]["lat"][:]
    time = hdf_attr["FileHeader"]["StartGranuleDateTime"][:-1]
    time = np.array(
        np.datetime64(time) + np.timedelta64(30, "m"), ndmin=1
    )  # TODO: document why + 30 min
    coords = {
        "time": time,
        "lon": lon,
        "lat": lat,
    }
    return coords


def get_coords(hdf, scan_mode):
    """Get coordinates from GPM objects."""
    coords = (
        get_grid_coords(hdf, scan_mode) if scan_mode == "Grid" else get_orbit_coords(hdf, scan_mode)
    )
    return coords
