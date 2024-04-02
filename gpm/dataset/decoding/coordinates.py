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
"""This module contains functions to sanitize GPM-API Dataset coordinates."""
import functools
import os

import numpy as np

from gpm.utils.yaml import read_yaml


def ensure_valid_coords(ds, raise_error=False):
    """Ensure geographic coordinates are within expected range."""
    invalid_coords = np.logical_or(
        np.logical_or(ds["lon"].data < -180, ds["lon"].data > 180),
        np.logical_or(ds["lat"].data < -90, ds["lat"].data > 90),
    )
    if np.any(invalid_coords):
        if raise_error:
            raise ValueError("Invalid geographic coordinate in the granule.")
        da_invalid_coords = ds["lon"].copy()
        da_invalid_coords.data = invalid_coords
        # For each variable, set NaN value where invalid coordinates
        ds = ds.where(~da_invalid_coords)
        # Add NaN to longitude and latitude
        ds["lon"] = ds["lon"].where(~da_invalid_coords)
        ds["lat"] = ds["lat"].where(~da_invalid_coords)
    return ds


def _add_cmb_range_coordinate(ds, scan_mode):
    """Add range coordinate to 2B-<CMB> products."""
    if "range" in list(ds.dims) and scan_mode in ["NS", "KuKaGMI", "KuGMI", "KuTMI"]:
        range_values = np.arange(0, 88 * 250, step=250)
        ds = ds.assign_coords({"range": range_values})
        ds["range"].attrs["units"] = "m"
    return ds


def _add_cmb_coordinates(ds, product, scan_mode):
    """Set coordinates of 2B-GPM-CORRA product."""
    if "pmw_frequency" in list(ds.dims):
        pmw_frequency = get_pmw_frequency_corra(product)
        ds = ds.assign_coords({"pmw_frequency": pmw_frequency})

    if (scan_mode in ("KuKaGMI", "NS")) and "radar_frequency" in list(ds.dims):
        ds = ds.assign_coords({"radar_frequency": ["Ku", "Ka"]})

    return _add_cmb_range_coordinate(ds, scan_mode)


def _add_radar_range_coordinate(ds, scan_mode):
    """Add range coordinate to 2A-<RADAR> products."""
    # - V6 and V7: 1BKu 260 bins NS and MS, 130 at HS
    if "range" in list(ds.dims):
        if scan_mode in ["HS"]:
            range_values = np.arange(0, 88 * 250, step=250)
            ds = ds.assign_coords({"range": range_values})
        if scan_mode in ["FS", "MS", "NS"]:
            range_values = np.arange(0, 176 * 125, step=125)
            ds = ds.assign_coords({"range": range_values})
        ds["range"].attrs["units"] = "m"
    return ds


def _add_wished_coordinates(ds):
    """Add wished coordinates to the dataset."""
    from gpm.dataset.groups_variables import WISHED_COORDS

    for var in WISHED_COORDS:
        if var in list(ds):
            ds = ds.set_coords(var)
    return ds


def _add_radar_coordinates(ds, product, scan_mode):
    """Add range, height, radar_frequency, paramDSD coordinates to <RADAR> products."""
    if product == "2A-DPR" and "radar_frequency" in list(ds.dims):
        ds = ds.assign_coords({"radar_frequency": ["Ku", "Ka"]})
    if product in ["2A-DPR", "2A-Ku", "2A-Ka", "2A-PR"] and "paramDSD" in list(ds):
        ds = ds.assign_coords({"DSD_params": ["Nw", "Dm"]})
    # Add radar range
    return _add_radar_range_coordinate(ds, scan_mode)


@functools.cache
def get_pmw_frequency_dict():
    """Get PMW info dictionary."""
    from gpm import _root_path

    filepath = os.path.join(_root_path, "gpm", "etc", "pmw", "frequencies.yaml")
    return read_yaml(filepath)


@functools.cache
def get_pmw_frequency(sensor, scan_mode):
    """Get product info dictionary."""
    pmw_dict = get_pmw_frequency_dict()
    return pmw_dict[sensor][scan_mode]


def get_pmw_frequency_corra(product):
    if product == "2B-GPM-CORRA":
        return get_pmw_frequency("GMI", scan_mode="S1") + get_pmw_frequency("GMI", scan_mode="S2")
    if product == "2B-TRMM-CORRA":
        return (
            get_pmw_frequency("TMI", scan_mode="S1")
            + get_pmw_frequency("TMI", scan_mode="S2")
            + get_pmw_frequency("TMI", scan_mode="S3")
        )
    raise ValueError("Invalid (CORRA) product {product}.")


def _parse_sun_local_time(ds):
    """Ensure sunLocalTime to be in float type."""
    dtype = ds["sunLocalTime"].data.dtype
    if dtype == "timedelta64[ns]":
        ds["sunLocalTime"] = ds["sunLocalTime"].astype(float) / 10**9 / 60 / 60
    elif np.issubdtype(dtype, np.floating):
        pass
    else:
        raise ValueError("Expecting sunLocalTime as float or timedelta64[ns]")
    ds["sunLocalTime"].attrs["units"] = "decimal hours"  # to avoid open_dataset netCDF convert to timedelta64[ns]
    return ds


def _add_1c_pmw_frequency(ds, product, scan_mode):
    """Add the 'pmw_frequency' coordinates to 1C-<PMW> products."""
    if "pmw_frequency" in list(ds.dims):
        pmw_frequency = get_pmw_frequency(sensor=product.split("-")[1], scan_mode=scan_mode)
        ds = ds.assign_coords({"pmw_frequency": pmw_frequency})
    return ds


def _add_pmw_coordinates(ds, product, scan_mode):
    """Add coordinates to PMW products."""
    if product.startswith("1C"):
        ds = _add_1c_pmw_frequency(ds, product, scan_mode)
    return ds


def set_coordinates(ds, product, scan_mode):
    # Ensure valid coordinates
    if "cross_track" in list(ds.dims):
        ds = ensure_valid_coords(ds, raise_error=False)

    # Add gpm_range_id coordinate
    if "range" in list(ds.dims):
        range_id = np.arange(ds.sizes["range"])
        ds = ds.assign_coords({"gpm_range_id": ("range", range_id)})

    # Add wished coordinates
    ds = _add_wished_coordinates(ds)

    # Convert sunLocalTime to float
    if "sunLocalTime" in ds:
        ds = _parse_sun_local_time(ds)
        ds = ds.set_coords("sunLocalTime")

    #### PMW
    # - 1C products
    if product.startswith("1C"):
        ds = _add_pmw_coordinates(ds, product, scan_mode)

    #### RADAR
    if product in ["2A-DPR", "2A-Ku", "2A-Ka", "2A-PR"]:
        ds = _add_radar_coordinates(ds, product, scan_mode)

    #### CMB
    if product in ["2B-GPM-CORRA", "2B-TRMM-CORRA"]:
        ds = _add_cmb_coordinates(ds, product, scan_mode)

    #### SLH and CSH products
    if product in ["2A-GPM-SLH", "2B-GPM-CSH"] and "nlayer" in list(ds.dims):
        # Fixed heights for 2HSLH and 2HCSH
        # - FileSpec v7: p.2395, 2463
        height = np.linspace(0.25 / 2, 20 - 0.25 / 2, 80) * 1000  # in meters
        ds = ds.rename_dims({"nlayer": "height"})
        ds = ds.assign_coords({"height": height})
        ds["height"].attrs["units"] = "km a.s.l"

    #### IMERG
    if "HQobservationTime" in ds:
        da = ds["HQobservationTime"]
        da = da.where(da >= 0)  # -9999.9 --> NaN
        da.attrs["units"] = "minutes from the start of the current half hour"
        da.encoding["dtype"] = "uint8"
        da.encoding["_FillValue"] = 255
        da.attrs.pop("_FillValue", None)
        da.attrs.pop("CodeMissingValue", None)
        ds["HQobservationTime"] = da

    if "MWobservationTime" in ds:
        da = ds["MWobservationTime"]
        da = da.where(da >= 0)  # -9999.9 --> NaN
        da.attrs["units"] = "minutes from the start of the current half hour"
        da.attrs.pop("_FillValue", None)
        da.attrs.pop("CodeMissingValue", None)
        da.encoding["dtype"] = "uint8"
        da.encoding["_FillValue"] = 255
        ds["MWobservationTime"] = da
    return ds
