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
"""This module contains functions to enforce CF-conventions into the GPM-API objects."""
import datetime
import warnings

from gpm_api.checks import (
    is_grid,
    is_orbit,
)
from gpm_api.dataset.attrs import add_history
from gpm_api.dataset.coords import set_coords_attrs
from gpm_api.dataset.crs import set_dataset_crs
from gpm_api.dataset.decoding.cf import apply_cf_decoding
from gpm_api.dataset.decoding.coordinates import set_coordinates
from gpm_api.dataset.decoding.dataarray_attrs import standardize_dataarrays_attrs
from gpm_api.dataset.decoding.routines import decode_variables
from gpm_api.utils.checks import has_valid_geolocation, is_regular
from gpm_api.utils.time import (
    ensure_time_validity,
    subset_by_time,
)
from gpm_api.utils.warnings import GPM_Warning

EPOCH = "seconds since 1970-01-01 00:00:00"


def _check_time_period_coverage(ds, start_time=None, end_time=None, raise_error=False):
    """Check time period start_time, end_time is covered with a tolerance of 5 seconds.

    If raise_error=True, raise error if time period is not covered.
    If raise_error=False, it raise a GPM warning.

    """
    # Define tolerance in seconds
    tolerance = datetime.timedelta(seconds=5)

    # Get first and last timestep from xr.Dataset
    first_start = ds["time"].data[0].astype("M8[s]").tolist()
    last_end = ds["time"].data[-1].astype("M8[s]").tolist()
    # Check time period is covered
    msg = ""
    if start_time and first_start - tolerance > start_time:
        msg = f"The dataset start at {first_start}, although the specified start_time is {start_time}. "
    if end_time and last_end + tolerance < end_time:
        msg1 = f"The dataset end_time {last_end} occurs before the specified end_time {end_time}. "
        msg = msg[:-1] + "; and t" + msg1[1:] if msg != "" else msg1
    if msg != "":
        msg += "Some granules may be missing!"
        if raise_error:
            raise ValueError(msg)
        else:
            warnings.warn(msg, GPM_Warning)


def reshape_dataset(ds):
    """Define the dataset dimension order.

    It ensures that the output dimension order is  (y, x)
    This shape is expected by i.e. pyresample and matplotlib
    For GPM GRID objects:  (..., time, lat, lon)
    For GPM ORBIT objects: (cross_track, along_track, ...)
    """
    if "lat" in ds.dims:
        ds = ds.transpose(..., "lat", "lon")
    else:
        if "cross_track" in ds.dims:
            ds = ds.transpose("cross_track", "along_track", ...)
        else:
            ds = ds.transpose("along_track", ...)
    return ds


def finalize_dataset(ds, product, decode_cf, scan_mode, start_time=None, end_time=None):
    """Finalize GPM xr.Dataset object."""
    import pyproj

    from gpm_api import config

    ##------------------------------------------------------------------------.
    # Clean out HDF5 attributes
    # - CodeMissingValue --> _FillValue
    # - FillValue --> _FillValue,
    # - Units --> units,
    # - Remove DimensionNames,
    # - Add <gpm_api_product> : <product> key : value
    ds = standardize_dataarrays_attrs(ds, product)

    ##------------------------------------------------------------------------.
    # Set relevant coordinates
    # - Add range id, radar and pmw frequencies ...
    ds = set_coordinates(ds, product, scan_mode)

    ##------------------------------------------------------------------------.
    # Decode dataset
    # - _FillValue is moved from attrs to encoding !
    if decode_cf:
        ds = apply_cf_decoding(ds)

    ###-----------------------------------------------------------------------.
    ## Check swath time coordinate
    # Ensure validity of the time dimension
    # - Infill up to 10 consecutive NaT
    # - Do not check for regular time dimension !
    ds = ensure_time_validity(ds, limit=10)

    ##------------------------------------------------------------------------.
    # Decode variables
    ds = decode_variables(ds, product)

    ##------------------------------------------------------------------------.
    # Add CF-compliant coordinates attributes and encoding
    ds = set_coords_attrs(ds)

    # Add time encoding
    encoding = {}
    encoding["units"] = EPOCH
    encoding["calendar"] = "proleptic_gregorian"
    ds["time"].encoding = encoding

    ##------------------------------------------------------------------------.
    # Transpose to have (y, x) dimension order
    ds = reshape_dataset(ds)

    ##------------------------------------------------------------------------.
    # Add CF-compliant CRS information
    # - See Geolocation toolkit ATBD at
    #   https://gpm.nasa.gov/sites/default/files/document_files/GPMGeolocationToolkitATBDv2.1-2012-07-31.pdf
    # TODO: set_dataset_crs should be migrated to cf_xarray ideally
    try:
        crs = pyproj.CRS(proj="longlat", ellps="WGS84")
        ds = set_dataset_crs(ds, crs=crs, grid_mapping_name="crsWGS84", inplace=False)
    except Exception:
        msg = "The CRS coordinate is not set because the dataset variables does not have 2D spatial dimensions."
        warnings.warn(msg, GPM_Warning)

    ##------------------------------------------------------------------------.
    # Add GPM-API global attributes
    ds = add_history(ds)
    ds.attrs["gpm_api_product"] = product

    ##------------------------------------------------------------------------.
    # Subset dataset for start_time and end_time
    # - Raise warning if the time period is not fully covered
    # - The warning can raise if some data are not downloaded or some granule
    #   at the start/end of the period are empty
    ds = subset_by_time(ds, start_time=start_time, end_time=end_time)
    _check_time_period_coverage(ds, start_time=start_time, end_time=end_time, raise_error=False)

    ###-----------------------------------------------------------------------.
    # Warn if:
    # - non-contiguous scans in orbit data
    # - non-regular timesteps in grid data
    # - invalid geolocation coordinates
    try:
        if is_grid(ds):
            if config.get("warn_non_contiguous_scans"):
                if not is_regular(ds):
                    msg = "Missing timesteps across the dataset !"
                    warnings.warn(msg, GPM_Warning)
        elif is_orbit(ds):
            if config.get("warn_invalid_geolocation"):
                if not has_valid_geolocation(ds):
                    msg = "Presence of invalid geolocation coordinates !"
                    warnings.warn(msg, GPM_Warning)
            if config.get("warn_non_contiguous_scans"):
                if not is_regular(ds):
                    msg = "Presence of non-contiguous scans !"
                    warnings.warn(msg, GPM_Warning)
    except Exception:
        pass

    ###-----------------------------------------------------------------------.
    return ds
