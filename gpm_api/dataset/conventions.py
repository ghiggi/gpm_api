#!/usr/bin/env python3
"""
Created on Tue Jul 18 17:33:38 2023

@author: ghiggi
"""
import warnings

from gpm_api.dataset.attrs import add_history
from gpm_api.dataset.coords import set_coords_attrs
from gpm_api.dataset.crs import set_dataset_crs
from gpm_api.dataset.decoding import decode_dataset
from gpm_api.dataset.encoding import set_encoding
from gpm_api.utils.time import (
    subset_by_time,
)
from gpm_api.utils.warnings import GPM_Warning

EPOCH = "seconds since 1970-01-01 00:00:00"


def _check_time_period_coverage(ds, start_time=None, end_time=None, raise_error=False):
    """Check time period start_time, end_time is covered.

    If raise_error=True, raise error if time period is not covered.
    If raise_error=False, it raise a GPM warning.

    """
    # Get first and last timestep from xr.Dataset
    first_start = ds["time"].data[0].astype("M8[s]").tolist()
    last_end = ds["time"].data[-1].astype("M8[s]").tolist()
    # Check time period is covered
    msg = ""
    if start_time and first_start > start_time:
        msg = f"The dataset start at {first_start}, although the specified start_time is {start_time}."
    if end_time and last_end < end_time:
        msg1 = f"The dataset end_time {last_end} occurs before the specified end_time {end_time}."
        msg = msg[:-1] + "; and t" + msg1[1:] if msg != "" else msg1
    if msg != "":
        if raise_error:
            raise ValueError(msg)
        else:
            warnings.warn(msg, GPM_Warning)


def reshape_dataset(ds):
    """Define the dataset dimension order.

    It ensure that the output dimension order is  (y, x)
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


def finalize_dataset(ds, product, decode_cf, start_time=None, end_time=None):
    """Finalize GPM dataset."""
    import pyproj

    ##------------------------------------------------------------------------.
    # Transpose to have (y, x) dimension order
    ds = reshape_dataset(ds)

    ##------------------------------------------------------------------------.
    # Decode dataset
    if decode_cf:
        ds = decode_dataset(ds)

    ##------------------------------------------------------------------------.
    # Add coordinates attributes
    ds = set_coords_attrs(ds)

    ##------------------------------------------------------------------------.
    # Add CRS information
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
    # Add time encoding
    encoding = {}
    encoding["units"] = EPOCH
    encoding["calendar"] = "proleptic_gregorian"
    ds["time"].encoding = encoding

    ##------------------------------------------------------------------------.
    # Add GPM-API global attributes
    ds = add_history(ds)
    ds.attrs["gpm_api_product"] = product

    ##------------------------------------------------------------------------.
    # Add coordinates and variables encoding
    ds = set_encoding(ds)

    ##------------------------------------------------------------------------.
    # Subset dataset for start_time and end_time
    # - Raise warning if the time period is not fully covered
    # - The warning can raise if some data are not downloaded or some granule
    #   at the start/end of the period are empty
    ds = subset_by_time(ds, start_time=start_time, end_time=end_time)
    _check_time_period_coverage(ds, start_time=start_time, end_time=end_time, raise_error=False)

    return ds
