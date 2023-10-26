#!/usr/bin/env python3
"""
Created on Tue Jul 18 17:33:38 2023

@author: ghiggi
"""
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
from gpm_api.utils.checks import is_regular
from gpm_api.utils.time import (
    ensure_time_validity,
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
    try:
        if is_grid(ds):
            if config.get("warn_non_contiguous_scans"):
                if not is_regular(ds):
                    msg = "Missing timesteps across the dataset !"
                    warnings.warn(msg, GPM_Warning)
        elif is_orbit(ds):
            if config.get("warn_non_contiguous_scans"):
                if not is_regular(ds):
                    msg = "Presence of non-contiguous scans !"
                    warnings.warn(msg, GPM_Warning)
    except Exception:
        pass

    ###-----------------------------------------------------------------------.
    ## Check geolocation latitude/longitude coordinates
    # TODO: check_valid_geolocation
    # TODO: ensure_valid_geolocation (1 spurious pixel)
    # TODO: ds_gpm.gpm_api.valid_geolocation

    ###-----------------------------------------------------------------------.

    return ds
