#!/usr/bin/env python3
"""
Created on Tue Jul 18 17:06:05 2023

@author: ghiggi
"""
import os

import datatree
import xarray as xr

from gpm_api.dataset.attrs import decode_string
from gpm_api.dataset.dimensions import _rename_datatree_dimensions

# TODO:
# --> open datatrees and concat datatrees
# --> create datatree with option "flattened_scan_modes"
# --> gpm_api.open_granule(datatree=False)  # or if multiple scan_modes provided
# --> gpm_api.open_dataset(datatree=False)  # or if multiple scan_modes provided


def open_datatree(filepath, chunks={}, decode_cf=False, use_api_defaults=True):
    """Open HDF5 in datatree object.

    - chunks={} --> Lazy map to dask.array
      --> Wait for https://github.com/pydata/xarray/pull/7948
      --> Maybe need to implement "auto" option manually that defaults to full shape"
    - chunks="auto" --> datatree fails. Can not estimate size of object dtype !
    - chunks=None --> lazy map to numpy.array
    """
    try:
        dt = datatree.open_datatree(filepath, engine="netcdf4", chunks=chunks, decode_cf=decode_cf)
        check_non_empty_granule(dt, filepath)
    except Exception as e:
        check_valid_granule(filepath)
        raise ValueError(e)

    # Assign dimension names
    dt = _rename_datatree_dimensions(dt, use_api_defaults=use_api_defaults)
    return dt


def check_non_empty_granule(dt, filepath):
    """Check that the datatree (or dataset) is not empty."""
    attrs = dt.attrs
    attrs = decode_string(attrs["FileHeader"])
    is_empty_granule = attrs["EmptyGranule"] != "NOT_EMPTY"
    if is_empty_granule:
        raise ValueError(f"{filepath} is an EMPTY granule !")


def check_valid_granule(filepath):
    """Raise an explanatory error if the GPM granule is not readable."""
    # Check the file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The filepath {filepath} does not exist.")
    # Identify the cause of the error if xarray can't open the file
    try:
        with xr.open_dataset(filepath, engine="netcdf4", group="") as ds:
            check_non_empty_granule(ds, filepath)
    except Exception as e:
        _identify_error(e, filepath)


def _identify_error(e, filepath):
    """Identify error when opening HDF file."""
    error_str = str(e)
    # TODO: to create test case with corrupted file (i.e. interrupted download)
    # netCDF4._netCDF4._ensure_nc_success
    if "[Errno -101] NetCDF: HDF error" in error_str:
        # os.remove(filepath) # TODO: gpm_api flag !
        msg = f"The file {filepath} is corrupted and is being removed. It must be redownload."
        raise ValueError(msg)
    elif "[Errno -51] NetCDF: Unknown file format" in error_str:
        msg = f"The GPM-API is not currently able to read the file format of {filepath}. Report the issue please."
        raise ValueError(msg)
    elif "lock" in error_str:
        msg = "Unfortunately, HDF locking is occurring."
        msg += "Export the environment variable HDF5_USE_FILE_LOCKING = 'FALSE' into your environment (i.e. in the .bashrc).\n"  # noqa
        msg += f"The error is: '{error_str}'."
        raise ValueError(msg)
    else:
        msg = f"The following file is corrupted. Error is {e}. Redownload the file."
        raise ValueError(msg)
    return
