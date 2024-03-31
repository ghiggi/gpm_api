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
"""This module contains functions to read a GPM granule into a DataTree object."""
import os

import datatree
import xarray as xr

from gpm.dataset.attrs import decode_string
from gpm.dataset.dimensions import _rename_datatree_dimensions

# TODO:
# --> open datatrees and concat datatrees
# --> create datatree with option "flattened_scan_modes"
# --> gpm.open_granule(datatree=False)  # or if multiple scan_modes provided
# --> gpm.open_dataset(datatree=False)  # or if multiple scan_modes provided


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
    return _rename_datatree_dimensions(dt, use_api_defaults=use_api_defaults)


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
    if "[Errno -51] NetCDF: Unknown file format" in error_str:
        msg = f"The GPM-API is not currently able to read the file format of {filepath}. Report the issue please."
        raise ValueError(msg)
    if "lock" in error_str:
        msg = "Unfortunately, HDF locking is occurring."
        msg += "Export the environment variable HDF5_USE_FILE_LOCKING = 'FALSE' into your environment (i.e. in the .bashrc).\n"  # noqa
        msg += f"The error is: '{error_str}'."
        raise ValueError(msg)
    msg = f"The following file is corrupted. Error is {e}. Redownload the file."
    raise ValueError(msg)
