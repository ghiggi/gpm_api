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
"""This module contains utilities for the decoding of GPM product variables."""
import dask.array
import numpy as np


def is_dataarray_decoded(da):
    """Check if a DataArray has been decoded by GPM-API."""
    return da.attrs.get("gpm_api_decoded", "no") == "yes"


def add_decoded_flag(ds, variables):
    """Add gpm_api_decoded flag to GPM-API decoded variables."""
    for var in variables:
        if var in ds:
            ds[var].attrs["gpm_api_decoded"] = "yes"
    return ds


def remap_numeric_array(arr, remapping_dict):
    """Remap the values of a numeric array."""
    # TODO: this is erroneous
    # TODO: implement that works with dask array also !
    # TODO: implement it works if values not in remapping dict
    # TODO: implement it works if only np.nan values
    # remapping_dict = {-1111: 0, 0: 1, 10: 2, 11: 3, 20: 4, 21: 5}
    original_values = list(remapping_dict.keys())

    # Use np.searchsorted to remap the array
    # TODO: works only if not np.nan and reamp to 0-n ?
    return np.searchsorted(original_values, arr, sorter=np.argsort(original_values))

    # Correct Alternative (but less performant) :
    # np.vectorize(remapping_dict.__getitem__)(arr)


def ceil_datarray(da):
    """Ceil a xr.DataArray."""
    data = da.data
    data = np.ceil(data) if hasattr(data, "chunks") else dask.array.ceil(data)
    da.data = data
    return da
