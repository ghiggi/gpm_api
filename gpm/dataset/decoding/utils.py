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
    """Check if a xarray.DataArray has been decoded by GPM-API."""
    return da.attrs.get("gpm_api_decoded", "no") == "yes"


def add_decoded_flag(ds, variables):
    """Add gpm_api_decoded flag to GPM-API decoded variables."""
    for var in variables:
        if var in ds:
            ds[var].attrs["gpm_api_decoded"] = "yes"
    return ds


# def _np_remap_numeric_array1(arr, remapping_dict, fill_value=np.nan):
#     # VERY SLOW ALTERNATIVE
#     isna = np.isnan(arr)
#     arr[isna] = -1 # dummy
#     unique_values = np.unique(arr[~np.isnan(arr)])
#     _ = [remapping_dict.setdefault(value, fill_value) for value in unique_values if value not in remapping_dict]
#     remapping_dict = {float(k): float(v) for k, v in remapping_dict.items()}
#     new_arr = np.vectorize(remapping_dict.__getitem__)(arr)
#     new_arr[isna] = np.nan
#     return new_arr


def _np_remap_numeric_array(arr, remapping_dict, fill_value=np.nan):
    # Define conditions
    conditions = [arr == i for i in remapping_dict]
    # Define choices corresponding to conditions
    choices = remapping_dict.values()
    # Apply np.select to transform the array
    return np.select(conditions, choices, default=fill_value)


def _dask_remap_numeric_array(arr, remapping_dict, fill_value=np.nan):
    return dask.array.map_blocks(_np_remap_numeric_array, arr, remapping_dict, fill_value, dtype=arr.dtype)


def remap_numeric_array(arr, remapping_dict, fill_value=np.nan):
    """Remap the values of a numeric array."""
    if hasattr(arr, "chunks"):
        return _dask_remap_numeric_array(arr, remapping_dict, fill_value=fill_value)
    return _np_remap_numeric_array(arr, remapping_dict, fill_value=fill_value)


def ceil_dataarray(da):
    """Ceil a xarray.DataArray."""
    data = da.data
    data = np.ceil(data) if hasattr(data, "chunks") else dask.array.ceil(data)
    da.data = data
    return da
