#!/usr/bin/env python3
"""
Created on Fri Jul 28 11:34:55 2023

@author: ghiggi
"""
import dask.array
import numpy as np


def remap_numeric_array(arr, remapping_dict):
    """Remap the values of a numeric array."""
    # TODO: implement that works with dask array also !
    # TODO: implement it works if values not in remapping dict
    # TODO: implement it works if only np.nan values
    # remapping_dict = {-1111: 0, 0: 1, 10: 2, 11: 3, 20: 4, 21: 5}
    original_values = list(remapping_dict.keys())
    list(remapping_dict.values())

    # Use np.searchsorted to remap the array
    # TODO: works only if not np.nan and reamp to 0-n ?
    remapped_arr = np.searchsorted(original_values, arr, sorter=np.argsort(original_values))

    # Alternative: but less performant !!!
    # def remap_value(value):
    #     return remapping_dict[value]
    # remapped_arr1 = np.vectorize(remap_value)(arr)

    return remapped_arr


def ceil_datarray(da):
    """Ceil a xr.DataArray."""
    data = da.data
    if hasattr(data, "chunks"):
        data = np.ceil(data)
    else:
        data = dask.array.ceil(data)
    da.data = data
    return da
