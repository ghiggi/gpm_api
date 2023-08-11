#!/usr/bin/env python3
"""
Created on Fri Jul 28 11:34:55 2023

@author: ghiggi
"""
import numpy as np
import xarray as xr


def get_data_array(xr_obj, variable):
    if isinstance(xr_obj, xr.DataArray):
        if xr_obj.name != variable:
            print(f"Warning: the DataArray name is not '{variable}'!")
    else:
        if variable not in xr_obj:
            raise ValueError(f"'{variable}' is not a variable of the xarray Dataset.")
        xr_obj = xr_obj[variable]
    return xr_obj


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
