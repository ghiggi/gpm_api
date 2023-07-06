#!/usr/bin/env python3
"""
Created on Mon Jul  3 13:58:19 2023

@author: ghiggi
"""
import numpy as np
import xarray as xr

# TODO: mask by label first ?

# func must return a dictionary {key: value}

# str(slice(0,1))

# a = {'dim1': slice(0, 1, None)}
# b = str(a) # "{'dim1': slice(0, 1, None)}"
# c = eval(b)


def _get_stats_names(func):
    arr = np.zeros((2, 2))
    dictionary = func(arr)
    names = list(dictionary.keys())
    return names


def _get_stats_array(array, func):
    list_stats = func(array)
    stats = np.stack(list_stats.values(), axis=-1)
    return stats


def compute_patch_stats(xr_obj, func):
    # Retrieve statistics names
    stats_names = _get_stats_names(func)

    # Define gufunc kwargs
    input_core_dims = [[], []]
    dask_gufunc_kwargs = {
        "output_sizes": {
            "stats": len(stats_names),
        }
    }
    # Input dimensions (over which to reduce --> all)
    input_core_dims = xr_obj.dims

    # Apply ufunc
    ds_stats = xr.apply_ufunc(
        _get_stats_array,
        xr_obj,
        kwargs={"func": func},
        input_core_dims=[input_core_dims],
        output_core_dims=[["stats"]],
        vectorize=False,
        dask="allowed",
        dask_gufunc_kwargs=dask_gufunc_kwargs,
        output_dtypes=["float64"],
    )
    ds_stats = ds_stats.assign_coords({"stats": stats_names})

    if isinstance(xr_obj, xr.Dataset):
        ds_stats = ds_stats.to_array(dim="variable")
    return ds_stats
