#!/usr/bin/env python3
"""
Created on Sat Dec 10 18:44:56 2022

@author: ghiggi
"""


def xr_exclude_variables_without(ds, dim):
    # ds.filter.variables_without_dims()
    valid_vars = [var for var, da in ds.items() if dim in list(da.dims)]
    if len(valid_vars) == 0:
        raise ValueError(f"No dataset variables with dimension {dim}")
    ds_subset = ds[valid_vars]
    return ds_subset
