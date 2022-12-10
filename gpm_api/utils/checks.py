#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 18:41:48 2022

@author: ghiggi
"""
import xarray as xr 


def check_is_xarray(x):
    if not isinstance(x, (xr.DataArray, xr.Dataset)):
        raise TypeError("Expecting a xr.Dataset or xr.DataArray.")


def check_is_xarray_dataarray(x):
    if not isinstance(x, xr.DataArray):
        raise TypeError("Expecting a xr.DataArray.")


def check_is_xarray_dataset(x):
    if not isinstance(x, xr.Dataset):
        raise TypeError("Expecting a xr.Dataset.")

        
def check_is_spatial_2D_field(da):
    from .geospatial import is_spatial_2D_field
    if not is_spatial_2D_field(da):
        raise ValueError("Expecting a 2D GPM field.")