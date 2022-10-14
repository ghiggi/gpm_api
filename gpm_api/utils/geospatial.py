#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 09:30:29 2022

@author: ghiggi
"""
import numpy as np 
import xarray as xr


def unwrap_longitude_degree(x, period=360):
    """Unwrap longitude array."""
    x = np.asarray(x)
    mod = period / 2
    return (x + mod) % (2 * mod) - mod


def crop_dataset(ds, bbox):
    # TODO: Check bbox 
    
    # Crop orbit data 
    # - Subset only along_track to allow concat on cross_track !
    if 'cross_track' in list(ds.dims):
        lon = ds['lon'].data
        lat = ds['lat'].data
        idx_row, idx_col = np.where(
              (lon >= bbox[0])
            & (lon <= bbox[1])
            & (lat >= bbox[2])
            & (lat <= bbox[3])
        )
        if idx_row.size == 0: # or idx_col.size == 0:
            raise ValueError("No data inside the provided bounding box.")
        # TODO: Check continuous ... otherwise warn and return a list of datasets 
        ds_subset = ds.isel(along_track=slice((min(idx_row)), (max(idx_row) + 1)))
    elif "lon" in list(ds.dims):  
        idx_row = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]
        idx_col = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]
        # If no data in the bounding box in current granule, return empty list
        if idx_row.size == 0 or idx_col.size == 0:
            raise ValueError("No data inside the provided bounding box.")
        else:  
            ds_subset = ds.isel(lon=idx_row, lat=idx_col)
    else: 
        orbit_dims = ("cross_track", "along_track")
        grid_dims = ("lon", "lat")
        raise ValueError(f"Dataset not recognized. Expecting dimensions {orbit_dims} or {grid_dims}.")
    
    return ds_subset


def is_orbit(ds): 
    if "along_track" in list(ds.dims): 
        return True 
    else:
        return False 


def is_grid(ds): 
    if "longitude" in list(ds.dims): 
        return True 
    else:
        return False 
    
    
def get_pyresample_area(ds): 
    from pyresample import SwathDefinition, AreaDefinition
    # If Orbit Granule --> Swath Definition
    if is_orbit(ds): 
        # Define SwathDefinition with xr.DataArray lat/lons
        # - Otherwise fails https://github.com/pytroll/satpy/issues/1434
        lons = ds["lon"].values
        lats = ds["lat"].values
        
        # TODO: this might be needed 
        # - otherwise ValueError 'ndarray is not C-contiguous' when resampling
        # lons = np.ascontiguousarray(lons) 
        # lats = np.ascontiguousarray(lats)
        
        lons = xr.DataArray(lons, dims=["y", "x"])
        lats = xr.DataArray(lats, dims=["y", "x"])
        swath_def = SwathDefinition(lons, lats)
        return swath_def
    # If Grid Granule --> AreaDefinition
    elif is_grid(ds): 
         # Define AreaDefinition 
         raise NotImplementedError()
    # Unknown 
    else: 
        raise NotImplementedError()
      
