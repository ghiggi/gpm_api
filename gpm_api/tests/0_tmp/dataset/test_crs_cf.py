#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 11:09:49 2023

@author: ghiggi
"""
import numpy as np
import pyproj
import xarray as xr

### TODO: WAIT TO TEST CRS FUNCTIONS BECAUSE WE MIGHT DEPEND ON GEOXARRAY FOR THAT


def create_dummy_swath_ds():
    # Define dataset
    variable = "variable"
    shape = (3, 5)
    data = np.zeros(shape)
    lat = np.linspace(0, 10, shape[0])
    lon = np.linspace(20, 40, shape[1])
    lons, lats = np.meshgrid(lon, lat)
    coords = {"longitude": (("y", "x"), lons), "latitude": (("y", "x"), lats)}
    dims = ("y", "x")
    ds = xr.DataArray(data=data, coords=coords, dims=dims, name=variable).to_dataset()
    # Define pyproj CRS
    crs = pyproj.CRS(proj="longlat", ellps="WGS84")
    return ds, crs


def create_dummy_proj_wgs84_ds():
    # Define dataset
    variable = "variable"
    shape = (3, 5)
    data = np.zeros(shape)
    lat = np.linspace(0, 10, shape[0])
    lon = np.linspace(20, 40, shape[1])
    coords = {"longitude": lon, "latitude": lat}
    dims = ("latitude", "longitude")
    ds = xr.DataArray(data=data, coords=coords, dims=dims, name=variable).to_dataset()
    # Define pyproj CRS
    crs = pyproj.CRS(proj="longlat", ellps="WGS84")
    return ds, crs


def create_dummy_proj_geo_radian_ds():
    # Define pyproj CRS (GOES-16)
    proj_str = "+proj=geos +sweep=x +lon_0=-75 +h=35786023 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs +type=crs"
    crs = pyproj.CRS(proj_str)
    # Define dataset
    variable = "variable"
    shape = (3, 5)
    data = np.zeros(shape)
    coords = {
        "y": np.arange(shape[0]),  # radians [0-3]
        "x": np.arange(shape[1]),  # radians [0-5]
    }
    dims = ("y", "x")
    ds = xr.DataArray(data=data, dims=dims, coords=coords, name=variable).to_dataset()
    # Set x,y unit to radian
    ds["x"].attrs["units"] = "radian"
    ds["y"].attrs["units"] = "radian"
    # Return info
    return ds, crs


def create_dummy_proj_geo_metre_ds():
    # Define pyproj CRS (GOES-16)
    proj_str = "+proj=geos +sweep=x +lon_0=-75 +h=35786023 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs +type=crs"
    crs = pyproj.CRS(proj_str)
    # Retrieve satellite height
    satellite_height = crs.to_dict()["h"]
    # Define dataset
    variable = "variable"
    shape = (3, 5)
    data = np.zeros(shape)
    coords = {
        "y": np.arange(shape[0]) * satellite_height,  # meter
        "x": np.arange(shape[1]) * satellite_height,  # meter
    }
    dims = ("y", "x")
    ds = xr.DataArray(data=data, dims=dims, coords=coords, name=variable).to_dataset()
    # Set x,y unit to metre
    ds["x"].attrs["units"] = "metre"
    ds["y"].attrs["units"] = "metre"
    return ds, crs


def create_dummy_proj_polar_ds():
    # Define pyproj CRS
    proj_str = "+proj=laea +lat_0=-90 +lon_0=0 +a=6371228.0 +units=m"
    crs = pyproj.CRS(proj_str)
    # Define dataset
    variable = "variable"
    shape = (3, 5)
    data = np.zeros(shape)
    coords = {
        "y": np.arange(shape[0]),  # radians [0-3]
        "x": np.arange(shape[1]),  # radians [0-5]
    }
    dims = ("y", "x")
    ds = xr.DataArray(data=data, dims=dims, coords=coords, name=variable).to_dataset()
    return ds, crs
