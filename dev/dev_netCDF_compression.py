#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:05:52 2023

@author: ghiggi
"""
# - Define xr.Dataset 
ds_grid = xr.Dataset(data_vars={},
                     coords=coords_dict,
)

ds_grid.to_netcdf("/tmp/dummy2.nc")

d = xr.open_dataset("/tmp/dummy.nc")

ds_grid["latitude"].encoding.pop("zlib")
ds_grid["latitude"].encoding["zlib"] = True
ds_grid.to_netcdf("/tmp/dummy1.nc")
ds_grid["latitude"].encoding["compression"] = "None"
ds_grid.to_netcdf("/tmp/dummy2.nc")
ds_grid["latitude"].encoding["compression"] = "szip"
ds_grid.to_netcdf("/tmp/dummy3.nc")
ds_grid["latitude"].encoding["compression"] = "zstd"
ds_grid.to_netcdf("/tmp/dummy3.nc")
ds_grid["latitude"].encoding["compression"] = "bzip2"
ds_grid.to_netcdf("/tmp/dummy4.nc")
ds_grid["latitude"].encoding["compression"] = "blosc_lz4"
ds_grid.to_netcdf("/tmp/dummy5.nc")
ds_grid["latitude"].encoding["compression"] = "blosc_zlib"
ds_grid.to_netcdf("/tmp/dummy6.nc")
ds_grid["latitude"].encoding["compression"] = "blosc_lz4hc"
ds_grid.to_netcdf("/tmp/dummy7.nc")

