#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:23:49 2023

@author: ghiggi
"""
import os
import gpm_api

# import warnings
# warnings.filterwarnings("ignore")

#### Load GPM Swath
product_type = "RS"
product = "2A-DPR"
variables = ["precipRateNearSurface", "dataQuality", "SCorientation"]
version = 7
scan_mode = "FS"
chunks = "auto"  # otherwise concatenating datasets is very slow !
groups = None
decode_cf = False
prefix_group = False
verbose = False

# ------------------------------
### GPM DPR
# After 2019 change scan
filepath = "/ltenas8/data/GPM/RS/V07/RADAR/2A-DPR/2020/08/19/2A.GPM.DPR.V9-20211125.20200819-S092131-E105404.036787.V07A.HDF5"
ds_gpm = gpm_api.open_granule(
    filepath,
    scan_mode=scan_mode,
    variables=variables,
    # groups=groups, # TODO
    decode_cf=False,
    prefix_group=prefix_group,
    chunks=chunks,
)


ds_gpm["precipRateNearSurface"].gpm_api.plot_map()

xr_obj = ds_gpm
xr_obj.gpm_api.get_slices_valid_geolocation()  # till 7434
xr_obj.gpm_api.get_slices_contiguous_scans()  # till 7435
xr_obj.gpm_api.is_regular
xr_obj.isel(along_track=7434)["lon"]
xr_obj.isel(along_track=7435)["lon"]
ds_gpm.isel(along_track=slice(0, 7435, None))["precipRateNearSurface"].gpm_api.plot_map()
ds_gpm.isel(along_track=slice(0, 7435, None)).gpm_api.is_regular

from gpm_api.utils.checks import (
    check_valid_geolocation,
    get_slices_non_contiguous_scans,
    get_slices_non_wobbling_swath,
)

check_valid_geolocation(xr_obj)

xr_obj.gpm_api.get_slices_contiguous_scans(min_size=2)
xr_obj.gpm_api.get_slices_contiguous_scans(
    min_size=1
)  # seems buggy but is because can not verify if last element is contiguous
get_slices_non_contiguous_scans(xr_obj)

get_slices_non_wobbling_swath(xr_obj, threshold=100)
