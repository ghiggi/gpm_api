#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 19:36:49 2023

@author: ghiggi
"""
import datetime

import numpy as np

import gpm_api

#### Define analysis time period
start_time = datetime.datetime.strptime("2020-08-01 12:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-08-10 12:00:00", "%Y-%m-%d %H:%M:%S")

product = "2A-GMI"
product_type = "RS"
variable = "surfacePrecipitation"

ds = gpm_api.open_dataset(
    product=product,
    start_time=start_time,
    end_time=end_time,
    # Optional
    version=None,
    variables=variable,
    product_type=product_type,
    chunks={},
    prefix_group=False,
)

ds = ds.compute()

ds.isel(along_track=slice(0, 10000)).gpm_api.plot_map(variable=variable)

###---------------------------------------------------------------------------.
from gpm_api.utils.checks import get_contiguous_scan_slices, get_discontiguous_scan_slices

ds.gpm_api.is_regular
ds.gpm_api.has_regular_timesteps
ds.gpm_api.has_contiguous_scans
ds.gpm_api.get_regular_time_slices()
ds.gpm_api.get_contiguous_scan_slices()

#### Operational checks
gpm_api.check_regular_timesteps(ds)
gpm_api.check_contiguous_scans(ds)
gpm_api.check_valid_geolocation(ds)

###---------------------------------------------------------------------------.
# Mock discontinguous and test plotting
indices = np.arange(0, 1000)
ds.isel(along_track=indices).gpm_api.plot_map(variable=variable)
indices = np.append(indices[0:200], indices[800:1000])
ds.isel(along_track=indices).gpm_api.get_contiguous_scan_slices()
ds.isel(along_track=indices).gpm_api.plot_map(variable=variable)

###---------------------------------------------------------------------------.
# Investigate PMW scan discontinuities
list_slices = get_discontiguous_scan_slices(ds)
ds.isel(along_track=list_slices[0])["gpm_id"]  # at granule change
ds.isel(along_track=list_slices[0]).gpm_api.plot_map(variable=variable)

slice_problem_occurrence = list_slices[0]
print(slice_problem_occurrence)
da = ds[variable].isel(along_track=slice_problem_occurrence)  # size 2
get_contiguous_scan_slices(da)  # not possible to discern if only 2 scans
p = da.gpm_api.plot_map_mesh(edgecolors="r")

slice_enlarged = slice(slice_problem_occurrence.start - 1, slice_problem_occurrence.stop + 1)
print(slice_enlarged)
da = ds[variable].isel(along_track=slice_enlarged)  # size 4
get_contiguous_scan_slices(da)  # 0-1 ... and 2-3
da.gpm_api.plot_map_mesh(ax=p.axes)

###---------------------------------------------------------------------------.
