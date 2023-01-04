#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:56:37 2022

@author: ghiggi
"""
import os
import datetime
import numpy as np
import gpm_api
from dask.diagnostics import ProgressBar

base_dir = "/home/ghiggi"

#### Define analysis time period
start_time = datetime.datetime.strptime("2020-08-01 12:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-08-10 12:00:00", "%Y-%m-%d %H:%M:%S")

product = "2A-GMI"
product = "2A-MHS-METOPB" # non-regular timesteps 
product_type = "RS"
variables = "surfacePrecipitation"

ds = gpm_api.open_dataset(
    base_dir=base_dir,
    product=product,
    start_time=start_time,
    end_time=end_time,
    # Optional
    version=7,
    variables=variables,
    product_type=product_type,
    chunks="auto",
    # decode_cf=True,
    prefix_group=False,
)

ds1 = ds
ds = ds.isel(along_track=slice(0, 10000))

ds = ds.compute()

ds['surfacePrecipitation'] 

ds.gpm_api.plot_map(variable="surfacePrecipitation")

timesteps = ds['time'].values
timesteps = timesteps.astype("M8[s]")
np.unique(np.diff(timesteps), return_counts=True)

# 3 seconds is too much for orbits? 
# - 1 occurence for 2A-GMI 
timesteps.shape
np.where(np.diff(timesteps).astype(int) == 3)
ds.isel(along_track=slice(220049-2, 220049+2))['time']


get_contiguous_scan_slices(ds)
list_slices = get_discontiguous_scan_slices(ds)
ds.isel(along_track=list_slices[0])['gpm_id']





np.where(~is_contiguous)




# get_contiguous_scans 


### TODO: Add decorator to plot_orbit_map to iterate on regular blocks !!!! 
# --> Iterate and check that lon diff not higher than XXXX
#     --> Avoid plotting large pixels due to data interruption


from gpm_api.utils.checks import get_contiguous_scan_slices





from gpm_api.utils.checks import get_nonregular_time_slices
from gpm_api.utils.checks import get_discontiguous_scan_slices
ds.gpm_api.is_regular
ds.gpm_api.has_regular_timesteps
ds.gpm_api.has_contiguous_scans
ds.gpm_api.get_regular_time_slices()
ds.gpm_api.get_contiguous_scan_slices()

# Investigate discontinuities
list_slices = get_discontiguous_scan_slices(ds)
ds.isel(along_track=list_slices[0])['gpm_id']
ds.isel(along_track=list_slices[0]).gpm_api.plot_map(variable="surfacePrecipitation")
# TODO: 
# - plot_footprint
# - plot_mesh


ds.gpm_api.get_contiguous_scan_slices()
 

ds.isel(along_track=list_slices[0])

#### Operational checks 
# - No missing timesteps 
# - No unvalid coordinates
#   --> Mask for plotting 
#   --> Do not mask for crop 

# -----------------------------------------------------------------------------.
### TODO: add checks when missing coordinates
# - 2A-SSMI-F16
# Coordinate -9999
idx_nan = np.where(ds["lat"].data == -9999)


ds = ds.where(ds["lat"] != -9999, np.nan)
ds = ds.where(ds["lon"] != -9999, np.nan)

xr.where(ds.lon == -9999)

x, y = np.where(ds.lon.values == -9999.0)
x_unique, counts = np.unique(x, return_counts=True)
ds.isel(along_track=slice(x_unique[0], x_unique[0] + 2)).lon
len(x_unique) / (ds.lon.shape[0] * ds.lon.shape[1]) * 100
