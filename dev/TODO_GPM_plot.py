#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:56:37 2022

@author: ghiggi
"""
import os
import datetime
from gpm_api.io import download_GPM_data, GPM_PMW_2A_GPROF_RS_products
from gpm_api.dataset import GPM_Dataset, GPM_variables, read_GPM, GPM_Dataset
from dask.diagnostics import ProgressBar

BASE_DIR = "/home/ghiggi"

#### Define analysis time period 
start_time = datetime.datetime.strptime("2020-08-01 12:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2020-08-10 12:00:00", '%Y-%m-%d %H:%M:%S')

product = "2A-GMI"
product_type = 'RS'

ds = GPM_Dataset(
     base_DIR = BASE_DIR,
     product = product,
     variables = ['surfacePrecipitation'],
     start_time = start_time,
     end_time = end_time, 
     scan_mode=None,
     GPM_version=6,
     product_type=product_type,
     bbox=None,
     enable_dask=False,
     chunks="auto",
)
 
#-----------------------------------------------------------------------------.
### TODO: add checks when missing coordinates
# - 2A-SSMI-F16
# Coordinate -9999
ds = ds.where(ds['lat'] != -9999, np.nan)  
ds = ds.where(ds['lon'] != -9999, np.nan)  

xr.where(ds.lon == -9999)

x, y = np.where(ds.lon.values == -9999.)
x_unique, counts  = np.unique(x, return_counts=True)
ds.isel(along_track=slice(x_unique[0], x_unique[0]+2)).lon
len(x_unique)/(ds.lon.shape[0]*ds.lon.shape[1])*100


# GPM_yaml variables 
# ---> 

#-----------------------------------------------------------------------------.
##### Plot PMW Swaths 
# TODO 
# --> Iterate and check that lon diff not higher than XXXX 
#     --> Avoid plotting large pixels due to data interruption 

# pcolormesh works across antimeridian when using cartopy
# - https://github.com/SciTools/cartopy/pull/1622
# - https://github.com/geoschem/GCHP/issues/39
# - https://github.com/SciTools/cartopy/issues/1447
# --> Split at antimeridian, plot with polygon the remaining 
 

import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
from gpm_api.utils.utils_cmap import * # cmap, norm, cbar_kwargs, clevs_str

ds = ds1.isel(along_track=slice(0,5_000))

ds = ds.compute()
# Replace all values equal to -9999 with np.nan
# ds = ds.where(ds['surfacePrecipitation'] > 0.2, 0)  

variable = "surfacePrecipitation"
dpi = 100
figsize = (12,10)
crs_proj = ccrs.PlateCarree()
crs_proj = ccrs.Robinson()

# Create figure 
fig, ax = plt.subplots(subplot_kw={'projection': crs_proj}, figsize=figsize, dpi=dpi)
# - Add coastlines
ax.set_global()
ax.coastlines()
# - TODO Add swath lines 

# - Add variable field 
p  = ds[variable].plot.pcolormesh(x="lon", y="lat",
                                  ax=ax, 
                                  transform=ccrs.PlateCarree(),
                                  cmap=cmap, 
                                  norm=norm, 
                                  cbar_kwargs=cbar_kwargs,
                                  )

cbar = p.colorbar
_ = cbar.ax.set_yticklabels(clevs_str)

#-----------------------------------------------------------------------------.