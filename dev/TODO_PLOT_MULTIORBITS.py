#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:56:37 2022

@author: ghiggi
"""
import os
import datetime
import numpy as np
from gpm_api.io_future.dataset import open_dataset
from dask.diagnostics import ProgressBar

base_dir = "/home/ghiggi"

#### Define analysis time period 
start_time = datetime.datetime.strptime("2020-08-01 12:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2020-08-10 12:00:00", '%Y-%m-%d %H:%M:%S')

product = "2A-GMI"
product_type = 'RS'
variables = "surfacePrecipitation"

ds = open_dataset(base_dir=base_dir,
                  product=product, 
                  start_time=start_time,
                  end_time=end_time,
                  # Optional 
                  version=7,
                  variables = variables,
                  product_type=product_type,
                  chunks="auto",
                  decode_cf = False, 
                  prefix_group = False)
 
#-----------------------------------------------------------------------------.

 

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

#-----------------------------------------------------------------------------.
#ds = ds.compute()
ds1 = ds 
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
from gpm_api.utils.utils_cmap import * # cmap, norm, cbar_kwargs, clevs_str
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

ds = ds1.isel(along_track=slice(100,300))
ds.lon

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
# ax.set_global()
ax.coastlines()
# - TODO Add swath lines 

# - Add variable field with xarray 
# p  = ds[variable].plot.pcolormesh(x="lon", y="lat",
#                                   ax=ax, 
#                                   transform=ccrs.PlateCarree(),
#                                   cmap=cmap, 
#                                   norm=norm, 
#                                   cbar_kwargs=cbar_kwargs,
#                                   )
# cbar = p.colorbar
# _ = cbar.ax.set_yticklabels(clevs_str)

# p  = ax.pcolor(ds["lon"].data, ds['lat'].data, ds[variable].data,
#                transform=ccrs.PlateCarree(),
#                cmap=cmap, 
#                norm=norm, 
#                # cbar_kwargs=cbar_kwargs)
# )
p  = ax.pcolormesh(ds["lon"].data, ds['lat'].data, ds[variable].data,
                   transform=ccrs.PlateCarree(),
                   cmap=cmap, 
                   norm=norm, 
                   # cbar_kwargs=cbar_kwargs)
)


cbar = p.colorbar
_ = cbar.ax.set_yticklabels(clevs_str)

# pcolormesh --> 'QuadMesh'

#-----------------------------------------------------------------------------.
# Create figure 
fig, ax = plt.subplots(subplot_kw={'projection': crs_proj}, figsize=figsize, dpi=dpi)
# - Add coastlines
ax.set_global()
ax.coastlines()
# - TODO Add swath lines 

# - Add variable field 
# p  = ds[variable].plot.pcolormesh(x="lon", y="lat",
#                                   ax=ax, 
#                                   transform=ccrs.PlateCarree(),
#                                   cmap=cmap,
#                                   )
ax.pcolormesh(ds['lon'].data, ds['lat'].data, ds[variable].data,
              transform=ccrs.PlateCarree(),
              cmap=cmap, norm=norm, 
              )



#-----------------------------------------------------------------------------.
### TODO: add checks when missing coordinates
# - 2A-SSMI-F16
# Coordinate -9999
idx_nan = np.where(ds['lat'].data == -9999)


ds = ds.where(ds['lat'] != -9999, np.nan)  
ds = ds.where(ds['lon'] != -9999, np.nan)  

xr.where(ds.lon == -9999)

x, y = np.where(ds.lon.values == -9999.)
x_unique, counts  = np.unique(x, return_counts=True)
ds.isel(along_track=slice(x_unique[0], x_unique[0]+2)).lon
len(x_unique)/(ds.lon.shape[0]*ds.lon.shape[1])*100