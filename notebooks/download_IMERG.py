#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:47:27 2021

@author: ghiggi
"""
import os
import datetime 
from dask.diagnostics import ProgressBar

# import sys
# import numpy as np 
# import xarray as xr 
# import h5py 
# import netCDF4
# import pandas as pd
# import dask.array  
# from datetime import timedelta
# from datetime import time

# os.chdir('/home/ghiggi/gpm_api') # change to the 'scripts_GPM.py' directory
### GPM Scripts ####
from gpm_api.io import download_GPM_data

from gpm_api.DPR.DPR_ENV import create_DPR_ENV
from gpm_api.dataset import read_GPM
from gpm_api.dataset import GPM_Dataset, GPM_variables # read_GPM (importing here do)

##----------------------------------------------------------------------------.
### Donwload data 
base_DIR = '/ltenas3/0_Data_Raw'
base_DIR = "/media/ghiggi/New Volume/Data"

username = "gionata.ghiggi@epfl.ch"
 

start_time = datetime.datetime.strptime("2019-07-03 11:30:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2019-07-03 12:30:00", '%Y-%m-%d %H:%M:%S')

 
product = 'IMERG-FR'  # 'IMERG-ER' 'IMERG-LR'

download_GPM_data(base_DIR = base_DIR, 
                  username = username,
                  product = product, 
                  start_time = start_time,
                  end_time = end_time)

download_GPM_data(base_DIR = base_DIR, 
                  GPM_version = 5, 
                  username = username,
                  product = product, 
                  start_time = start_time,
                  end_time = end_time)

GPM_version = 6
product_type = 'RS'
bbox = None
enable_dask=True
chunks='auto'
variables = GPM_variables(product)  
scan_mode = None 
print(variables)
ds = GPM_Dataset(base_DIR = base_DIR,
                 product = product, 
                 product_type = product_type,
                 variables = variables,
                 start_time = start_time,
                 end_time = end_time,
                 bbox = bbox, enable_dask = True, chunks = 'auto') 
print(ds)

variable = "precipitationUncal" 
variable = "precipitationCal"
ds = ds.isel(time=0)

da_is_liquid = ds["probabilityLiquidPrecipitation"] > 90 
da_precip = ds[variable]
da_liquid = ds[variable].where(da_is_liquid)
da_solid = ds[variable].where(~da_is_liquid)

import matplotlib 
from gpm_api.utils.utils_cmap import get_colormap
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}
matplotlib.rc('font', **font)
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'axes.titlesize': 12)
matplotlib.rcParams.update({'axes.labelsize': 12})
matplotlib.rcParams.update({'figure.titlesize': 12})

fontsize = 12 
##----------------------------------------------------------------------------.
#### GPM style - Liquid 
colorscale= "GPM_liquid"
units="mm/h"
ptype="intensity"
cmap, norm, clevs, clevs_str = get_colormap(ptype=ptype,
                                            units=units,
                                            colorscale=colorscale,
                                            bad_color="none") # "none"
if ptype in ["intensity", "depth"]:
    extend = "max"
else:
    extend = "neither"

      
cbar_kwargs = {'ticks': clevs,
               'spacing': 'uniform', 
               'extend': extend,
               'shrink': 0.8,
               'pad': 0.01, 
              }       

import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
crs_proj = ccrs.Robinson()
crs_ref = ccrs.PlateCarree()
crs_proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(18,8), subplot_kw=dict(projection=crs_proj)) 
# - Add Blue Marble  background
ax.background_img(name='BM', resolution='high') # high
# - Add precip map 
p = da_precip.plot.imshow(ax=ax, 
                          x="lon", y="lat", 
                          transform = crs_ref,
                          interpolation="nearest", # "nearest", "bicubic"
                          cmap=cmap, norm=norm, cbar_kwargs=cbar_kwargs)
# - Set title 
p.axes.set_title(pd.to_datetime(da_precip.time.values).strftime("%Y-%m-%d %H:%M"), 
                 size=fontsize, weight='bold')

# - Add colorbar details 
cbar = p.colorbar
_ =  cbar.ax.set_yticklabels(clevs_str)
_ =  cbar.set_label('Precipitation [mm/h]', size=fontsize, weight='bold')

# - Add coastlines 
# ax.coastlines()

# - Add gridlines 
# gl = ax.gridlines(draw_labels=True)
# gl.xlabels_top = False
# gl.ylabels_right = False

fig.savefig("/home/ghiggi/IMERG1.png", edgecolor='black', dpi=600)

##----------------------------------------------------------------------------.
#### GPM solid + Liquid  
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
crs_proj = ccrs.Robinson()
crs_ref = ccrs.PlateCarree()
crs_proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(18,8), subplot_kw=dict(projection=crs_proj)) 
# - Add Blue Marble  background
ax.background_img(name='BM', resolution='high') # high
# - Add Liquid precipitation 
colorscale= "GPM_liquid"
units="mm/h"
ptype="intensity"
cmap, norm, clevs, clevs_str = get_colormap(ptype=ptype,
                                            units=units,
                                            colorscale=colorscale,
                                            bad_color="none") # "none"
if ptype in ["intensity", "depth"]:
    extend = "max"
else:
    extend = "neither"

cbar_kwargs = {'ticks': clevs,
               'spacing': 'uniform', 
               'extend': extend,
               'shrink': 0.8,
               'pad': 0.01, 
              }   

p = da_liquid.plot.imshow(ax=ax, 
                          x="lon", y="lat", 
                          transform = crs_ref,
                          interpolation="nearest", # "nearest", "bicubic",
                          cmap=cmap, norm=norm, cbar_kwargs=cbar_kwargs)
# - Add colorbar details 
cbar = p.colorbar
_ =  cbar.ax.set_yticklabels(clevs_str)
_ =  cbar.set_label('Liquid precipitation [mm/h]', size=fontsize, weight='bold')

# - Add solid precipitation 
colorscale= "GPM_solid"
units="mm/h"
ptype="intensity"
cmap, norm, clevs, clevs_str = get_colormap(ptype=ptype,
                                            units=units,
                                            colorscale=colorscale,
                                            bad_color="none") # "none"
if ptype in ["intensity", "depth"]:
    extend = "max"
else:
    extend = "neither"

cbar_kwargs = {'ticks': clevs,
               'spacing': 'uniform', 
               'extend': extend,
               'shrink': 0.8,
               'pad': 0.01, 
              }   

p = da_solid.plot.imshow(ax=p.axes, 
                         x="lon", y="lat", 
                         transform = crs_ref,
                         interpolation="bilinear", # "nearest", "bicubic"
                         cmap=cmap, norm=norm, cbar_kwargs=cbar_kwargs)

# - Set title 
p.axes.set_title(pd.to_datetime(da_precip.time.values).strftime("%Y-%m-%d %H:%M"), 
                 size=fontsize, weight='bold')

# - Add colorbar details 
cbar1 = p.colorbar
_ =  cbar1.ax.set_yticklabels(clevs_str)
_ =  cbar1.set_label('Solid precipitation [mm/h]', size=fontsize, weight='bold')

# fig.axes[0].get_position()
# fig.axes[2].get_position() # cbar1.ax.get_position()
# fig.axes[1].get_position()

pos0 = fig.axes[2].get_position() # [x0, y0], [x1, y1]
pos1 = fig.axes[1].get_position()
width0 = pos0.width
pos1.x0 = pos0.x1 - 0.03  
pos1.x1 = pos0.x1 - 0.03 + width0  
fig.axes[1].set_position(pos1)

fig.savefig("/home/ghiggi/IMERG2.png", edgecolor='black', dpi=600)
 

##----------------------------------------------------------------------------.
#### MCH style 
colorscale= "pysteps"
units="mm/h"
ptype="intensity"
cmap, norm, clevs, clevs_str = get_colormap(ptype=ptype,
                                            units=units,
                                            colorscale=colorscale,
                                            bad_color="none") # "none"
if ptype in ["intensity", "depth"]:
    extend = "max"
else:
    extend = "neither"

      
cbar_kwargs = {'ticks': clevs,
               'spacing': 'uniform', 
               'extend': extend,
               'shrink': 0.8
               'pad': 0.01, 
              }       


import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
crs_proj = ccrs.Robinson()
crs_ref = ccrs.PlateCarree()
crs_proj = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw=dict(projection=crs_proj)) 
# - Add Blue Marble  background
ax.background_img(name='BM', resolution='high') # high
# - Add precip map 
p = da_precip.plot.imshow(ax=ax, 
                          x="lon", y="lat", 
                          transform = crs_ref,
                          interpolation="bilinear", # "nearest", "bicubic"
                          cmap=cmap, norm=norm, cbar_kwargs=cbar_kwargs)
# - Add colorbar details 
cbar = p.colorbar
_ =  cbar.ax.set_yticklabels(clevs_str)
_ =  cbar.set_label('Precipitation [mm/h]')




##----------------------------------------------------------------------------.
### TODO: 
# Add PlateCarre backgrounds at /home/ghiggi/anaconda3/envs/satpy39/lib/python3.9/site-packages/cartopy/data/raster/natural_earth
# https://neo.gsfc.nasa.gov/
# https://neo.gsfc.nasa.gov/view.php?datasetId=BlueMarbleNG

# bg_dict = {
#    "__comment__": """JSON file specifying the image to use for a given type/name and
#                      resolution. Read in by cartopy.mpl.geoaxes.read_user_background_images.""",
#   "BM": {
#     "__comment__": "Blue Marble Next Generation, July ",
#     "__source__": "https://neo.sci.gsfc.nasa.gov/view.php?datasetId=BlueMarbleNG-TB",
#     "__projection__": "PlateCarree",
#     "low": "BM_July_low.png",
#     "high": "BM_July_high.png"
#   },
# }

# import json
# fpath = "/home/ghiggi/Backgrounds/images.json"
# with open(fpath, 'w', encoding ='utf8') as f:
#     json.dump(bg_dict, f, allow_nan=False)