#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:27:52 2020

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
os.chdir('/home/ghiggi/Python_Packages/gpm_api') # change to the 'scripts_GPM.py' directory
os.chdir('/ltenas3/0_Projects/gpm_api')
### GPM Scripts ####
from gpm_api.io import download_GPM_data

from gpm_api.DPR.DPR_ENV import create_DPR_ENV
from gpm_api.dataset import read_GPM
from gpm_api.dataset import GPM_Dataset, GPM_variables # read_GPM (importing here do)


# allows access to all ports in the range of 64000-65000 for the DNS names of: 
#  ‘arthurhou.pps.eosdis.nasa.gov’ and ‘arthurhouftps.pps.eosdis.nasa.gov’.
# open/allow access to all ports in the range of 64000-65000 for the system
# ‘arthurhouftps.pps.eosdis.nasa.gov’  (ftps)
# ‘arthurhou.pps.eosdis.nasa.gov’ (ftps)
# Python 3 ftplib 
# curl ftps
# Explicit FTPS

##----------------------------------------------------------------------------.
### Donwload data 
base_DIR = '/home/ghiggi/Data'
base_DIR = '/ltenas3/0_Data'
username = "gionata.ghiggi@epfl.ch"
start_time = datetime.datetime.strptime("2020-07-01 00:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2020-09-01 00:00:00", '%Y-%m-%d %H:%M:%S')
# start_time = datetime.datetime.strptime("2017-01-01 01:02:30", '%Y-%m-%d %H:%M:%S')
# end_time = datetime.datetime.strptime("2017-01-01 04:02:30", '%Y-%m-%d %H:%M:%S')

# start_time = datetime.datetime.strptime("2014-08-09 00:00:00", '%Y-%m-%d %H:%M:%S')
# end_time = datetime.datetime.strptime("2014-08-09 03:00:00", '%Y-%m-%d %H:%M:%S')

product = '2A-Ka'
product = '2A-Ku'
product = '2A-DPR'
product = '1B-Ku'
product = '1B-GMI'
product = 'IMERG-FR'
product = '2A-SLH'

product = '2A-DPR'
download_GPM_data(base_DIR = base_DIR, 
                  username = username,
                  product = product, 
                  start_time = start_time,
                  end_time = end_time,
                  progress_bar = True,
                  n_threads = 8,
                  transfer_tool = "curl")

download_GPM_data(base_DIR = base_DIR, 
                  GPM_version = 5, 
                  username = username,
                  product = product, 
                  start_time = start_time,
                  end_time = end_time)

##-----------------------------------------------------------------------------. 
###  Load GPM dataset  
base_DIR = '/home/ghiggi/tmp'
product = '2A-DPR'
scan_mode = 'MS'
variables = GPM_variables(product)   
print(variables)

bbox = [20,50,30,50] 
bbox = [30,35,30,50] 
bbox = None
ds = GPM_Dataset(base_DIR = base_DIR,
                 product = product, 
                 scan_mode = scan_mode,  # only necessary for 1B and 2A Ku/Ka/DPR
                 variables = variables,
                 start_time = start_time,
                 end_time = end_time,
                 bbox = bbox, enable_dask = True, chunks = 'auto') 
print(ds)

## Test NRT 
start_time = datetime.datetime.strptime("2020-08-17 00:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2020-08-17 17:00:00", '%Y-%m-%d %H:%M:%S')
product = '2A-DPR'
scan_mode = 'MS'
ds = GPM_Dataset(base_DIR = base_DIR,
                 product = product, 
                 product_type = 'NRT',
                 scan_mode = scan_mode,  # only necessary for 1B and 2A Ku/Ka/DPR
                 variables = variables,
                 start_time = start_time,
                 end_time = end_time,
                 bbox = bbox, enable_dask = True, chunks = 'auto') 
print(ds)









# - Really load data in memory 
with ProgressBar():
    ds.compute()

##-----------------------------------------------------------------------------. 
# Some others utils functions
from gpm_api.dataset import GPM_variables_dict, GPM_variables
# Product variables infos 
GPM_variables(product)
GPM_variables_dict(product)

##-----------------------------------------------------------------------------. 

# Products infos 
from gpm_api.io import GPM_IMERG_products, GPM_NRT_products, GPM_RS_products, GPM_products_products
GPM_products_products()
GPM_IMERG_products()
GPM_NRT_products()
GPM_RS_products()
##-----------------------------------------------------------------------------. 
from gpm_api.dataset import read_GPM
from gpm_api.dataset import GPM_Dataset, GPM_variables # read_GPM (importing here do)

##-------------
base_DIR = '/home/ghiggi/tmp'
username = "gionata.ghiggi@epfl.ch"
start_time = datetime.datetime.strptime("2017-01-01 00:40:30", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2017-01-01 02:00:30", '%Y-%m-%d %H:%M:%S')
start_time = datetime.datetime.strptime("2014-08-09 00:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2014-08-09 03:00:00", '%Y-%m-%d %H:%M:%S')

product = '2A-SLH'
product = '2A-Ku'
product = '1B-Ku'
product = 'IMERG-FR'
product = '2A-DPR'
DPR = read_GPM(base_DIR = base_DIR,
               product = product, 
               start_time = start_time,
               end_time = end_time)
               # scan_mode = scan_mode,  # by default all 
               # variables = variables,  # by default all products
               # bbox = bbox, 
               # enable_dask = True, 
               # chunks = 'auto') 
DPR.NS
DPR.HS
DPR.MS

DPR.retrieve_ENV() # need to debug the import in DPR/DPR.py
DPR.NS



##----------------------------------------------------------------------------.
# To improve retrieve_ENV()
product = '2A-ENV-DPR'
ENV = read_GPM(base_DIR = base_DIR,
               product = product, 
               start_time = start_time,
               end_time = end_time)
ENV.NS
ENV.HS

DPR.NS
ENV.NS





















## Searching for maximum value and return bbox

da = DPR.NS.zFactorCorrectedNearSurface
da = da.compute()

da.
da.argmax()
da.idxmax()





import matplotlib.pyplot as plt

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.patheffects as PathEffects
import cartopy.io.shapereader as shpreader
from cartopy.io.shapereader import Reader
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.colors as colors
import matplotlib.patheffects as PathEffects

ds = DPR.NS
#make figure
f_size = 10
fig = plt.figure(figsize=(1.6*f_size, 0.9*f_size))
#add the map
ax = fig.add_subplot(1, 1, 1,projection=ccrs.PlateCarree())

ax.add_feature(cartopy.feature.OCEAN.with_scale('50m'))
ax.add_feature(cartopy.feature.LAND.with_scale('50m'), edgecolor='black',lw=0.5,facecolor=[0.95,0.95,0.95])
pm = ax.scatter(ds.lon,ds.lat,c=ds.zFactorCorrectedNearSurface,vmin=12,vmax=50,s=0.1,zorder=2)
plt.colorbar(pm,ax=ax,shrink=0.33)

def add_shape(filepath, projection):
    return cfeature.ShapelyFeature(Reader(filepath).geometries(),
                                   projection, facecolor='none')

def add_grid(ax, projection,
            west, east, south, north, lon_d, lat_d,
            xlabels_bottom=True, xlabels_top=False,
            ylabels_left=True, ylabels_right=False,
            xlabel_size=None, ylabel_size=None,
            xlabel_color=None, ylabel_color=None,
            linewidth=0.5, grid_color='k', zorder=3):
    
    ax.set_extent([west, east, south, north], crs=projection)
    gl = ax.gridlines(crs=projection, draw_labels=True,
                      linewidth=linewidth, color=grid_color, linestyle='--', zorder=zorder)

    gl.xlabels_bottom = xlabels_bottom; gl.xlabels_top  = xlabels_top
    gl.ylabels_left   = ylabels_left;  gl.ylabels_right = ylabels_right
    gl.xformatter     = LONGITUDE_FORMATTER
    gl.yformatter     = LATITUDE_FORMATTER
    gl.xlocator       = mticker.FixedLocator(np.arange(west-lon_d, east+lon_d, lon_d))
    gl.ylocator       = mticker.FixedLocator(np.arange(south-lon_d, north+lon_d, lon_d))
    
    if xlabel_size:
        gl.xlabel_style = {'size': xlabel_size}
    if ylabel_size:
        gl.ylabel_style = {'size': ylabel_size}
    if xlabel_color:
        gl.xlabel_color = {'color': xlabel_color}
    if ylabel_color:
        gl.ylabel_color = {'color': ylabel_color}


import xarray as xr
# Check for coordinates, where not same
# - Discard missing scan , add NaN to missing scan 
xr.merge([ENV.NS,DPR.NS])
xr.merge([ENV.HS,DPR.HS])
DPR.NS.combine_first(ENV.NS)
DPR.NS.combine_first(ENV.NS)

# 2A-ENV MS ... not present...need to subset

# SLH --> DPR.Swath 
# start_time, end_time correspond to request, not actual retrieved...

product = '2A-DPR'
  

import yaml 
from gpm_api.dataset import GPM_variables_dict
from gpm_api.io import GPM_products_products
from gpm_api.dataset import initialize_scan_modes



product =  "IMERG-LR"
scan_mode = "Grid"
d = GPM_variables_dict(product, scan_mode)
print(d['PMWprecipSource']['description'])

# Xradar folder????
    
p = da.plot.pcolormesh(x="lon", y="lat",
                       infer_intervals=True, # plot at the pixel center 
                       subplot_kws=dict(projection=ccrs.PlateCarree()))
# p.axes.scatter(lons, lats, 
#                transform=ccrs.PlateCarree())
p.axes.coastlines()
p.axes.gridlines(draw_labels=True)
plt.show()

    
    
    
