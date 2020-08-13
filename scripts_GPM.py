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

os.chdir('/home/ghiggi/gpm_api') # change to the 'scripts_GPM.py' directory
### GPM Scripts ####
from gpm_api.io import download_GPM_data
from gpm_api.dataset import read_GPM
from gpm_api.dataset import GPM_Dataset, GPM_variables # read_GPM (importing here do)

##----------------------------------------------------------------------------.
### Donwload data 
base_DIR = '/home/ghiggi/tmp'
username = "gionata.ghiggi@epfl.ch"
start_time = datetime.datetime.strptime("2020-08-09 15:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2020-08-01 16:00:00", '%Y-%m-%d %H:%M:%S')
start_time = datetime.datetime.strptime("2017-01-01 01:02:30", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2017-01-01 04:02:30", '%Y-%m-%d %H:%M:%S')
 
product = '2A-Ka'
product = '2A-Ku'
product = '2A-DPR'
product = '1B-Ku'
product = '1B-GMI'
product = 'IMERG-FR'
product = '2A-SLH'

download_GPM_data(base_DIR = base_DIR, 
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
# Not present
# DPR MS SLV : precipRate, paramDSD 
# DPR MS DSD : phase  



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
from gpm_api.io import GPM_IMERG_available, GPM_NRT_available, GPM_RS_available, GPM_products_available
GPM_products_available()
GPM_IMERG_available()
GPM_NRT_available()
GPM_RS_available()
##-----------------------------------------------------------------------------. 

base_DIR = '/home/ghiggi/tmp'
username = "gionata.ghiggi@epfl.ch"
start_time = datetime.datetime.strptime("2017-01-01 00:40:30", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2017-01-01 02:00:30", '%Y-%m-%d %H:%M:%S')

product = '2A-SL'
product = '2A-Ku'
product = '1B-Ku'
product = 'IMERG-FR'
product = '2A-DPR'
DPR = read_GPM(base_DIR = base_DIR,
               product = product, 
               start_time = start_time,
               end_time = end_time)
               # scan_mode = scan_mode,  # by default all 
               # variables = variables,  # by default all available
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


  
         
         