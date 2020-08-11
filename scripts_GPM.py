#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:27:52 2020

@author: ghiggi
"""
import sys
import os
import numpy as np 
import xarray as xr 
import h5py 
import netCDF4
import pandas as pd
import dask.array  

import datetime 
from datetime import timedelta
from datetime import time
from dask.diagnostics import ProgressBar

os.chdir('/home/ghiggi/SatScripts') # change to the 'scripts_GPM.py' directory
### GPM Scripts ####
from gpm_api.download_GPM import download_GPM_data
from gpm_api.gpm_parser import GPM_Dataset, GPM_variables

##----------------------------------------------------------------------------.
### Donwload data 
base_DIR = '/home/ghiggi/tmp'
start_time = datetime.datetime.strptime("2017-01-01 01:02:30", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2017-01-01 03:02:30", '%Y-%m-%d %H:%M:%S')

product = '2A-Ka'
product = '2A-Ku'
product = '2A-DPR'
product = '1B-Ku'
product = '1B-GMI'
product = 'IMERG-FR'
product = '2A-SLH'

download_GPM_data(base_DIR = base_DIR, 
                  product = product, 
                  start_time = start_time,
                  end_time = end_time)

##-----------------------------------------------------------------------------. 
###  Load GPM dataset  
base_DIR = '/home/ghiggi/tmp'
product = '2A-Ku'
scan_mode = 'NS'
start_time = datetime.datetime.strptime("2017-01-01 01:02:30", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2017-01-01 04:02:30", '%Y-%m-%d %H:%M:%S')
variables = GPM_variables(product)   
print(variables)

bbox = [20,50,30,50] 
bbox = [30,35,30,50] 
bbox = None
ds = GPM_Dataset(base_DIR = base_DIR,
                 product=product, 
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
from gpm_api.gpm_parser import GPM_variables_dict, GPM_variables
# Product variables infos 
GPM_variables(product)
GPM_variables_dict(product)

##-----------------------------------------------------------------------------. 

# Products infos 
from gpm_api.download_GPM import GPM_IMERG_available, GPM_NRT_available, GPM_RS_available, GPM_products_available
GPM_products_available()
GPM_IMERG_available()
GPM_NRT_available()
GPM_RS_available()

 
