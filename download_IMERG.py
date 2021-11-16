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
username = "gionata.ghiggi@epfl.ch"

start_time = datetime.datetime.strptime("2019-06-15 00:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2019-06-15 23:00:00", '%Y-%m-%d %H:%M:%S')


product = 'IMERG-FR'  # 'IMERG-ER' 'IMERG-LR'
product = 'IMERG-LR'
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