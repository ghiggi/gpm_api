#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:26:59 2020

@author: ghiggi
"""
import sys
import os
os.chdir('/home/ghiggi/SatScripts')
### GPM Scripts ####
from source.utils_string import *
from source.utils_numpy import *
from source.utils_HDF5 import *
from source.download_GPM import *
from source.gpm_parser import *

import xarray as xr 
import h5py 
import netCDF4
import pandas as pd
import dask.array  

##-----------------------------------------------------------------------------.    
## HDF utils 
filepath = '/home/ghiggi/tmp/1B-Ku/2017/01/01/GPMCOR_KUR_1701010036_0208_016155_1BS_DUB_05A.h5'
hdf = h5py.File(filepath,'r') # h5py._hl.files.File
hdf5_datasets_names(hdf)
hdf5_datasets(hdf)
hdf5_datasets_attrs(hdf)
hdf5_file_attrs(hdf)     # attributes at current level
hdf5_groups_attrs(hdf)    
hdf5_datasets_shape(hdf) # return dict of datasets shape
hdf5_datasets_dtype(hdf) # return dict of datasets dtype

parse_HDF5_GPM_attributes(hdf)
print_hdf5(hdf)   
h5dump(filepath)         
#----------------------------------------------------------------------------.
#### Define settings for testing 
base_DIR = '/home/ghiggi/tmp'
start_time = datetime.datetime.strptime("2017-01-01 00:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2017-01-01 04:02:30", '%Y-%m-%d %H:%M:%S')
bbox = [30,35,30,50] 
bbox = None
enable_dask=True
chunks='auto'
#----------------------------------------------------------------------------.
### 1B
filepath = '/home/ghiggi/tmp/1B-Ku/2017/01/01/GPMCOR_KUR_1701010036_0208_016155_1BS_DUB_05A.h5'
hdf = h5py.File(filepath,'r') # h5py._hl.files.File
product = '1B-Ku'
scan_mode = 'NS'
variables = GPM_variables(product)   
ds = GPM_granule_Dataset(hdf=hdf,
                         product=product, 
                         scan_mode = scan_mode,  
                         variables = variables,
                         enable_dask=True, chunks='auto')
ds
ds = GPM_Dataset(base_DIR = base_DIR,
                 product = product, 
                 scan_mode = scan_mode,  
                 variables = variables,
                 start_time = start_time,
                 end_time = end_time,
                 bbox = bbox, enable_dask = True, chunks = 'auto')
ds     
### 2A - Ku 
filepath = '/home/ghiggi/tmp/2A-Ku/2017/01/01/2A.GPM.Ku.V8-20180723.20170101-S173441-E190714.016166.V06A.HDF5'
hdf = h5py.File(filepath,'r') # h5py._hl.files.File
hdf5_datasets_names(hdf)
hdf5_file_attrs(hdf)
hdf5_datasets_attrs(hdf)

product = '2A-Ku'
scan_mode = 'NS'
variables = 'precipWaterIntegrated' 
variables = 'paramDSD' 
variables = 'zFactorCorrected'
variables = GPM_variables(product)   
ds = GPM_granule_Dataset(hdf=hdf,
                         product=product, 
                         scan_mode = scan_mode,  
                         variables = variables,
                         enable_dask=True, chunks='auto')
ds
ds = GPM_Dataset(base_DIR = base_DIR,
                 product = product, 
                 scan_mode = scan_mode,  
                 variables = variables,
                 start_time = start_time,
                 end_time = end_time,
                 bbox = bbox, enable_dask = True, chunks = 'auto')
ds     
### 2A - Ka
filepath = '/home/ghiggi/tmp/2A-Ka/2017/01/01/2A.GPM.Ka.V8-20180723.20170101-S190715-E203948.016167.V06A.HDF5'
hdf = h5py.File(filepath,'r')
product = '2A-Ka'
scan_mode = 'MS'
scan_mode = 'HS'
variables = 'precipWaterIntegrated' 
variables = 'paramDSD' 
variables = 'zFactorCorrected'
variables = GPM_variables(product)   
ds = GPM_granule_Dataset(hdf=hdf,
                         product=product, 
                         scan_mode = scan_mode,  
                         variables = variables,
                         enable_dask=True, chunks='auto')
ds
ds = GPM_Dataset(base_DIR = base_DIR,
                 product = product, 
                 scan_mode = scan_mode,  
                 variables = variables,
                 start_time = start_time,
                 end_time = end_time,
                 bbox = bbox, enable_dask = True, chunks = 'auto')
ds        
            
# 2A-ENV
filepath = '/home/ghiggi/tmp/2A-ENV-DPR/2017/01/01/2A-ENV.GPM.DPR.V8-20180723.20170101-S003624-E020857.016155.V06A.HDF5'
filepath = '/home/ghiggi/tmp/2A-ENV-DPR/2017/01/01/2A-ENV.GPM.DPR.V8-20180723.20170101-S003624-E020857.016155.V06A.HDF5'
hdf = h5py.File(filepath,'r') # h5py._hl.files.File
hdf5_datasets_names(hdf)
hdf5_file_attrs(hdf)
hdf5_datasets_attrs(hdf)

product = "2A-ENV-DPR"
scan_mode = 'NS'
scan_mode = 'HS'
variables = GPM_variables(product)
ds = GPM_granule_Dataset(hdf=hdf,
                         product=product, 
                         scan_mode = scan_mode,  
                         variables = variables,
                         enable_dask=True, chunks='auto')

ds = GPM_Dataset(base_DIR = base_DIR,
                 product = product, 
                 scan_mode = scan_mode,  
                 variables = variables,
                 start_time = start_time,
                 end_time = end_time,
                 bbox = bbox, enable_dask = True, chunks = 'auto')
ds

# 2A SLH
filepath = '/home/ghiggi/tmp/2A-SLH/2017/01/01/2A.GPM.DPR.GPM-SLH.20170101-S020858-E034131.016156.V06B.HDF5'
hdf = h5py.File(filepath,'r') # h5py._hl.files.File
hdf5_datasets_names(hdf)
hdf5_file_attrs(hdf)
hdf5_datasets_attrs(hdf)

product = "2A-SLH"
scan_mode = 'Swath'
variables = GPM_variables(product)
ds = GPM_granule_Dataset(hdf=hdf,
                         product=product, 
                         scan_mode = scan_mode,  
                         variables = variables,
                         enable_dask=False, chunks='auto')
ds = GPM_Dataset(base_DIR = base_DIR,
                 product = product, 
                 variables = variables,
                 start_time = start_time,
                 end_time = end_time,
                 bbox = bbox, enable_dask = True, chunks = 'auto') 
  
### IMERG 
filepath = '/home/ghiggi/tmp/IMERG-FR/2017/01/01/3B-HHR.MS.MRG.3IMERG.20170101-S010000-E012959.0060.V06B.HDF5'
hdf = h5py.File(filepath,'r') # h5py._hl.files.File
hdf5_file_attrs(hdf)
hdf5_datasets_names(hdf)
hdf5_datasets_attrs(hdf)

product = "IMERG-FR"
scan_mode = 'Grid'
variables = GPM_variables(product)
ds = GPM_granule_Dataset(hdf=hdf,
                         product=product, 
                         scan_mode = scan_mode,  
                         variables = variables,
                         enable_dask=False, chunks='auto')
ds = GPM_Dataset(base_DIR = base_DIR,
                 product = product, 
                 variables = variables,
                 start_time = start_time,
                 end_time = end_time,
                 bbox = bbox, enable_dask = True, chunks = 'auto')
 