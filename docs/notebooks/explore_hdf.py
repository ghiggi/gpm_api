#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:00:25 2022

@author: ghiggi
"""
#-----------------------------------------------------------------------------.
#### Explore GPM file structure 
from gpm_api.utils.utils_HDF5 import *
import h5py
filepath = '/home/ghiggi/GPM_V6/DPR_RS/2A-DPR/2020/10/28/2A.GPM.DPR.V9-20211125.20201028-S075448-E092720.037875.V07A.HDF5'

filepath = '/home/ghiggi/GPM_V6/PMW_RS/1B-TMI/2014/07/01/1B.TRMM.TMI.Tb2021.20140701-S063014-E080236.094691.V07A.HDF5'
filepath = '/home/ghiggi/GPM_V6/DPR_RS/2A-PR/2014/07/01/2A.TRMM.PR.V9-20220125.20140701-S063014-E080236.094691.V07A.HDF5'

hdf = h5py.File(filepath, "r") 
hdf5_file_attrs(hdf) # TODO ERROR--> BUG TO RESOLVE 

hdf5_groups(hdf)
hdf5_groups_names(hdf)

hdf5_datasets(hdf)
hdf5_datasets_names(hdf)

hdf5_datasets_attrs(hdf) 

hdf5_datasets_dtype(hdf) 

#-----------------------------------------------------------------------------.