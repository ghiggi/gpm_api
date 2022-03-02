#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:35:50 2021

@author: ghiggi
"""
import sys
import numpy as np 
import xarray as xr 
import h5py 
import netCDF4
import pandas as pd
import dask.array  

from gpm_api.dataset import *
from gpm_api.io import *
from gpm_api.utils import *

from gpm_api.io import find_GPM_files
from gpm_api.io import GPM_DPR_RS_products
from gpm_api.io import GPM_DPR_2A_ENV_RS_products
from gpm_api.io import GPM_PMW_2A_GPROF_RS_products
from gpm_api.io import GPM_PMW_2A_PRPS_RS_products
from gpm_api.io import GPM_IMERG_products
from gpm_api.io import GPM_products

# For create_GPM_Class
from gpm_api.DPR.DPR import create_DPR
from gpm_api.DPR.DPR_ENV import create_DPR_ENV
from gpm_api.PMW.GMI import create_GMI
from gpm_api.IMERG.IMERG import create_IMERG

from gpm_api.utils.utils_HDF5 import hdf5_file_attrs
from gpm_api.utils.utils_string import str_remove

from gpm_api.utils.utils_string import str_extract
from gpm_api.utils.utils_string import str_subset
from gpm_api.utils.utils_string import str_sub 
from gpm_api.utils.utils_string import str_pad 
from gpm_api.utils.utils_string import str_detect
from gpm_api.utils.utils_string import subset_list_by_boolean


# TO DEBUG 
fpath = "/home/ghiggi/Data/GPM_V6/DPR_RS/2A-DPR/2020/07/03/2A.GPM.DPR.V8-20180723.20200703-S001311-E014544.036050.V06A.HDF5"
enable_dask = True
chunks = "auto"
GPM_version = 6
product_type = 'RS'
n_threads = 10
transfer_tool = "curl"
progress_bar = False
force_download = False
verbose = True
bbox = None
# Date = Dates[0]
flag_first_Date = True
