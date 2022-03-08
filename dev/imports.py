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

from gpm_api.io import (
    GPM_1B_NRT_pattern_dict,
    GPM_1B_NRT_products,
    GPM_1B_RS_pattern_dict,
    GPM_1B_RS_products,
    GPM_1C_NRT_products,
    GPM_1C_RS_products,
    GPM_2A_NRT_pattern_dict,
    GPM_2A_NRT_products,
    GPM_2A_RS_pattern_dict,
    GPM_2A_RS_products,
    GPM_DPR_1B_NRT_products,
    GPM_DPR_1B_RS_pattern_dict,
    GPM_DPR_1B_RS_products,
    GPM_DPR_2A_ENV_NRT_products,
    GPM_DPR_2A_ENV_RS_products,
    GPM_DPR_2A_NRT_pattern_dict,
    GPM_DPR_2A_NRT_products,
    GPM_DPR_2A_RS_pattern_dict,
    GPM_DPR_2A_RS_products,
    GPM_DPR_NRT_pattern_dict,
    GPM_DPR_NRT_products,
    GPM_DPR_RS_pattern_dict,
    GPM_DPR_RS_products,
    GPM_IMERG_NRT_pattern_dict,
    GPM_IMERG_NRT_products,
    GPM_IMERG_RS_pattern_dict,
    GPM_IMERG_RS_products,
    GPM_IMERG_pattern_dict,
    GPM_IMERG_products,
    GPM_NRT_products,
    GPM_NRT_products_pattern_dict,
    GPM_PMW_1B_NRT_pattern_dict,
    GPM_PMW_1B_NRT_products,
    GPM_PMW_1B_RS_pattern_dict,
    GPM_PMW_1B_RS_products,
    GPM_PMW_1C_NRT_pattern_dict,
    GPM_PMW_1C_NRT_products,
    GPM_PMW_1C_RS_pattern_dict,
    GPM_PMW_1C_RS_products,
    GPM_PMW_2A_GPROF_NRT_pattern_dict,
    GPM_PMW_2A_GPROF_NRT_products,
    GPM_PMW_2A_GPROF_RS_pattern_dict,
    GPM_PMW_2A_GPROF_RS_products,
    GPM_PMW_2A_PRPS_NRT_pattern_dict,
    GPM_PMW_2A_PRPS_NRT_products,
    GPM_PMW_2A_PRPS_RS_pattern_dict,
    GPM_PMW_2A_PRPS_RS_products,
    GPM_PMW_NRT_pattern_dict,
    GPM_PMW_NRT_products,
    GPM_PMW_RS_pattern_dict,
    GPM_PMW_RS_products,
    GPM_RS_products,
    GPM_RS_products_pattern_dict,
    GPM_products,
    GPM_products_pattern_dict,
    find_GPM_files,
    check_Date,
    check_GPM_version,
    check_HHMMSS,
    check_product,
    check_product_type,
    check_time,
    concurrent,
    curl_cmd,
    datetime,
    download_GPM_data,
    download_daily_GPM_data,
    filter_GPM_query,
    filter_daily_GPM_files,
    find_GPM_files,
    find_daily_GPM_PPS_filepaths,
    find_daily_GPM_disk_filepaths,
    get_GPM_PPS_directory,
    get_GPM_disk_directory,
    get_name_first_daily_granule,
    get_time_first_daily_granule,
    granules_Dates,
    granules_end_HHMMSS,
    granules_start_HHMMSS,
    granules_time_info,
    is_empty,
    is_not_empty,
    run,
    str_detect,
    str_extract,
    str_pad,
    str_sub,
    str_subset,
    subset_list_by_boolean,
    wget_cmd,
)

from gpm_api.dataset import (
     GPM_DPR_2A_ENV_RS_products,
     GPM_DPR_RS_products,
     GPM_Dataset,
     GPM_IMERG_products,
     GPM_PMW_2A_GPROF_RS_products,
     GPM_PMW_2A_PRPS_RS_products,
     GPM_granule_Dataset,
     GPM_products,
     GPM_variables,
     GPM_variables_dict,
    check_GPM_version,
    check_bbox,
    check_product,
    check_scan_mode,
    check_variables,
    create_DPR,
    create_DPR_ENV,
    create_GMI,
    create_GPM_class,
    create_IMERG,
    find_GPM_files,
    flip_boolean,
    hdf5_file_attrs,
    initialize_scan_modes,
    is_empty,
    is_not_empty,
    parse_GPM_ScanTime,
    read_GPM,
    remove_dict_keys,
    str_remove,
    subset_dict,
)

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
# fpath = "/home/ghiggi/Data/GPM_V6/DPR_RS/2A-DPR/2020/07/03/2A.GPM.DPR.V8-20180723.20200703-S001311-E014544.036050.V06A.HDF5"
fpath = "/ltenas3/0_Data/GPM_V6/DPR_RS/2A-DPR/2020/07/04/2A.GPM.DPR.V8-20180723.20200704-S223000-E000233.036080.V06A.HDF5'
enable_dask = True
chunks = "auto"
GPM_version = 6
product_type = "RS"
n_threads = 10
transfer_tool = "curl"
progress_bar = False
force_download = False
verbose = True
bbox = None
# Date = Dates[0]
flag_first_Date = True
