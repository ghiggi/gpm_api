#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:48:20 2020

@author: ghiggi
"""
import os
import numpy as np
import datetime
import subprocess
import xarray
import h5py
import trollsift  
 
from gpm_api.utils.utils_string import (
    str_extract,
    str_subset,
    str_sub,
    str_pad,
    subset_list_by_boolean, 
    str_detect,
)
    
from gpm_api.io.products import (
    
    GPM_products,
    GPM_IMERG_products,
    
    GPM_NRT_products,
    GPM_RS_products, 
    
    GPM_DPR_NRT_products,
    GPM_PMW_NRT_products,
    GPM_IMERG_NRT_products,
    GPM_CMB_NRT_products,

    GPM_DPR_RS_products,
    GPM_PMW_RS_products,
    GPM_CMB_RS_products,
    GPM_IMERG_RS_products,
    
    GPM_DPR_1B_NRT_products,
    GPM_PMW_1C_NRT_products,
    GPM_DPR_2A_NRT_products,
    GPM_DPR_2A_ENV_NRT_products,
    GPM_PMW_1B_NRT_products,
    GPM_PMW_2A_GPROF_NRT_products,
    GPM_PMW_2A_PRPS_NRT_products,
    GPM_CMB_2B_NRT_products,  
    GPM_IMERG_NRT_products,   

    GPM_DPR_1B_RS_products,
    GPM_DPR_2A_RS_products, 
    GPM_DPR_2A_ENV_RS_products,
    GPM_PMW_1A_RS_products, 
    GPM_PMW_1B_RS_products,
    GPM_PMW_1C_RS_products, 
    GPM_PMW_2A_PRPS_RS_products, 
    GPM_PMW_2A_GPROF_RS_products,
    GPM_CMB_2B_RS_products,
    GPM_IMERG_RS_products,
   
    GPM_1B_NRT_products,
    GPM_1C_NRT_products,
    GPM_2A_NRT_products,
   
    GPM_1B_RS_products,
    GPM_1C_RS_products, 
    GPM_2A_RS_products, 
 
)
 
from gpm_api.io.checks import (
    check_time,
    check_version,
    check_product,
    check_product_type,
    check_variables, 
    check_groups, 
    check_date,
    check_hhmmss,
    check_scan_mode, 
    check_bbox,
    check_filepaths,
    is_not_empty,
    is_empty,
)

from gpm_api.io.directories import (
    get_GPM_PPS_directory,
    get_GPM_disk_directory,
)

from gpm_api.io.filter import (
    filter_daily_filepaths,
    granules_time_info, 
    granules_start_hhmmss,
    granules_end_hhmmss,
    granules_dates,
    get_name_first_daily_granule,
    get_time_first_daily_granule, 
    granules_end_hhmmss, 
)

from gpm_api.io.disk import (
    find_daily_filepaths,
    find_filepaths,
)
from gpm_api.io.pps import (
    find_pps_daily_filepaths,
    # find_pps_filepaths, # TODO 
)

from gpm_api.io.download import (
    wget_cmd,
    curl_cmd,
    run,
    filter_download_list, 
    download_daily_data,
    download_data,
)