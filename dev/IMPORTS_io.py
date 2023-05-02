#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:48:20 2020

@author: ghiggi
"""
import datetime
import os
import subprocess

import h5py
import numpy as np
import trollsift
import xarray

import gpm_api
from gpm_api.io.checks import (
    check_bbox,
    check_date,
    check_filepaths,
    check_groups,
    check_hhmmss,
    check_product,
    check_product_type,
    check_scan_mode,
    check_time,
    check_variables,
    check_version,
    is_empty,
    is_not_empty,
)
from gpm_api.io.directories import get_disk_directory, get_pps_directory
from gpm_api.io.disk import find_daily_filepaths, find_filepaths
from gpm_api.io.download import (
    curl_cmd,
    download_daily_data,
    download_data,
    filter_download_list,
    run,
    wget_cmd,
)
from gpm_api.io.filter import (
    filter_daily_filepaths,
    get_name_first_daily_granule,
    get_time_first_daily_granule,
    granules_dates,
    granules_end_hhmmss,
    granules_start_hhmmss,
    granules_time_info,
)
from gpm_api.io.pps import find_pps_daily_filepaths  # find_pps_filepaths, # TODO
from gpm_api.io.products import (
    GPM_1B_NRT_products,
    GPM_1B_RS_products,
    GPM_1C_NRT_products,
    GPM_1C_RS_products,
    GPM_2A_NRT_products,
    GPM_2A_RS_products,
    GPM_CMB_2B_NRT_products,
    GPM_CMB_2B_RS_products,
    GPM_CMB_NRT_products,
    GPM_CMB_RS_products,
    GPM_DPR_1B_NRT_products,
    GPM_DPR_1B_RS_products,
    GPM_DPR_2A_ENV_NRT_products,
    GPM_DPR_2A_ENV_RS_products,
    GPM_DPR_2A_NRT_products,
    GPM_DPR_2A_RS_products,
    GPM_DPR_NRT_products,
    GPM_DPR_RS_products,
    GPM_IMERG_NRT_products,
    GPM_IMERG_products,
    GPM_IMERG_RS_products,
    GPM_NRT_products,
    GPM_PMW_1A_RS_products,
    GPM_PMW_1B_NRT_products,
    GPM_PMW_1B_RS_products,
    GPM_PMW_1C_NRT_products,
    GPM_PMW_1C_RS_products,
    GPM_PMW_2A_GPROF_NRT_products,
    GPM_PMW_2A_GPROF_RS_products,
    GPM_PMW_2A_PRPS_NRT_products,
    GPM_PMW_2A_PRPS_RS_products,
    GPM_PMW_NRT_products,
    GPM_PMW_RS_products,
    GPM_products,
    GPM_RS_products,
)
from gpm_api.utils.utils_string import (
    str_detect,
    str_extract,
    str_pad,
    str_sub,
    str_subset,
    subset_list_by_boolean,
)
