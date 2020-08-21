#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:48:20 2020

@author: ghiggi
"""
import os
import numpy as np


os.chdir('/home/ghiggi/gpm_api') 
from gpm_api.utils.utils_string import str_extract
from gpm_api.utils.utils_string import str_subset
from gpm_api.utils.utils_string import str_sub 
from gpm_api.utils.utils_string import str_pad 
from gpm_api.utils.utils_string import subset_list_by_boolean
from gpm_api.utils.utils_string import str_detect

from gpm_api.dataset import read_GPM
from gpm_api.dataset import GPM_Dataset, GPM_variables # read_GPM (importing here do)

from gpm_api.io import GPM_DPR_1B_RS_products
from gpm_api.io import GPM_DPR_1B_NRT_products
from gpm_api.io import GPM_DPR_2A_RS_products
from gpm_api.io import GPM_DPR_2A_NRT_products
from gpm_api.io import GPM_DPR_2A_ENV_RS_products
from gpm_api.io import GPM_DPR_2A_ENV_NRT_products
from gpm_api.io import GPM_DPR_RS_products
from gpm_api.io import GPM_DPR_NRT_products

from gpm_api.io import GPM_PMW_1B_RS_products
from gpm_api.io import GPM_PMW_1B_NRT_products
from gpm_api.io import GPM_PMW_1C_RS_products
from gpm_api.io import GPM_PMW_1C_NRT_products
from gpm_api.io import GPM_PMW_2A_GPROF_RS_products
from gpm_api.io import GPM_PMW_2A_GPROF_NRT_products
from gpm_api.io import GPM_PMW_2A_PRPS_RS_products
from gpm_api.io import GPM_PMW_2A_PRPS_NRT_products
from gpm_api.io import GPM_PMW_RS_products
from gpm_api.io import GPM_PMW_NRT_products

from gpm_api.io import GPM_IMERG_NRT_products
from gpm_api.io import GPM_IMERG_RS_products
from gpm_api.io import GPM_IMERG_products

from gpm_api.io import GPM_1B_RS_products
from gpm_api.io import GPM_1B_NRT_products
from gpm_api.io import GPM_1C_RS_products 
from gpm_api.io import GPM_1C_NRT_products
from gpm_api.io import GPM_2A_RS_products
from gpm_api.io import GPM_2A_NRT_products

from gpm_api.io import GPM_RS_products
from gpm_api.io import GPM_NRT_products
from gpm_api.io import GPM_products

from gpm_api.io import get_GPM_PPS_directory
from gpm_api.io import get_GPM_disk_directory

from gpm_api.io import granules_time_info
from gpm_api.io import granules_start_HHMMSS
from gpm_api.io import granules_end_HHMMSS
from gpm_api.io import granules_Dates
from gpm_api.io import get_name_first_daily_granule
from gpm_api.io import get_time_first_daily_granule

from gpm_api.io import check_time
from gpm_api.io import filter_daily_GPM_files
from gpm_api.io import find_daily_GPM_disk_filepaths
from gpm_api.io import find_daily_GPM_PPS_filepaths
from gpm_api.io import find_GPM_files
