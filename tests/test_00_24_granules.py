#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 18:23:58 2020

@author: ghiggi
"""
import os
import datetime
os.chdir('/home/ghiggi/gpm_api') # change to the 'scripts_GPM.py' directory
 
from gpm_api.io import find_daily_GPM_PPS_filepaths
from gpm_api.utils.utils_string import str_extract
from gpm_api.utils.utils_string import str_subset
from gpm_api.utils.utils_string import str_sub 
from gpm_api.utils.utils_string import str_pad 

### Donwload data 
base_DIR = '/home/ghiggi/tmp'
username = "gionata.ghiggi@epfl.ch"
product = "2A-Ka"
Date = datetime.date.fromisoformat("2014-08-09")

start_HHMMSS = "000000"
end_HHMMSS = "240000"
 

start_HHMMSS = datetime.time.fromisoformat("00:00:00")
end_HHMMSS = datetime.time.fromisoformat("23:59:00")

file_list = find_daily_GPM_PPS_filepaths(username=username,
                                         product = product,
                                         Date = Date,
                                         start_HHMMSS = start_HHMMSS, 
                                         end_HHMMSS = end_HHMMSS,
                                         product_type = 'RS',
                                         GPM_version = 6)

l_files = file_list
l_s_HHMMSS = str_sub(str_extract(l_files,"S[0-9]{6}"), 1)
l_e_HHMMSS = str_sub(str_extract(l_files,"E[0-9]{6}"), 1)
# Subset specific time period     
# - Retrieve start_HHMMSS and endtime of GPM granules products (execept JAXA 1B reflectivities)
if product not in GPM_DPR_1B_RS_products():
    l_s_HHMMSS = str_sub(str_extract(l_files,"S[0-9]{6}"), 1)
    l_e_HHMMSS = str_sub(str_extract(l_files,"E[0-9]{6}"), 1)
# - Retrieve start_HHMMSS and endtime of JAXA 1B reflectivities
else: 
    # Retrieve start_HHMMSS of granules   
    l_s_HHMM = str_sub(str_extract(l_files,"[0-9]{10}"),6) 
    l_e_HHMM = str_sub(str_extract(l_files,"_[0-9]{4}_"),1,5) 
    l_s_HHMMSS = str_pad(l_s_HHMM, width=6, side="right",pad="0")
    l_e_HHMMSS = str_pad(l_e_HHMM, width=6, side="right",pad="0")
        
        
        
get_time_start_first_granule 
get_time_start_last_granule 
get_time_end_first_granule 
get_time_end_last_granule 
get_previous_day(Date)
get_next_day(Date)

            find_daily_GPM_PPS_filepaths(username=username,
                                        product = product,
                                        Date = get_previous_day(Date)
                                        start_HHMMSS = start_HHMMSS, 
                                        end_HHMMSS = end_HHMMSS,
                                        product_type = 'RS',
                                        GPM_version = 6)
