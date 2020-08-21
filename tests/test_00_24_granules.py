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
from gpm_api.io import find_daily_GPM_disk_filepaths
from gpm_api.io import find_GPM_files
from gpm_api.utils.utils_string import str_extract
from gpm_api.utils.utils_string import str_subset
from gpm_api.utils.utils_string import str_sub 
from gpm_api.utils.utils_string import str_pad 

base_DIR = '/home/ghiggi/tmp'
username = "gionata.ghiggi@epfl.ch"
product = "2A-DPR"
start_HHMMSS = "000000"
end_HHMMSS = "020000"
GPM_version = 6

## Still BUGGY for the following case
# - Need to take only the previous day 
start_HHMMSS = "000000"
end_HHMMSS = "001000"
# - Need to take only a single granule in current day 
start_HHMMSS = "020028"
end_HHMMSS = "020030"
# - Need to take last granule 
start_HHMMSS = "235000"
end_HHMMSS = "240000"


# start_HHMMSS and end_HHMMSS in datetime.time format
start_HHMMSS = datetime.time.fromisoformat("00:00:00")
end_HHMMSS = datetime.time.fromisoformat("23:59:00")
##----------------------------------------------------------------------------.
### Retrieve RS filepaths 
Date = datetime.date.fromisoformat("2014-08-09") 
product_type = 'RS'

(server_paths, disk_paths) = find_daily_GPM_PPS_filepaths(username = username,
                                                          base_DIR = base_DIR,
                                                          product = product,
                                                          Date = Date,
                                                          start_HHMMSS = start_HHMMSS, 
                                                          end_HHMMSS = end_HHMMSS,
                                                          product_type = product_type,
                                                          GPM_version = GPM_version,
                                                          flag_first_Date = True)
print(server_paths)
print(disk_paths)

find_daily_GPM_disk_filepaths( base_DIR = base_DIR,
                               product = product,
                               Date = Date,
                               start_HHMMSS = start_HHMMSS, 
                               end_HHMMSS = end_HHMMSS,
                               product_type = product_type,
                               GPM_version = GPM_version,
                               flag_first_Date = True)
##----------------------------------------------------------------------------.
### Retrieve NRT filepaths 
Date = datetime.date.fromisoformat("2020-08-16")
product_type = 'NRT'
 
(server_paths, disk_paths) = find_daily_GPM_PPS_filepaths(username = username,
                                                          base_DIR = base_DIR,
                                                          product = product, 
                                                          product_type = product_type,
                                                          GPM_version = GPM_version,
                                                          Date = Date, 
                                                          start_HHMMSS = start_HHMMSS, 
                                                          end_HHMMSS = end_HHMMSS,
                                                          flag_first_Date = True)
print(server_paths)
print(disk_paths)

find_daily_GPM_disk_filepaths( base_DIR = base_DIR,
                               product = product,
                               Date = Date,
                               start_HHMMSS = start_HHMMSS, 
                               end_HHMMSS = end_HHMMSS,
                               product_type = product_type,
                               GPM_version = GPM_version,
                               flag_first_Date = True)
         
### find_GPM_files()                                  
start_time = datetime.datetime.fromisoformat("2020-08-15 00:00:00")
end_time = datetime.datetime.fromisoformat("2020-08-17 00:00:00")
find_GPM_files(base_DIR = base_DIR, 
               product = product,
               start_time = start_time,
               end_time = end_time,
               product_type = product_type,
               GPM_version = GPM_version)
 


 