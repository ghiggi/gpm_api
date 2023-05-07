#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 18:23:58 2020

@author: ghiggi
"""
import datetime

from gpm_api.io.find import (
    find_daily_GPM_disk_filepaths,
    find_daily_GPM_PPS_filepaths,
    find_GPM_files,
)

##----------------------------------------------------------------------------.
base_dir = "/home/ghiggi/tmp"
username = "gionata.ghiggi@epfl.ch"
product = "2A-DPR"
start_hhmmss = "000000"
end_hhmmss = "020000"
version = 6

##----------------------------------------------------------------------------.
## Still BUGGY for the following case
# - Need to take only the previous day
start_hhmmss = "000000"
end_hhmmss = "001000"
# - Need to take only a single granule in current day
start_hhmmss = "020028"
end_hhmmss = "020030"
# - Need to take last granule
start_hhmmss = "235000"
end_hhmmss = "240000"

# start_hhmmss and end_hhmmss in datetime.time format
start_hhmmss = datetime.time.fromisoformat("00:00:00")
end_hhmmss = datetime.time.fromisoformat("23:59:00")
##----------------------------------------------------------------------------.
### Retrieve RS filepaths
date = datetime.date.fromisoformat("2014-08-09")
product_type = "RS"

(server_paths, disk_paths) = find_daily_GPM_PPS_filepaths(
    username=username,
    base_dir=base_dir,
    product=product,
    date=date,
    start_hhmmss=start_hhmmss,
    end_hhmmss=end_hhmmss,
    product_type=product_type,
    version=version,
    flag_first_date=True,
)
print(server_paths)
print(disk_paths)

find_daily_GPM_disk_filepaths(
    base_dir=base_dir,
    product=product,
    date=date,
    start_hhmmss=start_hhmmss,
    end_hhmmss=end_hhmmss,
    product_type=product_type,
    version=version,
    flag_first_date=True,
)
##----------------------------------------------------------------------------.
### Retrieve NRT filepaths
date = datetime.date.fromisoformat("2020-08-16")
product_type = "NRT"

(server_paths, disk_paths) = find_daily_GPM_PPS_filepaths(
    username=username,
    base_dir=base_dir,
    product=product,
    product_type=product_type,
    version=version,
    date=date,
    start_hhmmss=start_hhmmss,
    end_hhmmss=end_hhmmss,
    flag_first_date=True,
)
print(server_paths)
print(disk_paths)

find_daily_GPM_disk_filepaths(
    base_dir=base_dir,
    product=product,
    date=date,
    start_hhmmss=start_hhmmss,
    end_hhmmss=end_hhmmss,
    product_type=product_type,
    version=version,
    flag_first_date=True,
)

### find_GPM_files()
start_time = datetime.datetime.fromisoformat("2020-08-15 00:00:00")
end_time = datetime.datetime.fromisoformat("2020-08-17 00:00:00")
find_GPM_files(
    base_dir=base_dir,
    product=product,
    start_time=start_time,
    end_time=end_time,
    product_type=product_type,
    version=version,
)
