#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:22:34 2022

@author: ghiggi
"""
import datetime 
import gpm_api
from gpm_api.io.products import GPM_products
from gpm_api.io.find import find_daily_GPM_PPS_filepaths
from gpm_api.io.filter import filter_GPM_query
from gpm_api.io.checks import (
    check_version,
    check_product,
    check_product_type,
    check_time,
    check_date,
    is_empty,
)


#-------------------------------------------------------------------------.
base_dir = '/home/ghiggi/tmp'
username = "gionata.ghiggi@epfl.ch"
version = 7
product_type = 'RS'
products  = GPM_products(product_type)

date = datetime.date.fromisoformat("2020-08-17")

start_time = datetime.datetime.strptime("2020-08-17 00:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2020-08-17 04:00:00", '%Y-%m-%d %H:%M:%S')

product = "2A-DPR"
flag_first_date=False
verbose = True
force_download = False
#-------------------------------------------------------------------------.
date = check_date(date)
check_product_type(product_type = product_type)
check_product(product = product, product_type = product_type)
start_hhmmss = datetime.datetime.strftime(start_time,"%H%M%S")
end_hhmmss = datetime.datetime.strftime(end_time,"%H%M%S")
#-------------------------------------------------------------------------.
## Retrieve the list of files available on NASA PPS server
(server_paths, disk_paths) = find_daily_GPM_PPS_filepaths(username = username,
                                                          base_dir = base_dir, 
                                                          product = product, 
                                                          product_type = product_type,
                                                          version = version,
                                                          date = date, 
                                                          start_hhmmss = start_hhmmss, 
                                                          end_hhmmss = end_hhmmss,
                                                          flag_first_date = flag_first_date,
                                                          verbose = verbose)
#-------------------------------------------------------------------------.
## If no file to retrieve on NASA PPS, return None
if is_empty(server_paths):
    # print("No data found on PPS on Date", Date, "for product", product)
    return None
#-------------------------------------------------------------------------.
## If force_download is False, select only data not present on disk 
(server_paths, disk_paths) = filter_GPM_query(disk_paths = disk_paths, 
                                              server_paths = server_paths,  
                                              force_download = force_download)