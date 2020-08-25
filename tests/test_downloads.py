#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:54:23 2020

@author: ghiggi
"""
import os
import datetime
os.chdir('/home/ghiggi/gpm_api') # change to the 'scripts_GPM.py' directory
### GPM Scripts ####
from gpm_api.io import download_GPM_data, download_daily_GPM_data
from gpm_api.io import GPM_RS_products, GPM_NRT_products, GPM_products
##----------------------------------------------------------------------------.
### Donwload data 
base_DIR = '/home/ghiggi/tmp'
username = "gionata.ghiggi@epfl.ch"


##-----------------------------------------------------------------------------.
## Retrieve RS data
GPM_version   = 6
product_type = 'RS'
products  = GPM_RS_products() # GPM_products(product_type)
# Only GPM 
start_time = datetime.datetime.strptime("2020-08-09 15:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2020-08-09 17:00:00", '%Y-%m-%d %H:%M:%S')

# Both GPM-TRMM
start_time = datetime.datetime.strptime("2014-08-09 00:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2014-08-09 03:00:00", '%Y-%m-%d %H:%M:%S')

for product in products:
    print("Product:", product)
    download_GPM_data(base_DIR = base_DIR, 
                      username = username,
                      product = product, 
                      product_type = product_type,
                      GPM_version = GPM_version,
                      start_time = start_time,
                      end_time = end_time)
##-----------------------------------------------------------------------------.
## Retrieve NRT data 
GPM_version = 6
product_type = 'NRT'
products  = GPM_NRT_products()  # GPM_products(product_type)

Date = datetime.date.fromisoformat("2020-08-17")

start_time = datetime.datetime.strptime("2020-08-17 00:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2020-08-17 04:00:00", '%Y-%m-%d %H:%M:%S')

for product in products:
    print("Product:", product)
    download_GPM_data(base_DIR = base_DIR, 
                      username = username,
                      product = product, 
                      product_type = product_type,
                      start_time = start_time,
                      end_time = end_time)
    
##-----------------------------------------------------------------------------.
## Download data for a specific day
Date = np.datetime64('2017-01-01').astype(datetime.datetime)
GPM_version = 6
product_type = 'RS'
product = '2A-Ku'
base_DIR = '/home/ghiggi/tmp'
start_HHMMSS = datetime.datetime.strptime("01:00:00","%H:%M:%S")
end_HHMMSS =  datetime.datetime.strptime("03:00:00","%H:%M:%S")

download_daily_GPM_data(base_DIR=base_DIR, 
                        username = username,
                        GPM_version = GPM_version,
                        product=product,
                        product_type = product_type,
                        Date=Date, 
                        start_HHMMSS=start_HHMMSS,
                        end_HHMMSS=end_HHMMSS)
#-----------------------------------------------------------------------------.







