#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:54:23 2020

@author: ghiggi
"""
import datetime

import numpy as np

import gpm_api
from gpm_api.io.products import GPM_NRT_products, GPM_RS_products

##----------------------------------------------------------------------------.
### Donwload data
base_dir = "/home/ghiggi/tmp"
username = "gionata.ghiggi@epfl.ch"


##-----------------------------------------------------------------------------.
## Retrieve RS data
version = 7
product_type = "RS"
products = GPM_RS_products()  # GPM_products(product_type)

# Only GPM
start_time = datetime.datetime.strptime("2020-08-09 15:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-08-09 17:00:00", "%Y-%m-%d %H:%M:%S")

# Both GPM-TRMM
start_time = datetime.datetime.strptime("2014-08-09 00:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2014-08-09 03:00:00", "%Y-%m-%d %H:%M:%S")

for product in products:
    print("Product:", product)
    gpm_api.download(
        base_dir=base_dir,
        username=username,
        product=product,
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
        verbose=True,
    )

##-----------------------------------------------------------------------------.
## Retrieve NRT data
version = 6
product_type = "NRT"
products = GPM_NRT_products()  # GPM_products(product_type)

date = datetime.date.fromisoformat("2020-08-17")

start_time = datetime.datetime.strptime("2020-08-17 00:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-08-17 04:00:00", "%Y-%m-%d %H:%M:%S")

for product in products:
    print("Product:", product)
    gpm_api.download(
        base_dir=base_dir,
        username=username,
        product=product,
        product_type=product_type,
        start_time=start_time,
        end_time=end_time,
    )

##-----------------------------------------------------------------------------.
## Download data for a specific day
date = np.datetime64("2017-01-01").astype(datetime.datetime)
version = 6
product_type = "RS"
product = "2A-Ku"
base_dir = "/home/ghiggi/tmp"
start_hhmmss = datetime.datetime.strptime("01:00:00", "%H:%M:%S")
end_hhmmss = datetime.datetime.strptime("03:00:00", "%H:%M:%S")

gpm_api.download(
    base_dir=base_dir,
    username=username,
    version=version,
    product=product,
    product_type=product_type,
    date=date,
    start_hhmmss=start_hhmmss,
    end_hhmmss=end_hhmmss,
)
# -----------------------------------------------------------------------------.
