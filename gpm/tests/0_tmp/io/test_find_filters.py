#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:22:34 2022

@author: ghiggi
"""
import datetime
import gpm
from gpm.io.checks import (
    check_date,
    check_product,
    check_product_type,
)
from gpm.io.filter import (
    filter_by_product,
    filter_by_time,
)
from gpm.io.pps import find_pps_daily_filepaths

# -------------------------------------------------------------------------.
version = 7
product_type = "RS"
products = gpm.available_products(product_types=product_type)

date = datetime.date.fromisoformat("2020-08-17")
start_time = datetime.datetime.strptime("2020-08-17 00:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-08-17 04:00:00", "%Y-%m-%d %H:%M:%S")

product = "2A-DPR"
flag_first_date = False
verbose = True
force_download = False

# -------------------------------------------------------------------------.
date = check_date(date)
product_type = check_product_type(product_type=product_type)
product = check_product(product=product, product_type=product_type)


# -------------------------------------------------------------------------.
## Retrieve the list of files available on NASA PPS server
(pps_filepaths, local_filepaths) = find_pps_daily_filepaths(
    product=product,
    product_type=product_type,
    version=version,
    date=date,
    start_time=start_time,
    end_time=end_time,
    flag_first_date=flag_first_date,
    verbose=verbose,
)

# -------------------------------------------------------------------------.
filter_by_time(pps_filepaths, date, start_time, end_time)
filter_by_product(pps_filepaths, product=product)
