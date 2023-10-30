#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 15:46:47 2022

@author: ghiggi
"""
import datetime
from gpm_api.io.find import find_daily_filepaths, find_filepaths

product = "2A-DPR"
date = datetime.date(2020, 7, 5)  # OK
# date = datetime.date(2020, 7, 3) # NO
version = 7
start_time = None
end_time = None
product_type = "RS"
verbose = True

filepaths, _ = find_daily_filepaths(
    storage="local",
    product=product,
    product_type=product_type,
    date=date,
    version=version,
    start_time=start_time,
    end_time=end_time,
    verbose=verbose,
)


start_time = datetime.datetime(2020, 7, 5, 0, 2, 0)
end_time = datetime.datetime(2020, 7, 5, 0, 3, 0)

filepaths = find_filepaths(
    storage="local",
    product=product,
    product_type=product_type,
    version=version,
    start_time=start_time,
    end_time=end_time,
    verbose=verbose,
)
