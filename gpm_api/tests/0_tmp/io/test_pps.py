#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:59:07 2022

@author: ghiggi
"""
import datetime

from gpm_api.io.find import find_daily_filepaths, find_filepaths

product = "2A-DPR"
date = datetime.date(2020, 7, 5)  # OK

version = 7
start_time = None
end_time = None
product_type = "RS"
verbose = True
parallel = True

filepaths, available_version = find_daily_filepaths(
    storage="pps",
    product=product,
    product_type=product_type,
    date=date,
    version=version,
    start_time=start_time,
    end_time=end_time,
    verbose=verbose,
)


import time

import numpy as np

start_time = datetime.datetime(2020, 7, 1, 0, 2, 0)
end_time = datetime.datetime(
    2020, 7, 30, 0, 4, 0
)  # this time make raise the print error that should be removed ...

t_i = time.time()
filepaths = find_filepaths(
    storage="pps",
    product=product,
    product_type=product_type,
    version=version,
    start_time=start_time,
    end_time=end_time,
    verbose=verbose,
)
t_f = time.time()

t_elapsed = np.round(t_f - t_i, 2)
print(t_elapsed, "seconds")  # 10 seconds per month


#### Find imerg data
import datetime

from gpm_api.io.find import find_filepaths

start_time = datetime.datetime.strptime("2019-07-13 11:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2019-07-13 13:00:00", "%Y-%m-%d %H:%M:%S")
product = "IMERG-FR"  # 'IMERG-ER' 'IMERG-LR'
product_type = "RS"
version = 6
verbose = True
parallel = True
filepaths = find_filepaths(
    storage="pps",
    product=product,
    product_type=product_type,
    start_time=start_time,
    end_time=end_time,
    version=version,
    verbose=verbose,
    parallel=parallel,
)
len(filepaths)
