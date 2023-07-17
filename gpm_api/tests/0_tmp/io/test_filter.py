#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:14:43 2022

@author: ghiggi
"""
from gpm_api.io.filter import (
    _filter_filepath,
    filter_filepaths,
)

filepaths = [
    "/home/ghiggi/GPM/RS/V07/RADAR/2A-DPR/2020/07/05/2A.GPM.DPR.V9-20211125.20200705-S000234-E013507.036081.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/RADAR/2A-DPR/2020/07/05/2A.GPM.DPR.V9-20211125.20200705-S000234-E013507.036081.V07A.HDF5",
]
filepath = filepaths[0]


product = "2A-DPR"
version = 7
start_time = None
end_time = None
product_type = "RS"

filter_filepaths(
    filepaths, product=product, product_type=product_type, start_time=None, end_time=None
)

_filter_filepath(filepath, product=None, version=5, start_time=None, end_time=None)


# Test imerg filtering
# - First should be discarded
# - Second should be maintained
import datetime

start_time = datetime.datetime.strptime("2019-07-12 11:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2019-07-12 13:30:00", "%Y-%m-%d %H:%M:%S")
product_type = "RS"
version = 6
filepaths = [
    "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmallversions/V06/2019/07/12/imerg/3B-HHR.MS.MRG.3IMERG.20190712-S100000-E102959.0600.V06B.HDF5",
    "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmallversions/V06/2019/07/12/imerg/3B-HHR.MS.MRG.3IMERG.20190712-S130000-E132959.0780.V06B.HDF5",
]

filter_filepaths(
    filepaths, product=product, product_type=product_type, start_time=start_time, end_time=end_time
)
