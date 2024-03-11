#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 00:36:59 2023

@author: ghiggi
"""
import datetime

import numpy as np

import gpm

###----------------------------------------------------------------------------.
start_time = datetime.datetime.strptime("2020-08-01 12:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-08-02 12:00:00", "%Y-%m-%d %H:%M:%S")
product = "2A-SSMIS-F16-CLIM"
product_type = "RS"
variable = "surfacePrecipitation"
version = 7

gpm.download(
    product=product,
    product_type=product_type,
    version=version,
    start_time=start_time,
    end_time=end_time,
    force_download=False,
    storage="ges_disc",
    n_threads=6,
    verbose=True,
    progress_bar=True,
    check_integrity=True,
)


ds = gpm.open_dataset(
    product=product,
    product_type=product_type,
    start_time=start_time,
    end_time=end_time,
    # Optional
    version=version,
    variables=variable,
    # decode_cf=True,
    chunks={},
    prefix_group=False,
)

ds = ds.compute()
ds1 = ds

gpm.check_valid_geolocation(ds)

ds1.gpm.get_slices_contiguous_scans()

ds = ds.isel(along_track=slice(0, 10000))

ds[variable].gpm.plot_map()
