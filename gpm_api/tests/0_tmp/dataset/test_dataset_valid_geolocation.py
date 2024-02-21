#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 00:36:59 2023

@author: ghiggi
"""
import datetime

import numpy as np

import gpm_api

###----------------------------------------------------------------------------.
start_time = datetime.datetime.strptime("2020-08-01 12:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-08-02 12:00:00", "%Y-%m-%d %H:%M:%S")
product = "2A-SSMIS-F16-CLIM"
product_type = "RS"
variable = "surfacePrecipitation"
version = 7

gpm_api.download(
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


ds = gpm_api.open_dataset(
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

gpm_api.check_valid_geolocation(ds)

ds1.gpm_api.get_slices_contiguous_scans()

ds = ds.isel(along_track=slice(0, 10000))

ds[variable].gpm_api.plot_map()
