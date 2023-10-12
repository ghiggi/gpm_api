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
end_time = datetime.datetime.strptime("2020-08-10 12:00:00", "%Y-%m-%d %H:%M:%S")
product = "2A-SSMIS-F16"
product_type = "RS"
variable = "surfacePrecipitation"
version = None

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

ds1.gpm_api.get_contiguous_scan_slices()

ds = ds.isel(along_track=slice(0, 10000))

ds[variable].gpm_api.plot_map()

# -----------------------------------------------------------------------------.
### TODO: add checks when missing coordinates
# 3. Implement missing coordinates checks
# 4. Test get_contiguous_scan_slices when missing coordinates
# 4. Test plotting for missing coordinates ---> mask !
#   --> No invalid coordinates
#   --> Mask for plotting
#   --> Do not mask for crop

# --> When mask --- when decoding cf ... or custom afterwards !
# --> Mask variables only?
# --> xarray does not allow masked coordinates
gpm_api.check_valid_geolocation()

# Coordinate -9999

idx_nan = np.where(ds["lat"].data == -9999)

ds = ds.where(ds["lat"] != -9999, np.nan)
ds = ds.where(ds["lon"] != -9999, np.nan)

xr.where(ds.lon == -9999)

x, y = np.where(ds.lon.values == -9999.0)
x_unique, counts = np.unique(x, return_counts=True)
ds.isel(along_track=slice(x_unique[0], x_unique[0] + 2)).lon
len(x_unique) / (ds.lon.shape[0] * ds.lon.shape[1]) * 100
