#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:38:55 2023

@author: ghiggi
"""
import gpm_api
import datetime

#### Define analysis time period
start_time = datetime.datetime.strptime("2016-03-09 10:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2016-03-09 11:00:00", "%Y-%m-%d %H:%M:%S")


product = "2A-DPR"
version = 7
product_type = "RS"

ds = gpm_api.open_dataset(
    product=product,
    start_time=start_time,
    end_time=end_time,
    # Optional
    version=version,
    product_type=product_type,
    chunks={},
    decode_cf=True,
    prefix_group=False,
)

ds.gpm_api.available_retrievals()

ds.gpm_api.retrieve("clutterHeight")

ds["binClutterFreeBottomHeight"] = ds.gpm_api.retrieve("binClutterFreeBottomHeight")
ds["binRealSurfaceHeightKu"] = ds.gpm_api.retrieve("binRealSurfaceHeightKu")
ds["binRealSurfaceHeightKa"] = ds.gpm_api.retrieve("binRealSurfaceHeightKa")

ds["binClutterFreeBottomHeight"].gpm_api.plot_map(vmin=0, vmax=2000)
ds["binRealSurfaceHeightKu"].gpm_api.plot_map(vmin=0, vmax=2000)
ds["binRealSurfaceHeightKa"].gpm_api.plot_map(vmin=0, vmax=2000)
