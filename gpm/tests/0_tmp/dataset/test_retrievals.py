#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:38:55 2023

@author: ghiggi
"""
import gpm
import datetime

#### Define analysis time period
start_time = datetime.datetime.strptime("2016-03-09 10:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2016-03-09 11:00:00", "%Y-%m-%d %H:%M:%S")


product = "2A-DPR"
version = 7
product_type = "RS"

ds = gpm.open_dataset(
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

ds.gpm.available_retrievals()

ds.gpm.retrieve("clutterHeight")

ds["binClutterFreeBottomHeight"] = ds.gpm.retrieve("binClutterFreeBottomHeight")
ds["binRealSurfaceHeightKu"] = ds.gpm.retrieve("binRealSurfaceHeightKu")
ds["binRealSurfaceHeightKa"] = ds.gpm.retrieve("binRealSurfaceHeightKa")

ds["binClutterFreeBottomHeight"].gpm.plot_map(vmin=0, vmax=2000)
ds["binRealSurfaceHeightKu"].gpm.plot_map(vmin=0, vmax=2000)
ds["binRealSurfaceHeightKa"].gpm.plot_map(vmin=0, vmax=2000)
