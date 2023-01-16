#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 19:45:26 2023

@author: ghiggi
"""
import os
import datetime
import numpy as np
import gpm_api
from dask.diagnostics import ProgressBar

base_dir = "/home/ghiggi"

#### Define analysis time period
# - Forward
start_time = datetime.datetime.strptime("2005-08-28 20:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2005-08-28 23:00:00", "%Y-%m-%d %H:%M:%S")
version = 5

# Backward 
start_time = datetime.datetime.strptime("2014-07-01 06:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2014-07-01 09:00:00", "%Y-%m-%d %H:%M:%S")
version = 7

product_type = "RS"

product = "1B-TMI"
pmw_variable = (
    "Tb"  # sunGlintAngle, # solarAzimuthAngle, # solarZenAngle, # satAzimuthAngle
)

# gpm_api.download(
#     username="gionata.ghiggi@epfl.ch",
#     base_dir=base_dir,
#     product=product,
#     product_type = product_type,
#     start_time=start_time,
#     end_time=end_time,
#     # Optional
#     version=version,
#     )

# gpm_api.available_scan_modes(product='1A-tmi', version=7)
# gpm_api.available_scan_modes(product='1B-tmi', version=7)
# gpm_api.available_scan_modes(product='1C-tmi', version=7)
# gpm_api.available_scan_modes(product='2A-tmi', version=7)

ds_tmi = gpm_api.open_dataset(
    base_dir=base_dir,
    product=product,
    product_type=product_type,
    start_time=start_time,
    end_time=end_time,
    # Optional
    version=version,
    # variables=pmw_variable,
    chunks="auto",
    prefix_group=False,
)

ds_tmi["SCorientation"].compute() # 180 --> backward scan, 0 --> forward scan

da_tmi = ds_tmi[pmw_variable].isel(channel=0)
da_tmi = da_tmi.compute()

# Investigate PMW tmi direction
da_tmi.gpm_api.plot_map(cmap="Spectral")

da_tmi.isel(along_track=slice(920, 1000)).gpm_api.plot_map(cmap="Spectral")
da_tmi.isel(along_track=slice(0, 110)).gpm_api.plot_map(cmap="Spectral")
da_tmi.isel(along_track=slice(0, 5)).gpm_api.plot_map(cmap="Spectral")
da_tmi.isel(along_track=slice(0, 19)).gpm_api.plot_map(cmap="Spectral")
                                                        
da_tmi.isel(along_track=slice(0, 10), cross_track=slice(0, 10)).gpm_api.plot_map_mesh()
da_tmi.isel(along_track=slice(0, 10)).gpm_api.plot_map_mesh()
da_tmi.isel(along_track=slice(0, 20)).gpm_api.plot_map_mesh()
da_tmi.isel(along_track=slice(0, 21)).gpm_api.plot_map_mesh()
da_tmi.isel(along_track=slice(0, 40)).gpm_api.plot_map_mesh()
da_tmi.isel(along_track=slice(0, 200)).gpm_api.plot_map_mesh()


#### Define pr
product = "2A-PR"
product_type = "RS"
pr_variable = "precipRateNearSurface"

gpm_api.download(
    username="gionata.ghiggi@epfl.ch",
    base_dir=base_dir,
    product=product,
    product_type = product_type,
    start_time=start_time,
    end_time=end_time,
    # Optional
    version=version,
    )


ds_pr = gpm_api.open_dataset(
    base_dir=base_dir,
    product=product,
    product_type=product_type,
    start_time=start_time,
    end_time=end_time,
    # Optional
    version=7,
    variables=pr_variable,
    chunks="auto",
    prefix_group=False,
)

ds_pr = ds_pr.compute()

# Investigate pr direction
ds_pr[pr_variable].isel(along_track=slice(0, 10)).gpm_api.plot_map_mesh()
ds_pr[pr_variable].isel(along_track=slice(0, 20)).gpm_api.plot_map_mesh()
ds_pr[pr_variable].isel(along_track=slice(0, 21)).gpm_api.plot_map_mesh()
ds_pr[pr_variable].isel(along_track=slice(0, 40)).gpm_api.plot_map_mesh()
ds_pr[pr_variable].isel(along_track=slice(0, 400)).gpm_api.plot_map_mesh()
ds_pr[pr_variable].isel(along_track=slice(0, 400)).gpm_api.plot_map_mesh()

# Plot pr and tmi for same period
da_tmi = ds_tmi[pmw_variable].isel(along_track=slice(0, 200))
da_pr = ds_pr[pr_variable].gpm_api.subset_by_time(
    min(da_tmi["time"].data), max(da_tmi["time"].data)
)
p = da_tmi.gpm_api.plot_map_mesh()
p = da_pr.gpm_api.plot_map_mesh(ax=p.axes, edgecolors="r")

p = da_pr.gpm_api.plot_map()

p = da_pr.gpm_api.plot_map()
p = da_tmi.gpm_api.plot_map(ax=p.axes)

p = da_tmi.gpm_api.plot_map()
p = da_pr.gpm_api.plot_map(ax=p.axes)
