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
# - Backward scan
start_time = datetime.datetime.strptime("2020-08-01 12:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-08-01 15:00:00", "%Y-%m-%d %H:%M:%S")

# Forward scan
start_time = datetime.datetime.strptime("2016-06-15 00:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2016-06-16 00:00:00", "%Y-%m-%d %H:%M:%S")
# --> TODO: compare to this
# https://disc.gsfc.nasa.gov/datasets/GPM_1BGMI_07/summary?keywords=GPM%20GMI%201

product_type = "RS"

# product = "1A-GMI"
# pmw_variable = "satAzimuthAngle"

product = "1B-GMI"
pmw_variable = "Tb"  # sunGlintAngle, # solarAzimuthAngle, # solarZenAngle, # satAzimuthAngle

# product = "1C-GMI"
# pmw_variable = "Tc"  # Quality

# product = "2A-GMI"
# pmw_variable = "surfacePrecipitation"

version = 7
# product = "1A-GMI"
# product = "1B-GMI"
# product = "1C-GMI"
# product = "2A-GMI"
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

# gpm_api.available_scan_modes(product='1A-GMI', version=7)
# gpm_api.available_scan_modes(product='1B-GMI', version=7)
# gpm_api.available_scan_modes(product='1C-GMI', version=7)
# gpm_api.available_scan_modes(product='2A-GMI', version=7)

ds_gmi = gpm_api.open_dataset(
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


ds_gmi["SCorientation"].compute()  # 180 --> backward scan, 0 --> forward scan

da_gmi = ds_gmi[pmw_variable].isel(channel=0).compute()

da_gmi.gpm_api.plot_map(cmap="Spectral")

da_gmi.isel(along_track=slice(0, 200)).gpm_api.plot_map(cmap="Spectral")

# Investigate PMW GMI direction
da_gmi.isel(along_track=slice(0, 10), cross_track=slice(0, 10)).gpm_api.plot_map_mesh()
da_gmi.isel(along_track=slice(0, 10)).gpm_api.plot_map_mesh()
da_gmi.isel(along_track=slice(0, 20)).gpm_api.plot_map_mesh()
da_gmi.isel(along_track=slice(0, 21)).gpm_api.plot_map_mesh()
da_gmi.isel(along_track=slice(0, 40)).gpm_api.plot_map_mesh()
da_gmi.isel(along_track=slice(0, 200)).gpm_api.plot_map_mesh()


#### Define DPR
product = "2A-DPR"
product_type = "RS"
dpr_variable = "precipRateNearSurface"

# gpm_api.download(
#     username="gionata.ghiggi@epfl.ch",
#     base_dir=base_dir,
#     product=product,
#     product_type = product_type,
#     start_time=start_time,
#     end_time=end_time,
#     # Optional
#     version=7,
#     )


ds_dpr = gpm_api.open_dataset(
    base_dir=base_dir,
    product=product,
    product_type=product_type,
    start_time=start_time,
    end_time=end_time,
    # Optional
    version=7,
    variables=dpr_variable,
    chunks="auto",
    prefix_group=False,
)

ds_dpr = ds_dpr.compute()

# Investigate DPR direction
ds_dpr[dpr_variable].isel(along_track=slice(0, 10)).gpm_api.plot_map_mesh()
ds_dpr[dpr_variable].isel(along_track=slice(0, 20)).gpm_api.plot_map_mesh()
ds_dpr[dpr_variable].isel(along_track=slice(0, 21)).gpm_api.plot_map_mesh()
ds_dpr[dpr_variable].isel(along_track=slice(0, 40)).gpm_api.plot_map_mesh()
ds_dpr[dpr_variable].isel(along_track=slice(0, 400)).gpm_api.plot_map_mesh()
ds_dpr[dpr_variable].isel(along_track=slice(0, 400)).gpm_api.plot_map_mesh()

# Plot DPR and GMI for same period
da_gmi = ds_gmi[pmw_variable].isel(along_track=slice(0, 200))
da_dpr = ds_dpr[dpr_variable].gpm_api.subset_by_time(
    min(da_gmi["time"].data), max(da_gmi["time"].data)
)
p = da_gmi.gpm_api.plot_map_mesh()
p = da_dpr.gpm_api.plot_map_mesh(ax=p.axes, edgecolors="r")

p = da_dpr.gpm_api.plot_map()

p = da_dpr.gpm_api.plot_map()
p = da_gmi.gpm_api.plot_map(ax=p.axes)

p = da_gmi.gpm_api.plot_map()
p = da_dpr.gpm_api.plot_map(ax=p.axes)
