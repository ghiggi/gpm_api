#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 00:19:05 2023

@author: ghiggi
"""
import datetime

import gpm_api

#### Define analysis time period
start_time = datetime.datetime.strptime("2020-08-01 12:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-08-02 12:00:00", "%Y-%m-%d %H:%M:%S")
# end_time = datetime.datetime.strptime("2020-08-10 12:00:00", "%Y-%m-%d %H:%M:%S")

product = "2A-MHS-METOPB"
product = "2A-SSMIS-F17"

product_type = "RS"
variable = "surfacePrecipitation"

ds = gpm_api.open_dataset(
    product=product,
    start_time=start_time,
    end_time=end_time,
    # Optional
    variables=variable,
    product_type=product_type,
    chunks={},
    decode_cf=True,
    prefix_group=False,
)

# gpm_api.check_regular_timesteps(ds) # BAD INDICATOR !
# gpm_api.check_contiguous_scans(ds)
# ds.gpm_api.get_contiguous_scan_slices()

ds = ds.compute()
ds1 = ds

ds = ds1
# ds1.gpm_api.get_contiguous_scan_slices()

ds = ds.isel(along_track=slice(0, 20000))

ds = ds.isel(along_track=slice(0, 5000))  # ENOUGH FOR DEVELOPMENT

ds[variable]

ds.gpm_api.plot_map(variable=variable)

ds.gpm_api.plot_map(variable=variable, cmap="Spectral", add_swath_lines=False)

# -----------------------------------------------------------------------------.
# Custom plots
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from gpm_api.visualization.plot import plot_cartopy_background

dpi = 100
figsize = (12, 10)

# Polar projection
crs_proj = ccrs.Orthographic(180, -90)
fig, ax = plt.subplots(subplot_kw={"projection": crs_proj}, figsize=figsize, dpi=dpi)
plot_cartopy_background(ax)
p = ds.gpm_api.plot_map(variable=variable, ax=ax, add_swath_lines=True, cmap="Spectral")

# -----------------------------------------------------------------------------.
