#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 00:49:33 2023

@author: ghiggi
"""
import datetime

import gpm_api

#### Define analysis time period
start_time = datetime.datetime.strptime("2020-08-01 12:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-08-10 12:00:00", "%Y-%m-%d %H:%M:%S")

product_type = "RS"

product = "1B-GMI"
variable = "Tb"

product = "1C-GMI"
variable = "Tc"

ds = gpm_api.open_dataset(
    product=product,
    start_time=start_time,
    end_time=end_time,
    # Optional
    variables=variable,
    # scan_mode="S1",
    product_type=product_type,
    chunks={},
    prefix_group=False,
)

ds1 = ds
ds1 = ds1.compute()


# -----------------------------------------------------------------------------.
ds = ds1

# ds = ds.isel(along_track=slice(0, 1000))

# ds = ds.isel(along_track=slice(0, 5000))
# -----------------------------------------------------------------------------.


da = ds[variable].isel(pmw_frequency=8)
print(da.attrs["LongName"])

da.gpm_api.plot_map()
da.gpm_api.plot_map(add_swath_lines=False)
da.gpm_api.plot_map(add_colorbar=False)
da.gpm_api.plot_map(cmap="Spectral")

da.gpm_api.plot_map(cmap="Spectral", vmin=160, vmax=300)

from gpm_api.visualization.orbit import plot_orbit_map

plot_orbit_map(da)

# -----------------------------------------------------------------------------.
# Custom plots
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import gpm_api
from gpm_api.visualization.plot import plot_cartopy_background

dpi = 100
figsize = (12, 10)
crs_proj = ccrs.Robinson()
fig, ax = plt.subplots(subplot_kw={"projection": crs_proj}, figsize=figsize, dpi=dpi)
plot_cartopy_background(ax)
da.gpm_api.plot_map(ax=ax, cmap="Spectral")

# Polar projection
crs_proj = ccrs.Orthographic(180, -90)
fig, ax = plt.subplots(subplot_kw={"projection": crs_proj}, figsize=figsize, dpi=dpi)
plot_cartopy_background(ax)
p = da.gpm_api.plot_map(ax=ax, cmap="Spectral")
# p.axes.get_extent()

# -----------------------------------------------------------------------------.
