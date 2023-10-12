#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 19:21:30 2023

@author: ghiggi
"""
import datetime

import numpy as np

import gpm_api

#### Define analysis time period
start_time = datetime.datetime.strptime("2020-08-01 12:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-08-10 12:00:00", "%Y-%m-%d %H:%M:%S")

product = "2A-GMI"
product_type = "RS"
variable = "surfacePrecipitation"
version = None

ds = gpm_api.open_dataset(
    product=product,
    start_time=start_time,
    end_time=end_time,
    # Optional
    version=version,
    variables=variable,
    product_type=product_type,
    chunks={},
    prefix_group=False,
)

ds = ds.isel(along_track=slice(0, 30000))
ds = ds.compute()

ds[variable]

ds.gpm_api.plot_map(variable=variable)

# -----------------------------------------------------------------------------.
# Custom plots
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from gpm_api.visualization.plot import plot_cartopy_background

dpi = 100
figsize = (12, 10)
crs_proj = ccrs.Robinson()
fig, ax = plt.subplots(subplot_kw={"projection": crs_proj}, figsize=figsize, dpi=dpi)
plot_cartopy_background(ax)
ds.gpm_api.plot_map(variable=variable, ax=ax)

# Polar projection
crs_proj = ccrs.Orthographic(180, -90)
fig, ax = plt.subplots(subplot_kw={"projection": crs_proj}, figsize=figsize, dpi=dpi)
plot_cartopy_background(ax)
p = ds.gpm_api.plot_map(variable=variable, ax=ax)
p.axes.get_extent()

# -----------------------------------------------------------------------------.
### DEV CODE
# - SIMPLIFICATION: IT DISCARD CELLS CROSSING THE ANTIMERIDIAN
# - BUG IN CARTOPY: DOES NOT ALLOW BAD COLOR TO BE NOT TRANSPARENT !!!
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from gpm_api.utils.utils_cmap import get_colormap_setting
from gpm_api.visualization.plot import get_antimeridian_mask

plot_kwargs, cbar_kwargs, ticklabels = get_colormap_setting("pysteps_mm/hr")

lons = ds["lon"].data
lats = ds["lat"].data
arr = ds["surfacePrecipitation"].data

mask = get_antimeridian_mask(lons, buffer=True)
if np.any(mask):
    arr = np.ma.masked_where(mask, arr)
    # Sanitize cmap bad color to avoid cartopy bug
    if "cmap" in plot_kwargs:
        cmap = plot_kwargs["cmap"]
        bad = cmap.get_bad()
        bad[3] = 0  # enforce to 0 (transparent)
        cmap.set_bad(bad)
        plot_kwargs["cmap"] = cmap

# - with centroids
proj = ccrs.PlateCarree()
ax = plt.axes(projection=proj)
p = ax.pcolormesh(lons, lats, arr, transform=ccrs.PlateCarree(), **plot_kwargs)
ax.coastlines()

# -----------------------------------------------------------------------------.
