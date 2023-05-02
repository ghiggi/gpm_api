#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:42:49 2022

@author: ghiggi
"""
import datetime
import os

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xp
from dask.diagnostics import ProgressBar

import gpm_api
from gpm_api.io import download_GPM_data
from gpm_api.io_future.dataset import open_dataset
from gpm_api.utils.utils_cmap import get_colormap_setting

matplotlib.rcParams["axes.facecolor"] = [0.9, 0.9, 0.9]
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["legend.facecolor"] = "w"
matplotlib.rcParams["savefig.transparent"] = False

# Define  GPM settings
base_dir = "/home/ghiggi"
username = "gionata.ghiggi@epfl.ch"

#### Define analysis time period
start_time = datetime.datetime.strptime("2020-10-28 08:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-10-28 09:00:00", "%Y-%m-%d %H:%M:%S")

products = ["2A-DPR"]
version = 7
product_type = "RS"

#### Download products
# for product in products:
#     print(product)
#     download_GPM_data(base_dir=base_dir,
#                       username=username,
#                       product=product,
#                       product_type=product_type,
#                       version = version,
#                       start_time=start_time,
#                       end_time=end_time,
#                       force_download=False,
#                       transfer_tool="curl",
#                       progress_bar=True,
#                       verbose = True,
#                       n_threads=1)

##### Read data
product = "2A-DPR"
variables = ["precipRateNearSurface"]

ds = open_dataset(
    base_dir=base_dir,
    product=product,
    start_time=start_time,
    end_time=end_time,
    # Optional
    variables=variables,
    version=version,
    product_type=product_type,
    chunks="auto",
    decode_cf=True,
    prefix_group=False,
)


plot_kwargs, cbar_kwargs, ticklabels = get_colormap_setting("pysteps_mm/hr")

bbox = [-110, -70, 18, 32]
bbox_extent = [-100, -85, 18, 32]

# Crop dataset
ds_subset = ds.gpm_api.crop(bbox)

# Retrieve DataArray
da_subset = ds_subset[variable]
da_subset.data[da_subset.data > 159] = 159

# Define figure settings
dpi = 100
figsize = (12, 10)
crs_proj = ccrs.PlateCarree()

#### Check with cartopy
# Create figure
fig, ax = plt.subplots(subplot_kw={"projection": crs_proj}, figsize=figsize, dpi=dpi)

# - Add coastlines
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, facecolor=[0.9, 0.9, 0.9])
ax.add_feature(cartopy.feature.OCEAN, alpha=0.6)
ax.add_feature(cartopy.feature.STATES)


# Add variable field with xarray
# --> BUG TO REPORT ... if last value not present...colorbar is moved...
p = da_subset.plot.pcolormesh(
    x="lon",
    y="lat",
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs=cbar_kwargs,
    **plot_kwargs,
)

cbar = p.colorbar
_ = cbar.ax.set_yticklabels(ticklabels)

ax.set_extent(bbox_extent)


#### Check without cartopy
# Create figure
fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

# Add variable field with xarray
# --> BUG TO REPORT ... if last value not present...colorbar is moved...
p = da_subset.plot.pcolormesh(
    x="lon",
    y="lat",
    ax=ax,
    cbar_kwargs=cbar_kwargs,
    **plot_kwargs,
)

cbar = p.colorbar
_ = cbar.ax.set_yticklabels(ticklabels)


import matplotlib as mpl

# -----------------------------------------------------------------------------.
#### Reproducible example
import matplotlib.colors
import numpy as np
import xarray as xr

# Define DatArray
arr = np.array(
    [
        [0, 10, 15, 20],
        [np.nan, 40, 50, 100],
        [150, 158, 160, 161],
    ]
)
lon = np.arange(arr.shape[1])
lat = np.arange(arr.shape[0])[::-1]
lons, lats = np.meshgrid(lon, lat)
da = xr.DataArray(
    arr,
    dims=["y", "x"],
    coords={
        "lon": (("y", "x"), lons),
        "lat": (("y", "x"), lats),
    },
)
da

# Define colormap
color_list = ["#9c7e94", "#640064", "#009696", "#C8FF00", "#FF7D00"]
levels = [0.05, 1, 10, 20, 150, 160]
cmap = mpl.colors.LinearSegmentedColormap.from_list("cmap", color_list, len(levels) - 1)
norm = mpl.colors.BoundaryNorm(levels, cmap.N)
vmin = None  # cartopy and matplotlib complain if not None when norm is provided !
vmax = None  # cartopy and matplotlib complain if not None when norm is provided !
cmap.set_over("darkred")  # color for above 160
cmap.set_under("none")  # color for below 0.05
cmap.set_bad("gray", 0.2)  # color for nan

# Colorbar settings
ticks = levels
cbar_kwargs = {
    "extend": "max",
}

# Correct plot
p = da.plot.pcolormesh(x="lon", y="lat", cmap=cmap, norm=norm, cbar_kwargs=cbar_kwargs)
plt.show()

# Remove values larger than the max level
da1 = da.copy()
da1.data[da1.data >= norm.vmax] = norm.vmax - 1

# With matplotlib.pcolormesh [OK]
p = plt.pcolormesh(da1["lon"].data, da1["lat"], da1.data, cmap=cmap, norm=norm)
plt.colorbar(p, **cbar_kwargs)
plt.show()

# With matplotlib.imshow [OK]
p = plt.imshow(da1.data, cmap=cmap, norm=norm)
plt.colorbar(p, **cbar_kwargs)
plt.show()


# With xarray.pcolormesh [BUG]
# --> The colorbar shift !!!
da1.plot.pcolormesh(x="lon", y="lat", cmap=cmap, norm=norm, cbar_kwargs=cbar_kwargs)
plt.show()

# With xarray.imshow [BUG]
# --> The colorbar shift !!!
p = da1.plot.imshow(cmap=cmap, norm=norm, cbar_kwargs=cbar_kwargs)


da1.plot.imshow(cmap=cmap, levels=levels)


### Xarray BUG ####


p.get_cmap()  # It alters the colormap !

cmap(np.nan)  # bad

cmap(-np.inf)  # under
cmap(0)  # under
cmap(0.06)  # under
cmap(0.0624)  # under
cmap(0.0625)  # first bin  # why already here ... if min is 0.08
cmap(0.08)  # first bin

# cmap(80)
# cmap(100)
# cmap(160)
# cmap(np.inf) # over
