#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:19:45 2022

@author: ghiggi
"""
import os
import gpm_api
import cartopy
import datetime
import matplotlib
import numpy as np
import xarray as xp
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
from gpm_api.io import download_GPM_data
from gpm_api.io_future.dataset import open_dataset

####--------------------------------------------------------------------------.
#### Define GPM settings
base_dir = "/home/ghiggi"
username = "gionata.ghiggi@epfl.ch"

####--------------------------------------------------------------------------.
#### Define analysis time period
start_time = datetime.datetime.strptime("2016-03-09 10:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2016-03-09 11:00:00", "%Y-%m-%d %H:%M:%S")
bbox = [-100, -85, 18, 32]

products = [
    "2A-DPR",
    "2B-GPM-CORRA",
    "2B-GPM-CSH",
    "2A-GPM-SLH",
    # '2A-ENV-DPR', '1C-GMI'
]

version = 7
product_type = "RS"

####--------------------------------------------------------------------------.
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


####--------------------------------------------------------------------------.
#### Read datasets
groups = None
variables = None
scan_mode = None
decode_cf = False
chunks = "auto"
prefix_group = False

product_var_dict = {
    "2A-DPR": ["precipRateNearSurface"],
    "2B-GPM-CORRA": ["nearSurfPrecipTotRate"],
}

dict_product = {}
# product, variables = list(product_var_dict.items())[0]
# product, variables = list(product_var_dict.items())[2]
for product, variables in product_var_dict.items():
    ds = open_dataset(
        base_dir=base_dir,
        product=product,
        start_time=start_time,
        end_time=end_time,
        # Optional
        variables=variables,
        groups=groups,
        scan_mode=scan_mode,
        version=version,
        product_type=product_type,
        chunks="auto",
        decode_cf=True,
        prefix_group=False,
    )
    ds = ds.gpm_api.crop([bbox[0] - 5, bbox[1] + 5, bbox[2] - 5, bbox[3] + 5])

    dict_product[product] = ds

####--------------------------------------------------------------------------.
#### Plot Datasets
ds_dpr = dict_product["2A-DPR"]
ds_corra = dict_product["2B-GPM-CORRA"]

# ds_corra.swap_dims(along_track="gpm_id").sel(gpm_id = ds_dpr['gpm_id']).swap_dims(along_track="gpm_id")

# ds_corra["lon"].data
# ds_dpr["lon"].data

# ds_diff = ds_dpr["lon"] - ds_corra["lon"]
# ds_corra['nearSurfPrecipTotRate'].gpm_api.plot()
# ds_dpr['precipRateNearSurface'].gpm_api.plot()
# TODO:
# ds.gpm_api.plot( draw_swath_lines, draw_x_axis, draw_y_axis

dpi = 300
figsize = (7, 2.8)
crs_proj = ccrs.PlateCarree()
da = ds["nearSurfPrecipTotRate"]


da.gpm_api.plot()

# Create figure
fig, ax = plt.subplots(subplot_kw={"projection": crs_proj}, figsize=figsize, dpi=dpi)

ax.stock_img()
da.plot.pcolormesh(x="lon", y="lat", ax=ax)
ax.set_extent(bbox)

da.gpm_api.plot_swath_lines(ax, color="white", linestyle="--", linewidth=0.2)

# da['lon'].plot.imshow()


ax.plot(
    da["lon"][:, 0] + 0.0485,
    ds["lat"][:, 0],
    color="white",
    linestyle="--",
    linewidth=0.2,
)

ax.plot(
    da["lon"][:, -1] - 0.0485,
    ds["lat"][:, -1],
    color="white",
    linestyle="--",
    linewidth=0.2,
)


da["lon"][:, 0].data  # alongtrack
da["lon"][0, :].data  # crosstrack

da["lat"][:, 0].data  # alongtrack
da["lat"][0, :].data  # crosstrack

import pyproj
from pyproj import Geod

g = Geod(ellps="WGS84")
fwd_az, back_az, dist = g.inv(*start_lonlat, *end_lonlat, radians=False)
lon_r, lat_r, _ = g.fwd(*start_lonlat, az=fwd_az, dist=dist + 50000)  # dist in m
fwd_az, back_az, dist = g.inv(*end_lonlat, *start_lonlat, radians=False)
lon_l, lat_l, _ = g.fwd(*end_lonlat, az=fwd_az, dist=dist + 50000)  # dist in m


ax

ds["nearSurfPrecipTotRate"].plot.imshow()
ds["nearSurfPrecipTotRate"].plot.pcolormesh(x="lon", y="lat")


ds["nearSurfPrecipTotRate"].sel(cross_track=slice(12, 37)).compute().data


a = ds_corra["nearSurfPrecipTotRate"].compute()
a.plot.imshow()
ds = ds_corra


ds_corra["lon"].sel(cross_track=slice(12, 37)).plot.imshow()
ds_corra["lon"].plot.imshow()

ds_dpr["lon"].plot.imshow()


ds_corra["nearSurfPrecipTotRate"].gpm_api.plot()
