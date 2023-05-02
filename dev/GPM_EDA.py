#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:19:14 2021

@author: ghiggi
"""
import datetime
import os
import sys

import matplotlib.pyplot as plt

# import h5py
# import netCDF4
# import pandas as pd
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from gpm_geo.utils_GPM import *
from pyresample_dev.utils_swath import *

from gpm_api.dataset import GPM_Dataset, GPM_variables, read_GPM  # read_GPM (importing here do)
from gpm_api.DPR.DPR_ENV import create_DPR_ENV

### GPM Scripts ####
from gpm_api.io import download_GPM_data
from gpm_api.utils.utils_cmap import *

##----------------------------------------------------------------------------.
#### Download data
# base_DIR = '/ltenas3/0_Data'
base_DIR = "/home/ghiggi/Data"

username = "gionata.ghiggi@epfl.ch"
# Servers
# start_time = datetime.datetime.strptime("2020-07-01 00:00:00", '%Y-%m-%d %H:%M:%S')
# end_time = datetime.datetime.strptime("2020-09-01 00:00:00", '%Y-%m-%d %H:%M:%S')

# Home PC
start_time = datetime.datetime.strptime("2020-07-01 00:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-07-02 00:00:00", "%Y-%m-%d %H:%M:%S")

product = "2A-DPR"

download_GPM_data(
    base_DIR=base_DIR,
    username=username,
    product=product,
    start_time=start_time,
    end_time=end_time,
    progress_bar=True,
    n_threads=5,  # 8
    transfer_tool="curl",
)

##-----------------------------------------------------------------------------.
####  Load GPM dataset
scan_mode = "NS"
variables = GPM_variables(product)
print(variables)

bbox = [20, 50, 30, 50]
bbox = [30, 35, 30, 50]
bbox = None
ds = GPM_Dataset(
    base_DIR=base_DIR,
    product=product,
    scan_mode=scan_mode,  # only necessary for 1B and 2A Ku/Ka/DPR
    variables=variables,
    start_time=start_time,
    end_time=end_time,
    bbox=bbox,
    enable_dask=True,
    chunks="auto",
)
print(ds)

### Plot along-cross_track plot daily data
da_precip = ds["precipRateNearSurface"].load()
# cmap = plt.get_cmap('Spectral').copy()
# cmap.set_under('white')
p = da_precip.plot.imshow(x="along_track", y="cross_track", cmap="Spectral", vmin=0.1)
p.cmap.set_under("white")
plt.show()

##----------------------------------------------------------------------------.
#### Identify precip areas
# --> label = 0 is rain below threshold
da_labels_area = get_areas_labels(ds["precipRateNearSurface"], thr=0.1)
da_labels_area.name = "precip_label_area"

# da_labels_area = da_labels.where(da_labels_area.values < 20)

p = da_labels_area.plot.imshow(x="along_track", y="cross_track", cmap="Spectral", vmin=1)
p.cmap.set_under("white")
plt.show()

p = da_labels_area.plot.imshow(x="along_track", y="cross_track", cmap="Spectral", vmin=1, vmax=20)
p.cmap.set_under("white")
plt.show()

##----------------------------------------------------------------------------.
#### Plot largest precipitating areas
# rows: along_track, cols: cross_track
ds["precip_label_area"] = da_labels_area
ds = ds.set_coords("precip_label_area")
n_plots = 20


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


for label_id in np.arange(1, n_plots):
    rmin, rmax, cmin, cmax = bbox(da_labels_area.data == label_id)
    ds_subset = ds.isel(along_track=slice(rmin, rmax + 1), cross_track=slice(cmin, cmax + 1))

    da_precip_subset = ds_subset["precipRateNearSurface"]
    p = da_precip_subset.plot.imshow(
        x="along_track",
        y="cross_track",
        # interpolation="nearest",
        interpolation="bilinear",
        # interpolation="bilinear", # "nearest", "bicubic"
        cmap=cmap,
        norm=norm,
        cbar_kwargs=cbar_kwargs,
    )
    # cmap="Spectral", vmin=0.5, vmax=50)
    # p.cmap.set_under("white")
    plt.show()

##----------------------------------------------------------------------------.
#### Project data into given projection
from pyresample import SwathDefinition
from pyresample.kd_tree import resample_nearest

lons = da_precip_subset["lon"]
lats = da_precip_subset["lat"]
data = da_precip_subset.values
swath_def = SwathDefinition(lons, lats)
area_def = swath_def.compute_optimal_bb_area()

data_proj = resample_nearest(swath_def, data, area_def, radius_of_influence=20000, fill_value=None)
crs = area_def.to_cartopy_crs()
fig, ax = plt.subplots(subplot_kw=dict(projection=crs))
coastlines = ax.coastlines()
ax.set_global()
img = plt.imshow(data_proj, transform=crs, extent=crs.bounds, origin="upper")
cbar = plt.colorbar()

##----------------------------------------------------------------------------.
#### Check pixel centroids
import cartopy.crs as ccrs

da_precip_subset1 = da_precip_subset.isel(along_track=slice(0, 5), cross_track=slice(0, 5))
lon = da_precip_subset1["lon"]
lat = da_precip_subset1["lat"]
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
da_precip_subset1.plot.pcolormesh(x="lon", y="lat", ax=ax, infer_intervals=False)
ax.scatter(lon, lat, transform=ccrs.PlateCarree(), c="r", s=1)

##----------------------------------------------------------------------------.
#### Plot on PlateCarree()
import cartopy.crs as ccrs

fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
p = da_precip_subset.plot.pcolormesh(
    x="lon",
    y="lat",
    ax=ax,
    cmap=cmap,
    norm=norm,
    cbar_kwargs=cbar_kwargs,
    infer_intervals=False,
)
# - Add colorbar details
cbar = p.colorbar
cbar.ax.set_yticklabels(clevs_str)
# - Add coastlines
ax.coastlines()
# - Add gridlines
gl = ax.gridlines(draw_labels=True)
gl.xlabels_top = False
gl.ylabels_right = False


# ax.set_global()

# - Add swath
lons = da_precip_subset["lon"]
lats = da_precip_subset["lat"]
ax.plot(lons[:, 0] + 0.0485, lats[:, 0], "--k")
ax.plot(lons[:, -1] - 0.0485, lats[:, -1], "--k")

# - Extend the plot extent
extent = list(p.axes.get_extent())  # (x_min, xmax, y_min, y_max)
extent[0] = extent[0] - 2
extent[1] = extent[1] + 2
extent[2] = extent[2] - 2
extent[3] = extent[3] + 2
p.axes.set_extent(tuple(extent))

# - Add stock img with transparency
# --> Need to be applied after extending the extent
p = ax.stock_img()
p.set_alpha(0.5)

##----------------------------------------------------------------------------.
#### Indentify area with maximum precip (slow ... to optimize)
da_labels_max = get_max_labels(ds["precipRateNearSurface"], min_thr=200)
da_labels_max = da_labels_max.where(da_labels_max > 0)

p = da_labels_max.plot.imshow(x="along_track", y="cross_track", cmap="Spectral", vmin=1)
p.cmap.set_under("white")
plt.show()

##----------------------------------------------------------------------------.
#### Plot precipitating with maximum intensity
ds["precip_label_max"] = da_labels_max
ds = ds.set_coords("precip_label_max")

for label_id in np.arange(1, len(np.unique(ds["precip_label_max"]))):
    rmin, rmax, cmin, cmax = bbox(ds["precip_label_max"] == label_id)
    ds_subset = ds.isel(along_track=slice(rmin, rmax + 1), cross_track=slice(cmin, cmax + 1))

    da_precip_subset = ds_subset["precipRateNearSurface"]
    p = da_precip_subset.plot.imshow(
        x="along_track",
        y="cross_track",
        interpolation="nearest",  # "nearest",
        # interpolation="bilinear", # "nearest", "bicubic"
        cmap=cmap,
        norm=norm,
        cbar_kwargs=cbar_kwargs,
    )
    # cmap="Spectral", vmin=0.5, vmax=50)
    # p.cmap.set_under("white")
    plt.show()


##----------------------------------------------------------------------------.
### Compute stats in each precip regions
# max_list = dask_image.ndmeasure.labeled_comprehension(image=da_precip.data,
#                                                       label_image=labels,
#                                                       index=index,
#                                                       out_dtype=float,
#                                                       default=None,
#                                                       func=np.max)
##----------------------------------------------------------------------------.
##### Count occurence
da1 = da_precip.where(da_precip.values > 100)
da1.count()
np.invert(np.isnan(da1.values.flatten())).sum()

##----------------------------------------------------------------------------.
### Plot distribution
da_precip.where(da_precip.values > 1).plot.hist(xlim=(1, 100), nbins=100)

##----------------------------------------------------------------------------.
#### Interscan timestep (1s)
# --> 10 min GEO --> 60*60 = 360 scans
timedelta = da_precip.time.values[1:] - da_precip.time.values[0:-1]
timedelta = timedelta.astype("m8[s]")
timedelta = timedelta[timedelta < np.timedelta64(2, "s")]
plt.hist(timedelta.astype(int))

import cartopy.crs as ccrs

##----------------------------------------------------------------------------.
#### GEO bounding box polygon
import satpy.resample
from pyresample.geometry import get_geostationary_bounding_box
from shapely.geometry import Polygon

area_def = satpy.resample.get_area_def("goes_east_abi_f_500m")
area_def = satpy.resample.get_area_def("goes_east_abi_f_1km")
area_def = satpy.resample.get_area_def("goes_east_abi_f_2km")

lon_arr, lat_arr = get_geostationary_bounding_box(area_def, nb_points=50)

# - Define shapely Polygon
areadef_bbox_geom = Polygon(zip(lon_arr, lat_arr))
areadef_bbox_geom.bounds

# - Plot Polygon
# extent = extent_from_bounds(areadef_bbox_geom.bounds) # minx, maxx, miny, maxy
# extent = extend_extent(extent, 5, 5)
# areadef_bbox_geom = ensure_shapely_counterclockwise(areadef_bbox_geom)

proj_crs = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw=dict(projection=proj_crs))
ax.add_geometries(
    [areadef_bbox_geom],
    crs=ccrs.PlateCarree(),
    facecolor="red",
    edgecolor="black",
    alpha=0.4,
)
# ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.stock_img()
ax.coastlines()  # ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.75)
gl = ax.gridlines(draw_labels=True, linestyle="--")
gl.xlabels_top = False
gl.ylabels_right = False

import cartopy.crs as ccrs

##----------------------------------------------------------------------------.
#### GPM time to scan across GEO during one orbit
from pyresample.geometry import SwathDefinition

lons = da_precip["lon"].values
lats = da_precip["lat"].values
lons = lons[3497:6867]
lats = lats[3497:6867]
swath_def = SwathDefinition(lons, lats)
lon_edge, lat_edge = swath_def.get_edge_lonlats()
swath_def_bbox_geom = Polygon(zip(lon_edge, lat_edge))

proj_crs = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw=dict(projection=proj_crs))
ax.add_geometries(
    [areadef_bbox_geom],
    crs=ccrs.PlateCarree(),
    facecolor="red",
    edgecolor="black",
    alpha=0.4,
)

ax.add_geometries(
    [swath_def_bbox_geom],
    crs=ccrs.PlateCarree(),
    facecolor="orange",
    edgecolor="black",
    alpha=0.4,
)
# ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.stock_img()
ax.coastlines()  # ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.75)
gl = ax.gridlines(draw_labels=True, linestyle="--")
gl.xlabels_top = False
gl.ylabels_right = False

da_precip_cross_geo = da_precip.isel(along_track=slice(3497, 6867))
da_precip_cross_geo["time"][0]
da_precip_cross_geo["time"][-1]
dt = da_precip_cross_geo["time"][-1] - da_precip_cross_geo["time"][0]
dt.values.astype("m8[m]")  # 39 minutes

import cartopy.crs as ccrs

##----------------------------------------------------------------------------.
#### Plot GPM swaths
from pyresample.geometry import SwathDefinition

lons = da_precip["lon"].values
lats = da_precip["lat"].values
swath_def = SwathDefinition(lons, lats)
lon_edge, lat_edge = swath_def.get_edge_lonlats()
vertices = np.vstack((lon_edge.data, lat_edge.data)).transpose()
# side_x, side_y = swath_def.get_bbox_lonlats()
# x_coords = np.concatenate(side_x)
# y_coords = np.concatenate(side_y)
# vertices = np.vstack((x_coords, y_coords)).transpose()

list_vertices = get_antimeridian_safe_polygon_vertices(vertices)
list_polygons = [Polygon(v) for v in list_vertices]
swath_def_bbox_geom = MultiPolygon(list_polygons)

proj_crs = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw=dict(projection=proj_crs))
# ax.add_geometries([areadef_bbox_geom], crs=ccrs.PlateCarree(),
#                   facecolor = 'red', edgecolor='black', alpha=0.4)

ax.add_geometries(
    [swath_def_bbox_geom],
    crs=ccrs.PlateCarree(),
    facecolor="orange",
    edgecolor="black",
    alpha=0.4,
)
# ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.stock_img()
ax.coastlines()  # ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.75)
gl = ax.gridlines(draw_labels=True, linestyle="--")
gl.xlabels_top = False
gl.ylabels_right = False
