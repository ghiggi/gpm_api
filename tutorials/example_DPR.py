#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:16:00 2022

@author: ghiggi
"""
import gpm_api
import datetime

base_dir = "/home/ghiggi/GPM"
start_time = datetime.datetime.strptime("2020-07-05 02:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-07-05 06:00:00", "%Y-%m-%d %H:%M:%S")
product = "2A-DPR"
product_type = "RS"
version = 7
username = "gionata.ghiggi@epfl.ch"

# Download the data
gpm_api.download(
    base_dir=base_dir,
    username=username,
    product=product,
    product_type=product_type,
    version=version,
    start_time=start_time,
    end_time=end_time,
    force_download=False,
    verbose=True,
    progress_bar=True,
    check_integrity=False,
)

####--------------------------------------------------------------------------.
#### Load GPM DPR 2A product dataset (with group prefix)
ds = gpm_api.open_dataset(
    base_dir=base_dir,
    product=product,
    product_type=product_type,
    version=version,
    start_time=start_time,
    end_time=end_time,
    prefix_group=True,
)

# Display dataset structure
ds

# Available variables
variables = list(ds.data_vars)
print(variables)

#### Load GPM DPR 2A product dataset (without group prefix)
variables = [
    "airTemperature",
    "precipRate",
    "paramDSD",
    "zFactorFinal",
    "zFactorMeasured",
    "precipRateNearSurface",
    "precipRateESurface",
    "precipRateESurface2",
    "zFactorFinalESurface",
    "zFactorFinalNearSurface",
    "heightZeroDeg",
    "binEchoBottom",
    "landSurfaceType",
]
ds = gpm_api.open_dataset(
    base_dir=base_dir,
    product=product,
    product_type=product_type,
    version=version,
    start_time=start_time,
    end_time=end_time,
    variables=variables,
    prefix_group=False,
)
ds

# Available variables
variables = list(ds.data_vars)
print(variables)

# Select variable
variable = "precipRateNearSurface"

####--------------------------------------------------------------------------.
#### GPM-API methods
# xr.Dataset methods provided by gpm_api
print(dir(ds.gpm_api))

# xr.DataArray methods provided by gpm_api
print(dir(ds[variable].gpm_api))

#### - Check geometry and dimensions
ds.gpm_api.is_grid   # False
ds.gpm_api.is_orbit  # True

ds.gpm_api.is_spatial_2d  # False, because not only cross-track and along-track
ds["zFactorFinal"].gpm_api.is_spatial_2d  # False, because there is the range dimension
ds["zFactorFinal"].isel(range=[0]).gpm_api.is_spatial_2d  # True,  because selected a single range
ds["zFactorFinal"].isel(range=0).gpm_api.is_spatial_2d  # True,  because no range dimension anymore

ds.gpm_api.has_contiguous_scans
ds.gpm_api.get_slices_contiguous_scans()  # List of along-track slices with contiguous scans

ds.gpm_api.is_regular

#### - Get pyresample SwathDefinition
ds.gpm_api.pyresample_area

#### - Get the dataset title
ds.gpm_api.title(add_timestep=True)
ds.gpm_api.title(add_timestep=False)

ds[variable].gpm_api.title(add_timestep=False)
ds[variable].gpm_api.title(add_timestep=True)

####--------------------------------------------------------------------------.
#### Plotting with gpm_api
# - In geographic space
ds[variable].gpm_api.plot_map()  
ds[variable].isel(along_track=slice(0, 500)).gpm_api.plot_map()

# - In swath view
ds[variable].gpm_api.plot_image()  # In swath view
ds[variable].isel(along_track=slice(0, 500)).gpm_api.plot_image()

####--------------------------------------------------------------------------.
#### Zoom on a geographic area 
from gpm_api.utils.countries import get_country_extent
title = ds.gpm_api.title(add_timestep=False)
extent = get_country_extent("United States")
p = ds[variable].gpm_api.plot_map()  
p.axes.set_extent(extent)
p.axes.set_title(label=title)

####--------------------------------------------------------------------------.
#### Crop the dataset 
# - A orbit can cross an area multiple times 
# - Therefore first retrieve the crossing slices, and then select the intersection of interest
# - The extent must be specified following the cartopy and matplotlib convention
# ---> extent = [lon_min, lon_max, lat_min, lat_max]
extent = get_country_extent("United States")
list_slices = ds.gpm_api.get_crop_slices_by_extent(extent)
print(list_slices)
for slc in list_slices:
    da_subset = ds[variable].isel(along_track=slc)
    slice_title = da_subset.gpm_api.title(add_timestep=True)
    p = da_subset.gpm_api.plot_map()  
    p.axes.set_extent(extent)
    p.axes.set_title(label=slice_title)

####--------------------------------------------------------------------------.
#### Plot precipitation patches 
da = ds[variable].isel(along_track=slice(0, 10000))
da.gpm_api.plot_patches(
    min_value_threshold=10,
    min_area_threshold=5,
    footprint=3,
    sort_by="max", # "area"
    sort_decreasing=True,
    n_patches=10,
    min_patch_size = (48, 20),
    interpolation="nearest",
)

