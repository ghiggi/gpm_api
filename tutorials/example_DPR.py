#!/usr/bin/env python3
"""
Created on Sat Dec 10 14:16:00 2022

@author: ghiggi
"""
import datetime

import gpm
from gpm.utils.geospatial import get_country_extent

start_time = datetime.datetime.strptime("2020-07-05 02:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-07-05 06:00:00", "%Y-%m-%d %H:%M:%S")
product = "2A-DPR"
product_type = "RS"
version = 7

# Download the data
gpm.download(
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
ds = gpm.open_dataset(
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
ds = gpm.open_dataset(
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
# xr.Dataset methods provided by gpm
print(dir(ds.gpm))

# xr.DataArray methods provided by gpm
print(dir(ds[variable].gpm))

#### - Check geometry and dimensions
ds.gpm.is_grid  # False
ds.gpm.is_orbit  # True

ds.gpm.is_spatial_2d  # False, because not only cross-track and along-track
ds["zFactorFinal"].gpm.is_spatial_2d  # False, because there is the range dimension
ds["zFactorFinal"].isel(range=[0]).gpm.is_spatial_2d  # True,  because selected a single range
ds["zFactorFinal"].isel(range=0).gpm.is_spatial_2d  # True,  because no range dimension anymore

ds.gpm.has_contiguous_scans
ds.gpm.get_slices_contiguous_scans()  # List of along-track slices with contiguous scans

ds.gpm.is_regular

#### - Get pyresample SwathDefinition
ds.gpm.pyresample_area

#### - Get the dataset title
ds.gpm.title(add_timestep=True)
ds.gpm.title(add_timestep=False)

ds[variable].gpm.title(add_timestep=False)
ds[variable].gpm.title(add_timestep=True)

####--------------------------------------------------------------------------.
#### Plotting with gpm
# - In geographic space
ds[variable].gpm.plot_map()
ds[variable].isel(along_track=slice(0, 500)).gpm.plot_map()

# - In swath view
ds[variable].gpm.plot_image()  # In swath view
ds[variable].isel(along_track=slice(0, 500)).gpm.plot_image()

####--------------------------------------------------------------------------.
#### Zoom on a geographic area
title = ds.gpm.title(add_timestep=False)
extent = get_country_extent("United States")
p = ds[variable].gpm.plot_map()
p.axes.set_extent(extent)
p.axes.set_title(label=title)

####--------------------------------------------------------------------------.
#### Crop the dataset
# - A orbit can cross an area multiple times
# - Therefore first retrieve the crossing slices, and then select the intersection of interest
# - The extent must be specified following the cartopy and matplotlib convention
# ---> extent = [lon_min, lon_max, lat_min, lat_max]
extent = get_country_extent("United States")
list_isel_dict = ds.gpm.get_crop_slices_by_extent(extent)
print(list_isel_dict)
for isel_dict in list_isel_dict:
    da_subset = ds[variable].isel(isel_dict)
    slice_title = da_subset.gpm.title(add_timestep=True)
    p = da_subset.gpm.plot_map()
    p.axes.set_extent(extent)
    p.axes.set_title(label=slice_title)

####--------------------------------------------------------------------------.
