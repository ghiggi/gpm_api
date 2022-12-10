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


# Load GPM DPR 2A product dataset (with group prefix)
ds = gpm_api.open_dataset(
    base_dir=base_dir,
    product=product,
    product_type=product_type,
    version=version,
    start_time=start_time,
    end_time=end_time,
    prefix_group=True,
)
ds

# Available variables
variables = list(ds.data_vars)
print(variables)

# Load GPM DPR 2A product dataset (without group prefix)
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

# Plot with xarray
ds[variable].plot.imshow(x="along_track", y="cross_track")

#### GPM-API methods
# xr.Dataset methods provided by gpm_api
print(dir(ds.gpm_api))

# xr.DataArray methods provided by gpm_api
print(dir(ds[variable].gpm_api))

#### Check geometry and dimensions
ds.gpm_api.is_grid  # False
ds.gpm_api.is_orbit  # True

ds.gpm_api.is_spatial_2D_field  # False, because not only cross-track and along-track
ds[
    "zFactorFinal"
].gpm_api.is_spatial_2D_field  # False, because there is the range dimension
ds["zFactorFinal"].isel(
    range=[0]
).gpm_api.is_spatial_2D_field  # True,  because selected a single range
ds["zFactorFinal"].isel(
    range=0
).gpm_api.is_spatial_2D_field  # True,  because no range dimension anymore

ds.gpm_api.has_regular_timesteps
ds.gpm_api.get_regular_time_slices()  # List of time slices with regular timesteps

#### Get pyresample SwathDefinition
ds.gpm_api.pyresample_area

#### Plotting with gpm_api
# --> TODO: SOLVE BUG AT ANTIMERIDIAN IN CARTOPY !!!
ds[variable].gpm_api.plot_map()  # In geographic space
ds[variable].isel(along_track=slice(0, 500)).gpm_api.plot_map()
ds[variable].gpm_api.plot_image()  # In swath view

ds[variable].isel(along_track=slice(0, 500)).gpm_api.plot_image()

#### Title
ds.gpm_api.title(add_timestep=True)
ds.gpm_api.title(add_timestep=False)

ds[variable].gpm_api.title(add_timestep=False)
ds[variable].gpm_api.title(add_timestep=True)

#### Cropping
# Crop by bbox
bbox = (6.02260949059, 45.7769477403, 10.4427014502, 47.8308275417)  # Switzerland
ds_ch = ds.gpm_api.crop(bbox=bbox)
ds_ch[variable].gpm_api.plot_map()

# Crop by country
ds_usa = ds.gpm_api.crop_by_country("United States")
ds_usa[variable].gpm_api.plot_map()

# TODO: get_crop_slices !!!!
# --> Select the largest slice


#### Patches
# TODO EXAMPLE
patch_generator
plot_patches
