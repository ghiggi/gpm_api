#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 14:16:00 2022

@author: ghiggi
"""
import gpm_api 
import datetime 

base_dir = '/home/ghiggi/GPM'
start_time = datetime.datetime.strptime("2019-07-13 11:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2019-07-13 13:00:00", '%Y-%m-%d %H:%M:%S')
product = 'IMERG-FR'  # 'IMERG-ER' 'IMERG-LR'
product_type = 'RS'
version = 6
username = "gionata.ghiggi@epfl.ch"

# Download the data
gpm_api.download(base_dir = base_dir, 
                 username = username,
                 product = product, 
                 product_type = product_type,
                 version = version, 
                 start_time = start_time,
                 end_time = end_time, 
                 force_download = False,                 
                 verbose=True, 
                 progress_bar=True,
                 check_integrity=False)


# Load IMERG dataset
ds = gpm_api.open_dataset(base_dir = base_dir,
                          product = product, 
                          product_type = product_type,
                          version = version,
                          start_time = start_time,
                          end_time = end_time)
ds

# Available variables 
variables = list(ds.data_vars)
print(variables)

# Select variable 
variable = "precipitationCal"

# Plot with xarray 
ds[variable].isel(time=0).plot.imshow(x="lon", y="lat")

#### GPM-API methods
# xr.Dataset methods provided by gpm_api
print(dir(ds.gpm_api))

# xr.DataArray methods provided by gpm_api
print(dir(ds[variable].gpm_api))

#### Check geometry and dimensions 
ds.gpm_api.is_grid  # True
ds.gpm_api.is_orbit # False 

ds.gpm_api.is_spatial_2D_field                 # False, because of multiple timesteps 
ds.isel(time=[0]).gpm_api.is_spatial_2D_field  # True,  because of a single timesteps 
ds.isel(time=0).gpm_api.is_spatial_2D_field    # True,  because no time dimension anymore 

ds.gpm_api.has_regular_timesteps
ds.gpm_api.get_regular_time_slices()   # List of time slices with regular timesteps 

ds.gpm_api.pyresample_area # TODO: not yet implemented 

#### Plotting with gpm_api a single timestep
ds[variable].isel(time=0).gpm_api.plot_map()    # With cartopy
ds[variable].isel(time=0).gpm_api.plot_image()  # Without cartopy

#### Plotting with gpm_api multiple timestep or multiple variables 
# TODO

#### Title 
ds.gpm_api.title(add_timestep=True)
ds.isel(time=0).gpm_api.title(add_timestep=True) 

ds[variable].gpm_api.title(add_timestep=False) 
ds[variable].gpm_api.title(add_timestep=True) 
ds[variable].isel(time=0).gpm_api.title(add_timestep=True) 

#### Cropping 
# Crop by bbox 
bbox = (6.02260949059, 10.4427014502, 45.7769477403, 47.8308275417) # Switzerland 
ds_ch = ds.gpm_api.crop(bbox=bbox)
ds_ch[variable].isel(time=0).gpm_api.plot_map()

# Crop by country 
ds_ch = ds.gpm_api.crop_by_country("Switzerland")
ds_ch[variable].isel(time=0).gpm_api.plot_map()

#### Patch generator 
patch_gen = ds.isel(time=0).gpm_api.patch_generator(variable=variable, 
                                                    min_value_threshold=3, 
                                                    min_area_threshold=5,
                                                    sort_by="max", # area
                                                    sort_decreasing=True,
                                                    n_patches=10,
                                                    patch_margin=(20,20),
                                                    )
list_patch = list(patch_gen)

#### Plot Patches 
ds[variable].isel(time=0).gpm_api.plot_patches(min_value_threshold=3, 
                                               min_area_threshold=5,
                                               sort_by="max", # area
                                               sort_decreasing=True,
                                               n_patches=10,
                                               patch_margin=(20,20),
                                               interpolation="nearest",
                                               )


ds.gpm_api.plot_patches(variable=variable, 
                        min_value_threshold=3, 
                        min_area_threshold=5,
                        sort_by="max", # area
                        sort_decreasing=True,
                        n_patches=10,
                        patch_margin=(20,20),
                        interpolation="nearest",
                        )


