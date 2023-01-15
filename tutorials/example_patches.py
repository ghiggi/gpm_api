#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 12:00:18 2023

@author: ghiggi
"""
import gpm_api
import datetime

base_dir = "/home/ghiggi/GPM"
start_time = datetime.datetime.strptime("2019-07-13 11:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2019-07-13 13:00:00", "%Y-%m-%d %H:%M:%S")
product = "IMERG-FR"  # 'IMERG-ER' 'IMERG-LR'
product_type = "RS"
version = 6
username = "gionata.ghiggi@epfl.ch"

# Download the data
# gpm_api.download(
#     base_dir=base_dir,
#     username=username,
#     product=product,
#     product_type=product_type,
#     version=version,
#     start_time=start_time,
#     end_time=end_time,
#     force_download=False,
#     verbose=True,
#     progress_bar=True,
#     check_integrity=False,
# )


# Load IMERG dataset
ds = gpm_api.open_dataset(
    base_dir=base_dir,
    product=product,
    product_type=product_type,
    version=version,
    start_time=start_time,
    end_time=end_time,
)
ds

# Available variables
variables = list(ds.data_vars)
print(variables)

# Select variable
variable = "precipitationCal"

da = ds[variable].isel(time=0)

####--------------------------------------------------------------------------.
#### Image Label extraction 
from gpm_api.patch.labels import xr_get_areas_labels
 
min_value_threshold=3
max_value_threshold=np.inf

min_area_threshold=5
max_area_threshold=np.inf
 
sort_by="max"  # area
sort_decreasing=True

footprint=None
 
da_labels, n_labels, values = xr_get_areas_labels(
    data_array=da,
    min_value_threshold=min_value_threshold,
    max_value_threshold=max_value_threshold,
    min_area_threshold=min_area_threshold,
    max_area_threshold=max_area_threshold,
    footprint=footprint,
    sort_by=sort_by,
    sort_decreasing=sort_decreasing,
)


####--------------------------------------------------------------------------.
#### Patch generator
 
  
patch_gen = da.gpm_api.labels_patch_generator(
    min_value_threshold=min_value_threshold,
    min_area_threshold=min_area_threshold,
    footprint=footprint, 
    sort_by="max",  # area
    sort_decreasing=True,
    n_patches=10,
)
list_patch = list(patch_gen)
 
