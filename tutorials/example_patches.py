#!/usr/bin/env python3
"""
Created on Wed Jan 11 12:00:18 2023

@author: ghiggi
"""
import datetime

import numpy as np

import gpm_api
from gpm_api.patch.labels import xr_get_areas_labels

start_time = datetime.datetime.strptime("2019-07-13 11:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2019-07-13 13:00:00", "%Y-%m-%d %H:%M:%S")
product = "IMERG-FR"  # 'IMERG-ER' 'IMERG-LR'
product_type = "RS"
version = 6


# Download the data
# gpm_api.download(
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
min_value_threshold = 3
max_value_threshold = np.inf

min_area_threshold = 5
max_area_threshold = np.inf

sort_by = "max"  # area
sort_decreasing = True

footprint = None

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
    # Label Options
    min_value_threshold=min_value_threshold,
    min_area_threshold=min_area_threshold,
    footprint=footprint,
    sort_by="maximum",  # area
    sort_decreasing=True,
    # Patch Output options
    n_patches=10,
    patch_size=(49, 49),
    # Label Patch Extraction Options
    centered_on="max",
)
list_patch = list(patch_gen)
