#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:04:56 2023

@author: ghiggi
"""
import datetime

import gpm_api

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


# patch_gen = da.gpm_api.labels_patch_generator(
#     # variable=variable,
#     min_value_threshold=3,
#     min_area_threshold=5,
#     sort_by="maximum",  # area
#     sort_decreasing=True,
#     n_patches=10,
#     # Output options
#     label_name=label_name,
#     patch_size=(20, 20)
#     n_patches=n_patches,
#     n_labels=n_labels,
#     labels_id=labels_id,
#     highlight_label_id=highlight_label_id,
#     # Patch extraction Options
#     padding=padding,
#     centered_on=centered_on,
#     n_patches_per_label=n_patches_per_label,
#     n_patches_per_partition=n_patches_per_partition,
#     # Tiling/Sliding Options
#     partitioning_method=partitioning_method,
#     n_partitions_per_label=n_partitions_per_label,
#     kernel_size=kernel_size,
#     buffer=buffer,
#     stride=stride,
#     include_last=include_last,
#     ensure_slice_size=ensure_slice_size,
#     debug=debug,
# )

# list_patch = list(patch_gen)


# ----------------------------------------------------------------------------.
# DEBUG
import numpy as np

from gpm_api.patch.labels import label_xarray_object
from gpm_api.patch.labels_patch import get_patches_from_labels
from gpm_api.visualization.labels import plot_label

# Args labels
min_value_threshold = 1
max_value_threshold = np.inf
min_area_threshold = 5
max_area_threshold = np.inf
footprint = None
sort_by = "area"
sort_decreasing = True
label_name = "label"

# Retrieve labeled xarray object
xr_obj = label_xarray_object(
    da,
    min_value_threshold=min_value_threshold,
    max_value_threshold=max_value_threshold,
    min_area_threshold=min_area_threshold,
    max_area_threshold=max_area_threshold,
    footprint=footprint,
    sort_by=sort_by,
    sort_decreasing=sort_decreasing,
    label_name=label_name,
)

# Build a generator returning patches around rainy areas
n_patches = 20
patch_size = (100, 100)
n_labels = None
labels_id = None
highlight_label_id = False
centered_on = "label_bbox"
padding = 0
n_patches_per_label = np.Inf
n_patches_per_partition = 1
partitioning_method = None
n_partitions_per_label = None
kernel_size = None
buffer = 0
stride = None
include_last = True
ensure_slice_size = True
debug = True


da_patch_gen = get_patches_from_labels(
    xr_obj,
    label_name=label_name,
    patch_size=patch_size,
    variable=variable,
    # Output options
    n_patches=n_patches,
    n_labels=n_labels,
    labels_id=labels_id,
    highlight_label_id=highlight_label_id,
    # Patch extraction Options
    padding=padding,
    centered_on=centered_on,
    n_patches_per_label=n_patches_per_label,
    n_patches_per_partition=n_patches_per_partition,
    # Tiling/Sliding Options
    partitioning_method=partitioning_method,
    n_partitions_per_label=n_partitions_per_label,
    kernel_size=kernel_size,
    buffer=buffer,
    stride=stride,
    include_last=include_last,
    ensure_slice_size=ensure_slice_size,
    debug=debug,
)

list_patch = list(da_patch_gen)

da_patch = list_patch[7]
da_patch.gpm_api.plot_image()
da_patch[label_name].plot.imshow()
plot_label(da_patch[label_name])

for da_patch in list_patch:
    da_patch.gpm_api.plot_image()

for da_patch in list_patch:
    print(da_patch.shape)

for da_patch in list_patch:
    plot_label(da_patch[label_name])
