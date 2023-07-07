#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:41:22 2023

@author: ghiggi
"""
import datetime

import numpy as np
import xarray as xr

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

####---------------------------------------------------------------------------.
import numpy as np

from gpm_api.patch.labels import xr_get_areas_labels

# Args labels
min_value_threshold = 1
max_value_threshold = np.inf
min_area_threshold = 5
max_area_threshold = np.inf
footprint = None
sort_by = "area"
sort_decreasing = True

data_array = da

da_labels, n_labels, values = xr_get_areas_labels(
    data_array=data_array,
    min_value_threshold=min_value_threshold,
    max_value_threshold=max_value_threshold,
    min_area_threshold=min_area_threshold,
    max_area_threshold=max_area_threshold,
    footprint=footprint,
    sort_by=sort_by,
    sort_decreasing=sort_decreasing,
)

da_labels.plot.imshow()  # 0 are plotted

####---------------------------------------------------------------------------.
#### Check equality
from gpm_api.patch.labels import label_xarray_object

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
    label_name="label",
)
xr_obj1 = da.gpm_api.label_object(
    min_value_threshold=min_value_threshold,
    max_value_threshold=max_value_threshold,
    min_area_threshold=min_area_threshold,
    max_area_threshold=max_area_threshold,
    footprint=footprint,
    sort_by=sort_by,
    sort_decreasing=sort_decreasing,
    label_name="label",
)

da_labels.plot.imshow()  # 0 are plotted

xr_obj["label"].plot.imshow()  # 0 became nan --> Are not plotted
xr_obj1["label"].plot.imshow()
xr.testing.assert_equal(xr_obj["label"], xr_obj1["label"])
np.testing.assert_equal(xr_obj["label"].data, da_labels.where(da_labels > 0).data)

####---------------------------------------------------------------------------.
#### Visualize labels patches

from gpm_api.visualization.labels import plot_label_patches

# Args labels
min_value_threshold = 1
max_value_threshold = np.inf
min_area_threshold = 5
max_area_threshold = np.inf
footprint = None
footprint = 15
sort_by = "area"
sort_decreasing = True

xr_obj = da.gpm_api.label_object(
    min_value_threshold=min_value_threshold,
    max_value_threshold=max_value_threshold,
    min_area_threshold=min_area_threshold,
    max_area_threshold=max_area_threshold,
    footprint=footprint,
    sort_by=sort_by,
    sort_decreasing=sort_decreasing,
    label_name="label",
)

# xr_obj["label"].plot.imshow()

# Plot the patches around the labels
n_patches = 10
label_name = "label"
patch_size = (100, 100)
highlight_label_id = False
labels_id = None
n_labels = None
centered_on = "label_bbox"
padding = 0
n_patches_per_partition = 1
n_patches_per_label = np.Inf
partitioning_method = None
n_partitions_per_label = None
kernel_size = (None,)
buffer = 0
stride = None
include_last = True
ensure_slice_size = True
debug = True

add_colorbar = True
interpolation = "nearest"
cmap = "Paired"
fig_kwargs = {}

plot_label_patches(
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
    # Plotting Options
    add_colorbar=add_colorbar,
    interpolation=interpolation,
    cmap=cmap,
)
