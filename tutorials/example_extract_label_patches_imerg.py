#!/usr/bin/env python3
"""
Created on Tue Jul 11 16:10:31 2023

@author: ghiggi
"""
import datetime

import matplotlib.pyplot as plt
import numpy as np
import ximage  # noqa

import gpm_api

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

####---------------------------------------------------------------------------.
###################
#### Labelling ####
###################
min_value_threshold = 1
max_value_threshold = np.inf
min_area_threshold = 5
max_area_threshold = np.inf
footprint = None
sort_by = "area"
sort_decreasing = True
label_name = "label"

# Retrieve labeled xarray object
xr_obj = da.ximage.label(
    min_value_threshold=min_value_threshold,
    max_value_threshold=max_value_threshold,
    min_area_threshold=min_area_threshold,
    max_area_threshold=max_area_threshold,
    footprint=footprint,
    sort_by=sort_by,
    sort_decreasing=sort_decreasing,
    label_name=label_name,
)

# Plot full label array
xr_obj[label_name].plot.imshow()  # 0 are plotted
plt.show()

# Plot label with ximage
xr_obj[label_name].ximage.plot_labels()

# Plot label with gpm_api
gpm_api.plot_labels(xr_obj[label_name])

# ----------------------------------------------------------------------------.
##################################
#### Label Patches Extraction ####
##################################
patch_size = (100, 100)

# Output Options
n_patches = 10
n_labels = None
labels_id = None
highlight_label_id = False
# Patch Extraction Options
centered_on = "label_bbox"
padding = 0
n_patches_per_label = np.Inf
n_patches_per_partition = 1
# Tiling/Sliding Options
partitioning_method = None
n_partitions_per_label = None
kernel_size = None
buffer = 0
stride = None
include_last = True
ensure_slice_size = True
debug = True
verbose = True

da_patch_gen = xr_obj.ximage.label_patches(
    label_name=label_name,
    patch_size=patch_size,
    variable=variable,
    # Output Options
    n_patches=n_patches,
    n_labels=n_labels,
    labels_id=labels_id,
    highlight_label_id=highlight_label_id,
    # Patch Extraction Options
    centered_on=centered_on,
    padding=padding,
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
    verbose=verbose,
)

gpm_api.plot_patches(da_patch_gen, variable=variable, interpolation="nearest")


# list_patch = list(da_patch_gen)
# label_id, da_patch = list_patch[7]
# for label_id, da_patch in list_patch:
#     da_patch.gpm_api.plot_image()
