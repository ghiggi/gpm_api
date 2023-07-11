#!/usr/bin/env python3
"""
Created on Wed Jan 11 15:41:22 2023

@author: ghiggi
"""
import datetime

import matplotlib.pyplot as plt
import numpy as np
import ximage  # noqa

import gpm_api
from gpm_api.visualization import plot_labels

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
footprint = 15
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

# Plot label subsets with ximage
xr_obj[label_name].ximage.plot_labels()
xr_obj[label_name].isel(lat=slice(0, 100), lon=slice(0, 100)).ximage.plot_labels()
xr_obj[label_name].isel(lat=slice(0, 500), lon=slice(0, 500)).ximage.plot_labels()

# Plot label subsets with gpm_api (nice axis with orbit data)
plot_labels(
    xr_obj[label_name],
    add_colorbar=True,
    interpolation="nearest",
    cmap="Paired",
)

####---------------------------------------------------------------------------.
##############################
#### Visualize each label ####
##############################
patch_size = (100, 100)
# Output Options
n_patches = 10
label_name = "label"
highlight_label_id = False
labels_id = None
n_labels = None
# Patch Extraction Options
centered_on = "label_bbox"
padding = 0

# Define the patch generator
patch_gen = xr_obj.ximage.label_patches(
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
    # Tiling/Sliding Options
    partitioning_method=None,
)

# Plot patches around the labels
# list_da = list(patch_gen)
# label_id, da = list_da[0]
for label_id, da in patch_gen:
    plot_labels(
        da[label_name],
        add_colorbar=True,
        interpolation="nearest",
        cmap="Paired",
    )
    plt.show()


####---------------------------------------------------------------------------.
