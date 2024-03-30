#!/usr/bin/env python3
"""
Created on Mon Nov 29 13:19:14 2021

@author: ghiggi
"""
import datetime
from pprint import pprint

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import ximage  # noqa
from matplotlib.colors import LogNorm

import gpm

##----------------------------------------------------------------------------.
#### Download data
start_time = datetime.datetime.strptime("2016-03-09 10:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2016-03-09 11:00:00", "%Y-%m-%d %H:%M:%S")
product = "2A-DPR"
version = 7

gpm.download(
    product=product,
    start_time=start_time,
    end_time=end_time,
    version=version,
    progress_bar=True,
    n_threads=2,  # 8
    transfer_tool="CURL",
)

##-----------------------------------------------------------------------------.
####  Load GPM dataset
ds = gpm.open_dataset(
    product=product,
    start_time=start_time,
    end_time=end_time,
    chunks={},
)
print(ds)

variable = "precipRateNearSurface"
da_precip = ds[variable].load()

# Plot along-cross_track data
p = da_precip.gpm.plot_image()

# Plot map
p = da_precip.gpm.plot_map()

# Plot swath coverage
p = da_precip.gpm.plot_swath()

# Plot swath lines
p = da_precip.gpm.plot_swath_lines()
p.axes.set_global()

##----------------------------------------------------------------------------.
#### Identify precipitation area sorted by maximum intensity
# --> label = 0 is rain below min_value_threshold
label_name = "label_precip_max_intensity"
da_precip = da_precip.ximage.label(min_value_threshold=1, label_name=label_name, sort_by="maximum")
gpm.plot_labels(da_precip[label_name])

# # Select only label with maximum intensity (set other labels to np.nan)
# da_precip[label_name] = da_precip[label_name].where(da_precip[label_name] == 1)
# gpm.plot_labels(da_precip[label_name])

##----------------------------------------------------------------------------.
#### Identify precipitation area sorted by maximum area
# --> label = 0 is rain below min_value_threshold
label_name = "label_precip_max_area"
da_precip = da_precip.ximage.label(min_value_threshold=0.1, label_name=label_name, sort_by="area")
gpm.plot_labels(da_precip[label_name])

##----------------------------------------------------------------------------.
#### Plot largest precipitating areas and associated labels
# Define the patch generator
label_name = "label_precip_max_area"
labels_id = None
labels_id = [1, 2, 3]
patch_size = 2  # min patch size
centered_on = "label_bbox"
padding = 0
highlight_label_id = False


# Plot labels (around each label)
patch_gen = da_precip.ximage.label_patches(
    label_name=label_name,
    patch_size=patch_size,
    # Output options
    labels_id=labels_id,
    highlight_label_id=highlight_label_id,
    # Patch extraction Options
    centered_on=centered_on,
    padding=padding,
)
gpm.plot_labels(patch_gen, label_name=label_name)

# Plot patches (around each label)
patch_gen = da_precip.ximage.label_patches(
    label_name=label_name,
    patch_size=patch_size,
    # Output options
    labels_id=labels_id,
    highlight_label_id=highlight_label_id,
    # Patch extraction Options
    centered_on=centered_on,
    padding=padding,
)

gpm.plot_patches(patch_gen, variable=variable, interpolation="bilinear")

##----------------------------------------------------------------------------.
#### Plot most intense precipitating areas and associated labels
# Define the patch generator
label_name = "label_precip_max_intensity"
labels_id = None
n_patches = 20
patch_size = (49, 49)
centered_on = "max"
padding = 0
highlight_label_id = False


# Plot labels (around each label)
patch_gen = da_precip.ximage.label_patches(
    label_name=label_name,
    patch_size=patch_size,
    # Output options
    n_patches=n_patches,
    labels_id=labels_id,
    highlight_label_id=highlight_label_id,
    # Patch extraction Options
    centered_on=centered_on,
    padding=padding,
)
gpm.plot_labels(patch_gen, label_name=label_name)

# Plot patches (around each label)
patch_gen = da_precip.ximage.label_patches(
    label_name=label_name,
    patch_size=patch_size,
    # Output options
    n_patches=n_patches,
    labels_id=labels_id,
    highlight_label_id=highlight_label_id,
    # Patch extraction Options
    centered_on=centered_on,
    padding=padding,
)

gpm.plot_patches(patch_gen, variable=variable, interpolation="bilinear")

##----------------------------------------------------------------------------.
#### Retrieve list of patches
patch_gen = da_precip.ximage.label_patches(
    label_name=label_name,
    patch_size=patch_size,
    # Output options
    labels_id=labels_id,
    highlight_label_id=highlight_label_id,
    # Patch extraction Options
    centered_on=centered_on,
    padding=padding,
)

patch_idx = 0
list_label_patches = list(patch_gen)
label_id, da_patch = list_label_patches[patch_idx]

##----------------------------------------------------------------------------.
#### Retrieve isel dictionary of patch slices
patches_isel_dicts = da_precip.ximage.label_patches_isel_dicts(
    label_name=label_name,
    patch_size=patch_size,
    # Output options
    labels_id=labels_id,
    # Patch extraction Options
    centered_on=centered_on,
    padding=padding,
)

pprint(patches_isel_dicts)
label_id = 1
label_patches_isel_dicts = patches_isel_dicts[label_id]
label_patch_isel_dict = patches_isel_dicts[label_id][0]
da_patch = da_precip.isel(label_patch_isel_dict)

##----------------------------------------------------------------------------.
#### Plot mesh and centroids
da_patch1 = da_patch.isel(along_track=slice(0, 20), cross_track=slice(0, 20))
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
p = da_patch1.gpm.plot_map_mesh(ax=ax, add_background=True)
p = da_patch1.gpm.plot_map_mesh_centroids(ax=ax, add_background=False)
bg_img = ax.stock_img()
bg_img.set_alpha(0.5)
p.axes.set_extent(da_patch1.gpm.extent(padding=0.1))

##----------------------------------------------------------------------------.
#### Plot on PlateCarree()
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
p = da_patch.plot.pcolormesh(
    x="lon",
    y="lat",
    ax=ax,
    cmap="Spectral_r",
    norm=LogNorm(vmin=0.1, vmax=300),
    infer_intervals=False,
)
# - Add coastlines
ax.coastlines()

# - Add gridlines
gl = ax.gridlines(draw_labels=True)
gl.top_labels = False
gl.right_labels = False

# - Add swath
da_patch.gpm.plot_swath_lines(ax=ax)

# - Add stock img with transparency
bg_img = ax.stock_img()
bg_img.set_alpha(0.5)

# - Extend the plot extent
extent = da_patch.gpm.extent(padding=1)
p.axes.set_extent(extent)

##----------------------------------------------------------------------------.
##### Count occurrence
da1 = da_precip.where(da_precip.values > 100)
da1.count()
np.invert(np.isnan(da1.values.flatten())).sum()

##----------------------------------------------------------------------------.
### Plot distribution
da_precip.where(da_precip.values > 1).plot.hist(xlim=(1, 100), bins=100)

##----------------------------------------------------------------------------.
