#!/usr/bin/env python3
"""
Created on Wed Jul 19 15:10:38 2023

@author: ghiggi
"""
import datetime

import matplotlib.pyplot as plt
import pyproj
import ximage  # noqa
from matplotlib.colors import LogNorm
from pyresample.kd_tree import resample_nearest

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
    n_threads=5,  # 8
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

# Select DataArray
variable = "precipRateNearSurface"
da_precip = ds[variable].load()


# Label the object to identify precipitation areas
label_name = "precip_label_area"
da_precip = da_precip.ximage.label(min_value_threshold=0.1, label_name=label_name)

# Define the patch generator
labels_id = None
labels_id = [1, 2, 3]
patch_size = 2  # min patch size
centered_on = "label_bbox"
padding = 0
highlight_label_id = False

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
#### Resample Swath data into bounding box optimal AreaDefinition
# Retrieve SwathDefinition
swath_def = da_patch.gpm.pyresample_area

# Define optimal AreaDefinition on SwathDefinition bounding box
# - The default is the Oblique Mercator (omerc)
area_def = swath_def.compute_optimal_bb_area()

# - On WGS84
proj_dict = pyproj.CRS.from_epsg(4326).to_dict()
area_def = swath_def.compute_optimal_bb_area(proj_dict=proj_dict)
area_def

# Resample data into  AreaDefinition
data = da_patch.data
data_proj = resample_nearest(swath_def, data, area_def, radius_of_influence=20000, fill_value=None)

# Plot
crs = area_def.to_cartopy_crs()
fig, ax = plt.subplots(subplot_kw=dict(projection=crs))

img = plt.imshow(
    data_proj,
    transform=crs,
    extent=crs.bounds,
    origin="upper",
    cmap="Spectral_r",
    norm=LogNorm(vmin=0.1, vmax=300),
)
cbar = plt.colorbar()
coastlines = ax.coastlines()
gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.5)
gl.top_labels = False  # Hide x-axis labels at the top
gl.right_labels = False  # Hide y-axis labels at the right


##----------------------------------------------------------------------------.
