#!/usr/bin/env python3
"""
Created on Wed Aug 16 13:35:20 2023

@author: ghiggi
"""
import datetime

import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt

import gpm

# Matplotlib settings
matplotlib.rcParams["axes.facecolor"] = [0.9, 0.9, 0.9]
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["legend.facecolor"] = "w"
matplotlib.rcParams["savefig.transparent"] = False


####--------------------------------------------------------------------------.
#### Define analysis settings
# Define analysis time period
start_time = datetime.datetime.strptime("2020-10-28 08:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-10-28 09:00:00", "%Y-%m-%d %H:%M:%S")

# Define products to analyze
products = [
    "2A-DPR",
    "2A-GMI",
]

version = 7
product_type = "RS"

####--------------------------------------------------------------------------.
#### Download products
# for product in products:
#     print(product)
#     gpm.download(product=product,
#                       product_type=product_type,
#                       version = version,
#                       start_time=start_time,
#                       end_time=end_time,
#                       force_download=False,
#                       transfer_tool="curl",
#                       progress_bar=True,
#                       verbose = True,
#                       n_threads=1)

####--------------------------------------------------------------------------.
#### Define product-variable dictionary
product_var_dict = {
    "2A-DPR": [
        "precipRate",
        "zFactorFinal",
        "precipRateNearSurface",
        "zFactorFinalNearSurface",
    ],
    "2A-GMI": [
        "surfacePrecipitation",
    ],
}

####--------------------------------------------------------------------------.
#### Open the datasets
dict_product = {}
# product, variables = list(product_var_dict.items())[0]
for product, variables in product_var_dict.items():
    ds = gpm.open_dataset(
        product=product,
        start_time=start_time,
        end_time=end_time,
        # Optional
        variables=variables,
        version=version,
        product_type=product_type,
        prefix_group=False,
    )
    dict_product[product] = ds


####--------------------------------------------------------------------------.
#### Retrieve datasets over AOI
# Define bounding box of interest
bbox = [-110, -70, 18, 32]
bbox_extent = [-94, -89, 22.5, 27.5]

# Crop dataset
ds_dpr = dict_product["2A-DPR"].gpm.crop(bbox)
ds_gmi = dict_product["2A-GMI"].gpm.crop(bbox)

####--------------------------------------------------------------------------.
#### Extract transect passing across the maximum intensity region
transect_kwargs = {}
transect_kwargs = {
    "trim_threshold": 1,
    "left_pad": 0,
    "right_pad": 0,
}

variable = "precipRate"
direction = "cross_track"  # "along_track"

ds_dpr_transect = ds_dpr.gpm.select_transect(
    direction=direction, variable=variable, transect_kwargs=transect_kwargs
)
ds_dpr_transect = ds_dpr_transect.compute()

####--------------------------------------------------------------------------.
#### Create 2D map showing the transect

# Define figure settings
dpi = 300
figsize = (6, 5)
crs_proj = ccrs.PlateCarree()

# Create figure
da = ds_dpr["precipRateNearSurface"]
fig, ax = plt.subplots(subplot_kw={"projection": crs_proj}, figsize=figsize, dpi=dpi)
p = da.gpm.plot_map(ax=ax, add_colorbar=True)
ds_dpr_transect.gpm.plot_transect_line(ax=ax, color="black")
title = da.gpm.title(time_idx=0)
ax.set_title(title)
ax.set_extent(bbox_extent)
plt.show()

####--------------------------------------------------------------------------.
#### Create 2D map comparing GPM DPR and GPM GMI (and adding transect)
# 1 cm = 0.394 inches
# A4 = (8.27, 11.7)
# A4 with 1.5 cm border on each side = (7, 10.5)
# border = 1.5*0.394
# 8.27 - border*2, 11.7 - border*2

# Define figure settings
dpi = 300
figsize = (7, 2.8)
crs_proj = ccrs.PlateCarree()

# Create figure
fig, axs = plt.subplots(
    1,
    2,
    subplot_kw={"projection": crs_proj},
    gridspec_kw={"width_ratios": [0.48, 0.52]},
    figsize=figsize,
    dpi=dpi,
)

# Plot GPM DPR
ax = axs[0]

da = ds_dpr["precipRateNearSurface"]
p = da.gpm.plot_map(ax=ax, add_colorbar=False)
ds_dpr_transect.gpm.plot_transect_line(ax=ax, color="black")
title = da.gpm.title(add_timestep=False)
ax.set_title(title)
ax.set_extent(bbox_extent)

# Plot GPM GMI
ax = axs[1]
da = ds_gmi["surfacePrecipitation"]
p = da.gpm.plot_map(ax=ax, add_colorbar=True)
ds_dpr_transect.gpm.plot_transect_line(
    ax=ax, color="black", add_direction=False, line_kwargs={"linestyle": "-", "alpha": 0.6}
)
ds_dpr.gpm.plot_swath_lines(ax=ax, alpha=0.4)
title = da.gpm.title(add_timestep=False)
ax.set_title(title)
ax.set_extent(bbox_extent)
ax.yaxis.set_ticklabels("")

fig.tight_layout()

plt.show()

####--------------------------------------------------------------------------.
#### Display GPM DPR transect
# Precip
dpi = 300
figsize = (4, 2.8)
fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

da_transect = ds_dpr_transect["precipRate"]
da_transect = da_transect.where(da_transect > 0.1)
p = da_transect.gpm.plot_transect(ax=ax, zoom=True)
title = da_transect.gpm.title(add_timestep=False)
ax.set_title(title)

plt.show()

####--------------------------------------------------------------------------.
