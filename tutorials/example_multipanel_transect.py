#!/usr/bin/env python3
"""
Created on Wed Aug 16 11:42:15 2023

@author: ghiggi
"""
import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ximage  # noqa

import gpm_api

# Matplotlib settings
matplotlib.rcParams["axes.facecolor"] = [0.9, 0.9, 0.9]
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["legend.facecolor"] = "w"
matplotlib.rcParams["savefig.transparent"] = False


# -----------------------------------------------------------------------------.
#### Define analysis settings
# Define analysis time period
start_time = datetime.datetime.strptime("2020-10-28 08:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-10-28 09:00:00", "%Y-%m-%d %H:%M:%S")

# Define products to analyze
products = [
    "2A-DPR",
    "2B-GPM-CSH",
    "2A-GPM-SLH",
]

version = 7
product_type = "RS"

####--------------------------------------------------------------------------.
#### Download products
# for product in products:
#     print(product)
#     gpm_api.download(product=product,
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
        "precipRateNearSurface",  # to identify region of interest
        # "airTemperature",
        "precipRate",
        # "paramDSD",
        "zFactorFinal",
        "zFactorMeasured",
        "zFactorFinalESurface",
        "zFactorFinalNearSurface",
    ],
    "2B-GPM-CSH": ["latentHeating"],
    "2A-GPM-SLH": ["latentHeating"],
}

####--------------------------------------------------------------------------.
#### Open the datasets
dict_product = {}
# product, variables = list(product_var_dict.items())[0]
for product, variables in product_var_dict.items():
    ds = gpm_api.open_dataset(
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

ds_dpr = dict_product["2A-DPR"]
ds_csh = dict_product["2B-GPM-CSH"]  # height: 80
ds_slh = dict_product["2A-GPM-SLH"]  # height: 80

####--------------------------------------------------------------------------.
#### Identify region of interest
# Identify precipitating areas
label_name = "label_precip_area"
ds_dpr = ds_dpr.ximage.label(
    variable="precipRateNearSurface",
    min_value_threshold=1,
    min_area_threshold=10,
    footprint=10,
    label_name=label_name,
    sort_by="area",
)
gpm_api.plot_labels(ds_dpr["label_precip_area"])

# Have a look at such areas
da_patch_gen = ds_dpr.ximage.label_patches(
    label_name=label_name,
    patch_size={"cross_track": -1, "along_track": 100},
    centered_on="label_bbox",
    variable="precipRateNearSurface",
)
gpm_api.plot_patches(da_patch_gen, variable="precipRateNearSurface")

# Extract isel_dict(s) for each area
list_isel_dicts = ds_dpr.ximage.label_patches_isel_dicts(
    label_name=label_name,
    patch_size={"cross_track": -1, "along_track": 100},
    padding=10,
    centered_on="label_bbox",
    variable="precipRateNearSurface",
)

# Select one isel_dict
label_id = 1
isel_dict = list_isel_dicts[label_id][0]

# Visualize it
ds_aoi = ds_dpr.isel(isel_dict)
ds_aoi.gpm_api.plot_map(variable="precipRateNearSurface")

####--------------------------------------------------------------------------.
#### Crop datasets to area of interest
# Retrieve extent and time period
extent = ds_aoi.gpm_api.extent(padding=0.5)
aoi_start_time = ds_aoi.gpm_api.start_time - datetime.timedelta(minutes=10)
aoi_end_time = ds_aoi.gpm_api.end_time + datetime.timedelta(minutes=10)

# Subset datasets to the time period of interest
ds_dpr = ds_dpr.gpm_api.subset_by_time_slice(slice=slice(aoi_start_time, aoi_end_time))
ds_slh = ds_slh.gpm_api.subset_by_time_slice(slice=slice(aoi_start_time, aoi_end_time))
ds_csh = ds_csh.gpm_api.subset_by_time_slice(slice=slice(aoi_start_time, aoi_end_time))

# Crop the datasets
ds_dpr = ds_dpr.gpm_api.crop(extent)
ds_slh = ds_slh.gpm_api.crop(extent)
ds_csh = ds_csh.gpm_api.crop(extent)

####--------------------------------------------------------------------------.
#### Compute DFR
# - zFactorMeasured # Raw
# - zFactorFinal    # Corrected
ds_dpr["dfrMeasured"] = ds_dpr.gpm_api.retrieve("dfrMeasured")
ds_dpr["dfrFinal"] = ds_dpr.gpm_api.retrieve("dfrFinal")
ds_dpr["dfrFinalNearSurface "] = ds_dpr.gpm_api.retrieve("dfrFinalNearSurface")

####--------------------------------------------------------------------------.
#### Extract transect

# Define transect passing across the maximum intensity region
transect_kwargs = {}
transect_kwargs = {
    "trim_threshold": 1,
    "left_pad": 0,
    "right_pad": 0,
}
variable = "precipRate"
direction = "cross_track"  # "along_track"
transect_slices = ds_dpr.gpm_api.define_transect_slices(
    direction=direction,
    variable=variable,
    transect_kwargs=transect_kwargs,
)

# Extract the transect
ds_dpr_transect = ds_dpr.gpm_api.select_spatial_3d_variables().isel(transect_slices)
ds_csh_transect = ds_csh.gpm_api.select_spatial_3d_variables().isel(transect_slices)
ds_slh_transect = ds_slh.gpm_api.select_spatial_3d_variables().isel(transect_slices)

ds_dpr_transect = ds_dpr_transect.compute()
ds_csh_transect = ds_csh_transect.compute()
ds_slh_transect = ds_slh_transect.compute()

# Check same transect geolocation between 2A-DPR, 2B-GPM-CSH and 2A-GPM-SLH
np.testing.assert_equal(ds_dpr_transect["lon"].data, ds_csh_transect["lon"].data)
np.testing.assert_equal(ds_dpr_transect["lat"].data, ds_csh_transect["lat"].data)

np.testing.assert_equal(ds_dpr_transect["lon"].data, ds_slh_transect["lon"].data)
np.testing.assert_equal(ds_dpr_transect["lat"].data, ds_slh_transect["lat"].data)

# Visualize transect position
p = ds_dpr.gpm_api.plot_map(variable="precipRateNearSurface")
ds_dpr_transect.gpm_api.plot_transect_line(ax=p.axes, color="black")

####---------------------------------------------------------------------------.
#### Create multipanel [Z,DFR,R,LH]
# 1 cm = 0.394 inches
# A4 = (8.27, 11.7)
# A4 with 1.5 cm border on each side = (7, 10.5)
# border = 1.5*0.394
# 8.27 - border*2, 11.7 - border*2

ylim = (0, 14000)
figsize = (7.2, 10)
dpi = 300

fig, axs = plt.subplots(3, 2, figsize=figsize, dpi=dpi)
fig.set_facecolor("w")

# Z Ku
da_transect = ds_dpr_transect["zFactorFinal"].sel(radar_frequency="Ku")
p = da_transect.gpm_api.plot_transect(ax=axs[0, 0])
p.axes.set_title("Ku-band Z Corrected")
p.axes.set_ylim(ylim)

# Z Ka
da_transect = ds_dpr_transect["zFactorFinal"].sel(radar_frequency="Ka")
p = da_transect.gpm_api.plot_transect(ax=axs[1, 0])
p.axes.set_title("Ka-band Z Corrected")
p.axes.set_ylim(ylim)

# DFR
da_transect = ds_dpr_transect["dfrFinal"]
p = da_transect.gpm_api.plot_transect(ax=axs[2, 0])
p.axes.set_title("DFR Corrected")
p.axes.set_ylim(ylim)

# Precip
da_transect = ds_dpr_transect["precipRate"]
p = da_transect.gpm_api.plot_transect(ax=axs[0, 1])
p.axes.set_title("Precipitation Intensity")
p.axes.set_ylim(ylim)

# CSH
da_transect = ds_csh_transect["latentHeating"]
p = da_transect.gpm_api.plot_transect(ax=axs[1, 1])
p.axes.set_title("CSH Latent Heating")
p.axes.set_ylim(ylim)

# SLH
da_transect = ds_slh_transect["latentHeating"]
p = da_transect.gpm_api.plot_transect(ax=axs[2, 1])
p.axes.set_title("SLH Latent Heating")
p.axes.set_ylim(ylim)

# Remove ylabel and yticks on right side plots
for i in range(0, 3):
    axs[i, 1].set_yticks([])
    axs[i, 1].set_ylabel(None)

# Remove xlabel and xticks except in bottom plots
for i in range(0, 2):
    for j in range(0, 2):
        axs[i, j].set_xticks([])
        axs[i, j].set_xlabel(None)

fig.tight_layout()
plt.show()


####--------------------------------------------------------------------------.
#### Create multipanel [Z & DFR: measured vs. corrected]
# 1 cm = 0.394 inches
# A4 = (8.27, 11.7)
# A4 with 1.5 cm border on each side = (7, 10.5)
# border = 1.5*0.394
# 8.27 - border*2, 11.7 - border*2

ylim = (0, 14000)

fig, axs = plt.subplots(
    3, 2, figsize=(7.2, 10), dpi=300, gridspec_kw={"width_ratios": [0.46, 0.54]}
)
fig.set_facecolor("w")

# Z Ku
da_transect = ds_dpr_transect["zFactorFinal"].sel(radar_frequency="Ku")
p = da_transect.gpm_api.plot_transect(ax=axs[0, 0])
p.axes.set_title("Ku-band Z Corrected")
p.axes.set_ylim(ylim)
p.colorbar.remove()

# Z Ka
da_transect = ds_dpr_transect["zFactorFinal"].sel(radar_frequency="Ka")
p = da_transect.gpm_api.plot_transect(ax=axs[1, 0])
p.axes.set_title("Ka-band Z Corrected")
p.axes.set_ylim(ylim)
p.colorbar.remove()

# DFR
da_transect = ds_dpr_transect["dfrFinal"]
p = da_transect.gpm_api.plot_transect(ax=axs[2, 0])
p.axes.set_title("DFR Corrected")
p.axes.set_ylim(ylim)
p.colorbar.remove()

# Ku measured
da_transect = ds_dpr_transect["zFactorMeasured"].sel(radar_frequency="Ku")
p = da_transect.gpm_api.plot_transect(ax=axs[0, 1])
p.axes.set_title("Ku-band Z Measured")
p.axes.set_ylim(ylim)


# Ka measured
da_transect = ds_dpr_transect["zFactorMeasured"].sel(radar_frequency="Ka")
p = da_transect.gpm_api.plot_transect(ax=axs[1, 1])
p.axes.set_title("Ka-band Z Measured")
p.axes.set_ylim(ylim)


# DFR measured
da_transect = ds_dpr_transect["dfrMeasured"]
p = da_transect.gpm_api.plot_transect(ax=axs[2, 1])
p.axes.set_title("DFR Measured")
p.axes.set_ylim(ylim)

# Remove ylabel and yticks on right side plots
for i in range(0, 3):
    axs[i, 1].set_yticks([])
    axs[i, 1].set_ylabel(None)

# Remove xlabel and xticks except in bottom plots
for i in range(0, 2):
    for j in range(0, 2):
        axs[i, j].set_xticks([])
        axs[i, j].set_xlabel(None)

fig.tight_layout()
plt.show()

####--------------------------------------------------------------------------.
