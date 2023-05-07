#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:34:55 2022

@author: ghiggi
"""
import os
import datetime
import gpm_api
import cartopy
import matplotlib
import numpy as np
import xarray as xp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
from gpm_api.io import download_GPM_data
from gpm_api.io_future.dataset import open_dataset
from gpm_api.utils.utils_cmap import get_colormap_setting
from gpm_api.utils.visualization import (
    get_transect_slices,
    xr_exclude_variables_without,
    plot_profile,
)

# Matplotlib settings
matplotlib.rcParams["axes.facecolor"] = [0.9, 0.9, 0.9]
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["axes.titlesize"] = 14
matplotlib.rcParams["xtick.labelsize"] = 12
matplotlib.rcParams["ytick.labelsize"] = 12
matplotlib.rcParams["legend.fontsize"] = 12
matplotlib.rcParams["legend.facecolor"] = "w"
matplotlib.rcParams["savefig.transparent"] = False

# Settings for gpm_api
base_dir = "/home/ghiggi"
username = "gionata.ghiggi@epfl.ch"

####--------------------------------------------------------------------------.
# Define analysis time period
start_time = datetime.datetime.strptime("2022-08-18 01:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2022-08-18 03:00:00", "%Y-%m-%d %H:%M:%S")

# Define products to analyze
products = [
    "2A-DPR",
    "2A-GPM-SLH",
    "2B-GPM-CORRA",
    "2B-GPM-CSH",
    "2A-ENV-DPR",
    "2A-GMI",
    "1C-GMI",
]

version = 7
product_type = "NRT"

####--------------------------------------------------------------------------.
### Download products
for product in products:
    print(product)
    download_GPM_data(
        base_dir=base_dir,
        username=username,
        product=product,
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
        force_download=True,
        transfer_tool="curl",
        progress_bar=True,
        verbose=True,
        n_threads=5,
    )

#### Define product-variable dictionary
product_var_dict = {
    "2A-DPR": [
        "airTemperature",
        "precipRate",
        "paramDSD",
        "zFactorFinal",
        "zFactorMeasured",
        "precipRateNearSurface",
        "precipRateESurface",
        "precipRateESurface2",
        "zFactorFinalESurface",
        "zFactorFinalNearSurface",
        "heightZeroDeg",
        "binEchoBottom",
        "landSurfaceType",
    ],
    "2A-GMI": [
        "rainWaterPath",
        "surfacePrecipitation",
        "cloudWaterPath",
        "iceWaterPath",
    ],
    "2B-GPM-CORRA": [
        "precipTotRate",
        "precipTotWaterCont",
        "cloudIceWaterCont",
        "cloudLiqWaterCont",
        "nearSurfPrecipTotRate",
        "estimSurfPrecipTotRate",
        # "OEestimSurfPrecipTotRate", "OEsimulatedBrightTemp",
        # "OEcolumnCloudLiqWater", "OEcloudLiqWaterCont", "OEcolumnWaterVapor"],
        # lowestClutterFreeBin, surfaceElevation
    ],
    "2B-GPM-CSH": ["latentHeating", "surfacePrecipRate"],
    "2A-GPM-SLH": ["latentHeating", "nearSurfacePrecipRate"],
    "2A-ENV-DPR": ["cloudLiquidWater", "waterVapor", "airPressure"],
}


dict_product = {}
# product, variables = list(product_var_dict.items())[0]
for product, variables in product_var_dict.items():
    ds = open_dataset(
        base_dir=base_dir,
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

# -----------------------------------------------------------------------------.
#### Define bounding box of interest
bbox = [6, 12, 43, 48]  # [lon_min, lon_max, lat_min, lat_max]
bbox_extent = [6, 12, 43, 48]

# -----------------------------------------------------------------------------.
#### Retrieve datasets
ds_dpr = dict_product["2A-DPR"]
ds_csh = dict_product["2B-GPM-CSH"]
ds_slh = dict_product["2A-GPM-SLH"]
# ds_corra = dict_product["2B-GPM-CORRA"]  # TODO: need to copy height from 2A-DPR (scan_mode='HS')
# ds_env = dict_product["2A-ENV-DPR"]      # TODO: need to copy height from 2A-DPR height

ds_latent = ds_csh
ds_latent["CSH_latentHeating"] = ds_latent["latentHeating"]  # 80
ds_latent["SLH_latentHeating"] = ds_slh["latentHeating"]  # 80

# -----------------------------------------------------------------------------.
#### Crop dataset
ds_dpr = ds_dpr.gpm_api.crop(bbox)
ds_latent = ds_latent.gpm_api.crop(bbox)

# -----------------------------------------------------------------------------.
#### Compute DFR
# - zFactorMeasured # Raw
# - zFactorFinal    # Corrected
ds_dpr["dfrMeasured"] = ds_dpr["zFactorMeasured"].sel(frequency="Ku") - ds_dpr[
    "zFactorMeasured"
].sel(frequency="Ka")
ds_dpr["dfrFinal"] = ds_dpr["zFactorFinal"].sel(frequency="Ku") - ds_dpr["zFactorFinal"].sel(
    frequency="Ka"
)
ds_dpr["dfrFinalNearSurface "] = ds_dpr["zFactorFinalNearSurface"].sel(frequency="Ku") - ds_dpr[
    "zFactorFinalNearSurface"
].sel(frequency="Ka")

# -----------------------------------------------------------------------------.
#### Extract profile along transect
transect_kwargs = {}
transect_kwargs = {
    "trim_threshold": 1,
    "left_pad": 0,
    "right_pad": 0,
}

variable = "precipRate"
direction = "cross_track"  # "along_track"


transect_slices = get_transect_slices(
    ds_dpr, direction=direction, variable=variable, transect_kwargs=transect_kwargs
)
ds_dpr_profile = ds_dpr.isel(transect_slices)
ds_latent_profile = ds_latent.isel(transect_slices)

ds_dpr_profile = xr_exclude_variables_without(ds_dpr_profile, dim="range")
ds_dpr_profile = ds_dpr_profile.compute()

ds_latent_profile = xr_exclude_variables_without(ds_latent_profile, dim="height")
ds_latent_profile = ds_latent_profile.compute()

np.testing.assert_equal(ds_dpr_profile["lon"].data, ds_latent_profile["lon"].data)
np.testing.assert_equal(ds_dpr_profile["lat"].data, ds_latent_profile["lat"].data)

# -----------------------------------------------------------------------------.
