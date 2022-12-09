#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 13:52:46 2022

@author: ghiggi
"""
import os
import gpm_api
import cartopy
import datetime
import matplotlib
import numpy as np
import xarray as xp
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar
from gpm_api.io import download_GPM_data
from gpm_api.io_future.dataset import open_dataset

####--------------------------------------------------------------------------.
#### Define matplotlib settings

matplotlib.rcParams["axes.facecolor"] = [0.9, 0.9, 0.9]
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["legend.facecolor"] = "w"
matplotlib.rcParams["savefig.transparent"] = False

####--------------------------------------------------------------------------.
#### Define GPM settings
base_dir = "/home/ghiggi"
username = "gionata.ghiggi@epfl.ch"

####--------------------------------------------------------------------------.
#### Define analysis time period
start_time = datetime.datetime.strptime("2016-03-09 10:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2016-03-09 11:00:00", "%Y-%m-%d %H:%M:%S")

products = [
    "2A-DPR",
    "2A-GMI",
    "2B-GPM-CORRA",
    # '2B-GPM-CSH', '2A-GPM-SLH',
    # '2A-ENV-DPR', '1C-GMI'
]

version = 7
product_type = "RS"

####--------------------------------------------------------------------------.
#### Download products
# for product in products:
#     print(product)
#     download_GPM_data(base_dir=base_dir,
#                       username=username,
#                       product=product,
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
#### Read datasets
groups = None
variables = None
scan_mode = None
decode_cf = False
chunks = "auto"
prefix_group = False

product_var_dict = {
    "2A-DPR": [
        "airTemperature",
        "heightZeroDeg",
        "precipRate",
        "precipRateNearSurface",
        "precipRateESurface",
        "precipRateESurface2",
        "zFactorFinalESurface",
        "zFactorFinalNearSurface",
        "zFactorFinal",
        "binEchoBottom",
        "landSurfaceType",
    ],
    "2A-GMI": [
        "rainWaterPath",
        "surfacePrecipitation",
        "cloudWaterPath",
        "iceWaterPath",
    ],
    # '2B-GPM-CORRA': ["precipTotRate", "precipTotWaterCont",
    #                  "cloudIceWaterCont", "cloudLiqWaterCont",
    #                  "nearSurfPrecipTotRate", "estimSurfPrecipTotRate",
    #                  # "OEestimSurfPrecipTotRate", "OEsimulatedBrightTemp",
    #                  # "OEcolumnCloudLiqWater", "OEcloudLiqWaterCont", "OEcolumnWaterVapor"],
    #                  # lowestClutterFreeBin, surfaceElevation
    #                 ],
    # '2B-GPM-CSH': ["latentHeating", "surfacePrecipRate"],
    # '2A-GPM-SLH': ["latentHeating", "nearSurfacePrecipRate"],
    # '2A-ENV-DPR': ["cloudLiquidWater", "waterVapor", "airPressure"],
    # '1C-GMI': ["Tc", "Quality"]
}


dict_product = {}
# product, variables = list(product_var_dict.items())[0]
# product, variables = list(product_var_dict.items())[2]
for product, variables in product_var_dict.items():
    ds = open_dataset(
        base_dir=base_dir,
        product=product,
        start_time=start_time,
        end_time=end_time,
        # Optional
        variables=variables,
        groups=groups,
        scan_mode=scan_mode,
        version=version,
        product_type=product_type,
        chunks="auto",
        decode_cf=True,
        prefix_group=False,
    )
    dict_product[product] = ds

####--------------------------------------------------------------------------.
#### Plot Datasets
plot_product_var_dict = {
    "2A-DPR": [
        "precipRateNearSurface",
        # "precipRateESurface",
        # "precipRateESurface2"
    ],
    # '2B-GPM-CORRA': ["nearSurfPrecipTotRate",
    #                   # "estimSurfPrecipTotRate"
    #                   ],
    "2A-GMI": ["surfacePrecipitation"],
}

product = "2A-DPR"
variable = "precipRateNearSurface"
bbox = [-104, -90, 26, 36]
bbox_extent = [-102, -92, 28, 35]

# Define figure settings
dpi = 100
figsize = (12, 10)
crs_proj = ccrs.PlateCarree()

# Create figure
for product, variables in plot_product_var_dict.items():
    for variable in variables:

        fig, ax = plt.subplots(
            subplot_kw={"projection": crs_proj}, figsize=figsize, dpi=dpi
        )

        ds = dict_product[product]
        # Crop dataset
        ds_subset = ds.gpm_api.crop(bbox)

        # Retrieve DataArray
        da_subset = ds_subset[variable]

        # Plot map
        p = da_subset.gpm_api.plot(ax=ax, add_colorbar=True)

        # Set title
        title = da_subset.gpm_api.title(time_idx=0)
        ax.set_title(title)

        # Set extent
        ax.set_extent(bbox_extent)

        # Show the plot
        plt.show()

# -----------------------------------------------------------------------------.
