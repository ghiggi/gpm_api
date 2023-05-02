#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 13:52:46 2022

@author: ghiggi
"""
import datetime
import os

import cartopy
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xp
from dask.diagnostics import ProgressBar

import gpm_api

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
#     gpm_api.download(base_dir=base_dir,
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
    ds = gpm_api.open_dataset(
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

# Create figure
for product, variables in plot_product_var_dict.items():
    for variable in variables:
        ds = dict_product[product]
        # Crop dataset
        ds = ds.gpm_api.crop(bbox)

        # Retrieve DataArray
        da = ds[variable]

        # Plot map
        p = da.gpm_api.plot_map(add_colorbar=True)

        # Set title
        title = da.gpm_api.title(time_idx=0)
        p.axes.set_title(title)

        # Set extent
        # ax.set_extent(bbox_extent)

        # Show the plot
        plt.show()

# -----------------------------------------------------------------------------.
