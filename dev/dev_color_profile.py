#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:00:40 2022

@author: ghiggi
"""
import datetime
import cartopy
import matplotlib
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from gpm_api.io_future.dataset import open_dataset
from gpm_api.utils.utils_cmap import get_colormap_setting
from gpm_api.utils.visualization import (
    get_transect_slices,
    xr_exclude_variables_without,
    plot_profile,
)

base_dir = "/home/ghiggi"
username = "gionata.ghiggi@epfl.ch"

#### Define analysis time period
start_time = datetime.datetime.strptime("2020-10-28 08:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-10-28 09:00:00", "%Y-%m-%d %H:%M:%S")

product = "2A-DPR"
variables = [
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
]

# Retrieve dataset
ds = open_dataset(
    base_dir=base_dir,
    product=product,
    start_time=start_time,
    end_time=end_time,
    # Optional
    variables=variables,
    prefix_group=False,
)


# Get transect
variable = "precipRate"
direction = "cross_track"
transect_kwargs = {
    "trim_threshold": 1,
    "left_pad": 0,
    "right_pad": 0,
}
transect_slices = get_transect_slices(
    ds, direction=direction, variable=variable, transect_kwargs=transect_kwargs
)
ds_transect = ds.isel(transect_slices)

# Keep variables with range (or height) dimension
ds_profile = xr_exclude_variables_without(ds_transect, dim="range")

da_profile = ds_profile["precipRate"]
da_profile = da_profile.compute()
x_direction = da_profile["lon"].dims[0]

# EDA
plt.hist(da_profile.data.flatten(), bins=100)

# Try good settings
p = da_profile.plot.pcolormesh(
    x=x_direction,
    y="height",
    vmin=-1,
    vmax=1,
    # norm = norm,
    cmap="RdBu_r",
)
plt.show()

# Implement and test it
plot_kwargs, cbar_kwargs, ticklabels = get_colormap_setting("pysteps_mm/hr")
plot_kwargs, cbar_kwargs, ticklabels = get_colormap_setting("GPM_LatentHeating")

p = da_profile.plot.pcolormesh(x=x_direction, y="height", cbar_kwargs=cbar_kwargs, **plot_kwargs)
plt.show()
