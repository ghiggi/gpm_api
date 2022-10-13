#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:58:16 2022

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
matplotlib.rcParams['axes.facecolor'] = [0.9,0.9,0.9]
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 11
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['legend.facecolor'] = 'w'
matplotlib.rcParams['savefig.transparent'] = False

# Settings for gpm_api  
base_dir = "/home/ghiggi"
username = "gionata.ghiggi@epfl.ch"

####--------------------------------------------------------------------------.
# Define analysis time period 
start_time = datetime.datetime.strptime("2020-10-28 08:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2020-10-28 09:00:00", '%Y-%m-%d %H:%M:%S')

# Define products to analyze 
products = ['2A-DPR', "2A-GMI", '2B-GPM-CORRA', '2B-GPM-CSH', '2A-GPM-SLH', 
            '2A-ENV-DPR', '1C-GMI']
           
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
#### Define product-variable dictionary 
product_var_dict = {'2A-DPR': ["airTemperature",
                               "precipRate","paramDSD", 
                               "zFactorFinal", "zFactorMeasured", 
                               "precipRateNearSurface","precipRateESurface","precipRateESurface2",
                               "zFactorFinalESurface","zFactorFinalNearSurface",
                               "heightZeroDeg", "binEchoBottom", "landSurfaceType"],
                    "2A-GMI": ["rainWaterPath", "surfacePrecipitation", 
                               "cloudWaterPath", "iceWaterPath"],
                    '2B-GPM-CORRA': ["precipTotRate", "precipTotWaterCont",
                                     "cloudIceWaterCont", "cloudLiqWaterCont", 
                                     "nearSurfPrecipTotRate", "estimSurfPrecipTotRate",
                                     # "OEestimSurfPrecipTotRate", "OEsimulatedBrightTemp",
                                     # "OEcolumnCloudLiqWater", "OEcloudLiqWaterCont", "OEcolumnWaterVapor"],
                                     # lowestClutterFreeBin, surfaceElevation
                                    ],
                    '2B-GPM-CSH': ["latentHeating", "surfacePrecipRate"],
                    '2A-GPM-SLH': ["latentHeating", "nearSurfacePrecipRate"],
                    '2A-ENV-DPR': ["cloudLiquidWater", "waterVapor", "airPressure"],
                    } 
                                     

dict_product = {}
# product, variables = list(product_var_dict.items())[0]
for product, variables in product_var_dict.items():
    ds = open_dataset(base_dir=base_dir,
                     product=product, 
                     start_time=start_time,
                     end_time=end_time,
                     # Optional 
                     variables=variables,   
                     version=version,
                     product_type=product_type,
                     prefix_group = False)
    dict_product[product] = ds   
    
#-----------------------------------------------------------------------------.
#### Define bounding box of interest 
bbox = [-110, -70, 18, 32]
bbox_extent = [-100,-85, 18, 32]

#-----------------------------------------------------------------------------.
#### Retrieve datasets
ds_gmi = dict_product["2A-GMI"]
ds_dpr = dict_product["2A-DPR"]
ds_csh = dict_product["2B-GPM-CSH"]
ds_slh = dict_product["2A-GPM-SLH"]
# ds_corra = dict_product["2B-GPM-CORRA"]  # TODO: need to copy height from 2A-DPR (scan_mode='HS')
# ds_env = dict_product["2A-ENV-DPR"]      # TODO: need to copy height from 2A-DPR height   
 
ds_latent = ds_csh
ds_latent['CSH_latentHeating'] = ds_latent['latentHeating'] # 80 
ds_latent['SLH_latentHeating'] = ds_slh['latentHeating'] # 80 

#-----------------------------------------------------------------------------.
#### Crop dataset 
ds_dpr = ds_dpr.gpm_api.crop(bbox)
ds_latent = ds_latent.gpm_api.crop(bbox)
ds_gmi = ds_gmi.gpm_api.crop(bbox)
#-----------------------------------------------------------------------------.
#### Compute DFR 
# - zFactorMeasured # Raw
# - zFactorFinal    # Corrected
ds_dpr['dfrMeasured'] = ds_dpr['zFactorMeasured'].sel(frequency="Ku") - ds_dpr['zFactorMeasured'].sel(frequency="Ka")
ds_dpr['dfrFinal'] = ds_dpr['zFactorFinal'].sel(frequency="Ku") - ds_dpr['zFactorFinal'].sel(frequency="Ka")
ds_dpr['dfrFinalNearSurface '] = ds_dpr['zFactorFinalNearSurface'].sel(frequency="Ku") - ds_dpr['zFactorFinalNearSurface'].sel(frequency="Ka")
        
#-----------------------------------------------------------------------------.
#### Extract profile along transect     
transect_kwargs = {}
transect_kwargs = {"trim_threshold": 1,
                   "left_pad": 0,
                   "right_pad": 0,
                  }

variable = "precipRate"
direction = "cross_track"   # "along_track"

 
transect_slices = get_transect_slices(ds_dpr, 
                                      direction=direction, 
                                      variable=variable, 
                                      transect_kwargs=transect_kwargs)
ds_dpr_profile = ds_dpr.isel(transect_slices)
ds_latent_profile = ds_latent.isel(transect_slices)

ds_dpr_profile = xr_exclude_variables_without(ds_dpr_profile, dim="range")
ds_dpr_profile = ds_dpr_profile.compute()

ds_latent_profile = xr_exclude_variables_without(ds_latent_profile, dim="height")
ds_latent_profile = ds_latent_profile.compute()

np.testing.assert_equal(ds_dpr_profile['lon'].data, ds_latent_profile['lon'].data)
np.testing.assert_equal(ds_dpr_profile['lat'].data, ds_latent_profile['lat'].data)

#-----------------------------------------------------------------------------.
#### Develop profiles 

# da_profile = ds_dpr_profile["precipRate"]
# da_profile.name = "Precipitation Intensity"
# plot_profile(da_profile, colorscale="pysteps_mm/hr", ylim=(0,15000))
# plt.show() 

# da_profile = ds_dpr_profile["zFactorFinal"].sel(frequency="Ku")
# da_profile.name = "Ku-band Z Corrected"
# plot_profile(da_profile, colorscale="GPM_Z", ylim=(0,15000))
# plt.show() 

# da_profile = ds_dpr_profile["zFactorMeasured"].sel(frequency="Ku")
# da_profile.name = "Ku-band Z Measured"
# plot_profile(da_profile, colorscale="GPM_Z", ylim=(0,15000))
# plt.show() 

# da_profile = ds_dpr_profile["zFactorFinal"].sel(frequency="Ka")
# da_profile.name = "Ka-band Z Corrected"
# plot_profile(da_profile, colorscale="GPM_Z", ylim=(0,15000))
# plt.show() 

# da_profile = ds_dpr_profile["zFactorMeasured"].sel(frequency="Ka")
# da_profile.name = "Ka-band Z Measured"
# plot_profile(da_profile, colorscale="GPM_Z", ylim=(0,15000))
# plt.show() 

# da_profile = ds_dpr_profile["dfrFinal"]
# da_profile.name = "DFR Corrected"
# plot_profile(da_profile, colorscale="GPM_DFR", ylim=(0,15000))
# plt.show() 

# da_profile = ds_dpr_profile["dfrMeasured"]
# da_profile.name = "DFR Measured"
# plot_profile(da_profile, colorscale="GPM_DFR", ylim=(0,15000))
# plt.show() 

# da_profile = ds_dpr_profile["paramDSD"].sel(DSD_params="Nw")/10
# da_profile.name = "Nw"
# plot_profile(da_profile, colorscale="GPM_Nw", ylim=(0,15000))
# plt.show() 

# da_profile = ds_dpr_profile["paramDSD"].sel(DSD_params="Dm")
# da_profile.name = "Dm"
# plot_profile(da_profile, colorscale="GPM_Dm", ylim=(0,15000))
# plt.show() 


# da_profile = ds_latent_profile["CSH_latentHeating"]
# da_profile.name = "CSH Latent Heating"
# plot_profile(da_profile, colorscale="GPM_LatentHeating", ylim=(0,15000))
# plt.show() 


# da_profile = ds_latent_profile["SLH_latentHeating"]
# da_profile.name = "SLH Latent Heating"
# plot_profile(da_profile, colorscale="GPM_LatentHeating", ylim=(0,15000))
# plt.show() 

# da_profile = ds_dpr_profile["airTemperature"] - 273.15
# plot_profile(da_profile, colorscale="RdBu_r", ylim=(0,15000))
# plt.show() 

#-----------------------------------------------------------------------------.
# TODO: Mask fields 

# mask by self.dpr.ds.zFactorFinal[:,:,:,1] >= 15
# mask by self.dpr.ds.zFactorFinal[:,:,:,0] >= 10

# da_Ku_mask = ds["zFactorFinal"].sel(frequency="Ku") <= 10
# da_Ka_mask = ds["zFactorFinal"].sel(frequency="Ka") <= 15
# da_dfr_mask = ds["dfrFinal"] < -0.5 

#-----------------------------------------------------------------------------.
#### Create profile plot [Z,DFR,R,LH]
# 1 cm = 0.394 inches 
# A4 = (8.27, 11.7)
# A4 with 1.5 cm border on each side = (7, 10.5)
# border = 1.5*0.394
# 8.27 - border*2, 11.7 - border*2

ylim = (0, 14000) 

fig, axs = plt.subplots(3, 2, figsize=(7.2, 10), dpi=300)
fig.set_facecolor('w')

# Z Ku
da_profile = ds_dpr_profile["zFactorFinal"].sel(frequency="Ku")
da_profile.name = "Ku-band Z Corrected"
p = plot_profile(da_profile, colorscale="GPM_Z", ax=axs[0,0], 
                 ylim=ylim)

# Z Ka
da_profile = ds_dpr_profile["zFactorFinal"].sel(frequency="Ka")
da_profile.name = "Ka-band Z Corrected"
p = plot_profile(da_profile, colorscale="GPM_Z", ax=axs[1,0],
                 ylim=ylim)

# DFR
da_profile = ds_dpr_profile["dfrFinal"]
da_profile.name = "DFR Corrected"
p = plot_profile(da_profile, colorscale="GPM_DFR", ax=axs[2,0],
                 ylim=ylim)

# Precip 
da_profile = ds_dpr_profile["precipRate"]
da_profile.name = "Precipitation Intensity"
p = plot_profile(da_profile, colorscale="pysteps_mm/hr", ax=axs[0, 1], 
                 ylim=ylim)

# CSH
da_profile = ds_latent_profile["CSH_latentHeating"]
da_profile.name = "CSH Latent Heating"
p = plot_profile(da_profile, colorscale="GPM_LatentHeating", ax=axs[1, 1], 
                 ylim=ylim)
 
# SLH 
da_profile = ds_latent_profile["SLH_latentHeating"]
da_profile.name = "SLH Latent Heating"
p = plot_profile(da_profile, colorscale="GPM_LatentHeating", ax=axs[2, 1], 
                 ylim=ylim)

# Remove ylabel and yticks on right side plots 
for i in range(0,3):
    axs[i,1].set_yticks([])
    axs[i,1].set_ylabel(None)    
    
# Remove xlabel and xticks execpt in bottom plots 
for i in range(0,2):
    for j in range(0,2):
        axs[i,j].set_xticks([])
        axs[i,j].set_xlabel(None)

fig.tight_layout()
plt.show()


####--------------------------------------------------------------------------.
#### Create profile plot [Z,DFR ... measured and corrected]
# 1 cm = 0.394 inches 
# A4 = (8.27, 11.7)
# A4 with 1.5 cm border on each side = (7, 10.5)
# border = 1.5*0.394
# 8.27 - border*2, 11.7 - border*2

ylim = (0, 14000) 

fig, axs = plt.subplots(3, 2, figsize=(7.2, 10), dpi=300,
                        gridspec_kw={'width_ratios': [0.46, 0.54]})
fig.set_facecolor('w')

# Z Ku
da_profile = ds_dpr_profile["zFactorFinal"].sel(frequency="Ku")
da_profile.name = "Ku-band Z Corrected"
p = plot_profile(da_profile, colorscale="GPM_Z", ax=axs[0,0], 
                 ylim=ylim)
p.colorbar.remove()

# Z Ka
da_profile = ds_dpr_profile["zFactorFinal"].sel(frequency="Ka")
da_profile.name = "Ka-band Z Corrected"
p = plot_profile(da_profile, colorscale="GPM_Z", ax=axs[1,0],
                 ylim=ylim)
p.colorbar.remove()

# DFR
da_profile = ds_dpr_profile["dfrFinal"]
da_profile.name = "DFR Corrected"
p = plot_profile(da_profile, colorscale="GPM_DFR", ax=axs[2,0],
                 ylim=ylim)
p.colorbar.remove()

# Ku measured  
da_profile = ds_dpr_profile["zFactorMeasured"].sel(frequency="Ku")
da_profile.name = "Ku-band Z Measured"
p = plot_profile(da_profile, colorscale="GPM_Z", ax=axs[0, 1], 
                 ylim=ylim)

# Ka measured 
da_profile = ds_dpr_profile["zFactorMeasured"].sel(frequency="Ka")
da_profile.name = "Ka-band Z Measured"
p = plot_profile(da_profile, colorscale="GPM_Z", ax=axs[1, 1], 
             ylim=ylim)
 
# DFR measured 
da_profile = ds_dpr_profile["dfrMeasured"]
da_profile.name = "DFR Measured"
p = plot_profile(da_profile, colorscale="GPM_DFR", ax=axs[2, 1], 
                ylim=ylim)

# Remove ylabel and yticks on right side plots 
for i in range(0,3):
    axs[i,1].set_yticks([])
    axs[i,1].set_ylabel(None)    
    
# Remove xlabel and xticks execpt in bottom plots 
for i in range(0,2):
    for j in range(0,2):
        axs[i,j].set_xticks([])
        axs[i,j].set_xlabel(None)

fig.tight_layout()
plt.show()

####--------------------------------------------------------------------------.
##### Create 2D map showing the profile
# Retrieve DataArray
variable = "precipRateESurface"
da = ds_dpr[variable]

# Define figure settings
dpi = 100
figsize = (12,10)
crs_proj = ccrs.PlateCarree()

# Create figure 
fig, ax = plt.subplots(subplot_kw={'projection': crs_proj}, figsize=figsize, dpi=dpi)

# - Plot map 
p = da.gpm_api.plot(ax=ax, add_colorbar=True)

# - Add transect line 
ds_dpr_profile.gpm_api.plot_transect_line(ax=ax, color="black")  

# - Set title 
title = da.gpm_api.title(time_idx=0)
ax.set_title(title)

# - Set extent 
ax.set_extent(bbox_extent)

plt.show()

####--------------------------------------------------------------------------.
##### Create 2D map comparing GPM DPR and GPM GMI (and adding transect)
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
fig, axs = plt.subplots(1,2, subplot_kw={'projection': crs_proj}, 
                        gridspec_kw={'width_ratios': [0.44, 0.56]},
                        figsize=figsize, dpi=dpi)

#### GPM DPR 
ax = axs[0]

# Retrieve DataArray
variable = "precipRateNearSurface"
da = ds_dpr[variable]

# - Plot map 
p = da.gpm_api.plot(ax=ax, add_colorbar=False)
 
# - Add transect line 
ds_dpr_profile.gpm_api.plot_transect_line(ax=ax, color="black")  

# - Set title 
title = da.gpm_api.title(time_idx=0, add_timestep=False)
ax.set_title(title)

# - Set extent 
ax.set_extent(bbox_extent)

#### GPM GMI 
ax = axs[1]

# Retrieve DataArray
variable = "surfacePrecipitation"
da = ds_gmi[variable]

# - Plot map 
p = da.gpm_api.plot(ax=ax, add_colorbar=True)

# TODO: REMOVE Y AXIS 
# ax.set_yticklabels(None)
#  ax.axes.get_yaxis().set_visible(False)

# - Add transect line 
ds_dpr_profile.gpm_api.plot_transect_line(ax=ax, color="black")  

# - Set title 
title = da.gpm_api.title(time_idx=0, add_timestep=False)
ax.set_title(title)

# - Set extent 
ax.set_extent(bbox_extent)

ax.get_ygridlines()
 
fig.tight_layout()

plt.show()

####--------------------------------------------------------------------------.




