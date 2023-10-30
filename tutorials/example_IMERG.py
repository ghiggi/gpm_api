#!/usr/bin/env python3
"""
Created on Sat Dec 10 14:16:00 2022

@author: ghiggi
"""
import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import gpm_api
from gpm_api.utils.countries import get_country_extent
from gpm_api.utils.utils_cmap import get_colorbar_settings
from gpm_api.visualization.plot import plot_cartopy_background

start_time = datetime.datetime.strptime("2011-06-13 11:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2011-06-13 13:00:00", "%Y-%m-%d %H:%M:%S")
product = "IMERG-FR"  # 'IMERG-ER' 'IMERG-LR'
product_type = "RS"
version = 6

# Download the data
gpm_api.download(
    product=product,
    product_type=product_type,
    version=version,
    start_time=start_time,
    end_time=end_time,
    force_download=False,
    verbose=True,
    progress_bar=True,
    check_integrity=False,
)

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

print("Available variables: ", variables)
# Available coordinates
coords = list(ds.coords)
print("Available coordinates: ", coords)

# Available dimensions
dims = list(ds.dims)
print("Available dimensions: ", dims)

# Select variable
variable = "precipitationCal"

#### GPM-API methods
# xr.Dataset methods provided by gpm_api
print(dir(ds.gpm_api))

# xr.DataArray methods provided by gpm_api
print(dir(ds[variable].gpm_api))

#### Check geometry and dimensions
ds.gpm_api.is_grid  # True
ds.gpm_api.is_orbit  # False

ds.gpm_api.is_spatial_2d  # False, because of multiple timesteps
ds.isel(time=[0]).gpm_api.is_spatial_2d  # True,  because of a single timesteps
ds.isel(time=0).gpm_api.is_spatial_2d  # True,  because no time dimension anymore

ds.gpm_api.has_regular_time
ds.gpm_api.get_slices_regular_time()  # List of time slices with regular timesteps

ds.gpm_api.is_regular

ds.gpm_api.pyresample_area  # TODO: not yet implemented

####--------------------------------------------------------------------------.
#### Plotting with gpm_api a single timestep
ds[variable].isel(time=0).gpm_api.plot_map()  # With cartopy
ds[variable].isel(time=0).gpm_api.plot_image()  # Without cartopy

#### Plotting with gpm_api multiple timestep or multiple variables
ds[variable].isel(time=slice(0, 4)).gpm_api.plot_map(col="time", col_wrap=2)
ds[variable].isel(time=slice(0, 4)).gpm_api.plot_image(col="time", col_wrap=2)

####--------------------------------------------------------------------------.
#### Title
ds.gpm_api.title(add_timestep=True)
ds.isel(time=0).gpm_api.title(add_timestep=True)

ds[variable].gpm_api.title(add_timestep=False)
ds[variable].gpm_api.title(add_timestep=True)
ds[variable].isel(time=0).gpm_api.title(add_timestep=True)

####--------------------------------------------------------------------------.
#### Zoom on a geographic area
title = ds.gpm_api.title(add_timestep=False)
extent = get_country_extent("United States")
da = ds[variable].isel(time=0)
p = da.gpm_api.plot_map()
p.axes.set_extent(extent)
p.axes.set_title(label=title)

####--------------------------------------------------------------------------.
#### Customize geographic maps
#### - Customize the projection
# --> See list at https://scitools.org.uk/cartopy/docs/latest/reference/projections.html?highlight=projections
dpi = 100
figsize = (12, 10)
crs_proj = ccrs.InterruptedGoodeHomolosine()
crs_proj = ccrs.Mollweide()
crs_proj = ccrs.Robinson()
da = ds[variable].isel(time=0)

fig, ax = plt.subplots(subplot_kw={"projection": crs_proj}, figsize=figsize, dpi=dpi)
plot_cartopy_background(ax)
da.gpm_api.plot_map(ax=ax)
# ---------------------------------------------------------------------
#### - Customize the colormap
# - In the classic way
da.gpm_api.plot_map(cmap="Spectral", norm=None, vmin=0.1, vmax=100)

# - Using gpm_api pre-implemented colormap and colorbar settings
plot_kwargs, cbar_kwargs = get_colorbar_settings("IMERG_Solid")
da.gpm_api.plot_map(cbar_kwargs=cbar_kwargs, **plot_kwargs)

plot_kwargs, cbar_kwargs = get_colorbar_settings("IMERG_Liquid")
da.gpm_api.plot_map(cbar_kwargs=cbar_kwargs, **plot_kwargs)

# ---------------------------------------------------------------------
#### - Create GPM IMERG Solid + Liquid precipitation map
ds_single_timestep = ds.isel(time=0)
da_is_liquid = ds_single_timestep["probabilityLiquidPrecipitation"] > 90
da_precip = ds_single_timestep[variable]
da_liquid = da_precip.where(da_is_liquid, 0)
da_solid = da_precip.where(~da_is_liquid, 0)

plot_kwargs, cbar_kwargs = get_colorbar_settings("IMERG_Liquid")
p = da_liquid.gpm_api.plot_map(cbar_kwargs=cbar_kwargs, **plot_kwargs, add_colorbar=False)
plot_kwargs, cbar_kwargs = get_colorbar_settings("IMERG_Solid")
p = da_solid.gpm_api.plot_map(ax=p.axes, cbar_kwargs=cbar_kwargs, **plot_kwargs, add_colorbar=False)
p.axes.set_title(label=da_solid.gpm_api.title())

####--------------------------------------------------------------------------.
#### Crop the dataset
# Crop by extent
extent = get_country_extent("United States")
ds_ch = ds.gpm_api.crop(extent=extent)
ds_ch[variable].isel(time=0).gpm_api.plot_map()

# Crop by country name
ds_ch = ds.gpm_api.crop_by_country("Switzerland")
ds_ch[variable].isel(time=0).gpm_api.plot_map()

# Crop by country name
ds_continent = ds.gpm_api.crop_by_continent("Australia")
ds_continent[variable].isel(time=0).gpm_api.plot_map()

####--------------------------------------------------------------------------.
