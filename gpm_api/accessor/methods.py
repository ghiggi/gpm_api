#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 09:31:39 2022

@author: ghiggi
"""
import xarray as xr 
from gpm_api.utils.geospatial import (
    crop_dataset, 
    is_orbit, 
    is_grid, 
    get_pyresample_area,
)
    
from gpm_api.utils.visualization import (
    get_dataset_title,
    _plot_map,
    plot_transect_line,
    _plot_swath_lines,
)


@xr.register_dataset_accessor("gpm_api")
class GPM_Dataset_Accessor:
    
    def __init__(self, xarray_obj):
        if not isinstance(xarray_obj, (xr.DataArray, xr.Dataset)):
            raise TypeError("The 'gpm_api' accessor is available only for xr.Datasets and xr.DataArray.")
        self._obj = xarray_obj


    def crop(self, bbox):
        """Crop dataset."""
        return crop_dataset(self._obj, bbox)


    def plot(self, variable, ax=None, add_colorbar=True):
        # Subset variable 
        da =  self._obj[variable]
        p = _plot_map(ax=ax, da=da, add_colorbar=add_colorbar)
        return p 


    def plot_transect_line(self, ax, color="black"):
        p = plot_transect_line(self._obj, ax=ax, color=color)
        return p


    def plot_swath_lines(self, ax, **kwargs):
        p = _plot_swath_lines(self._obj, ax=ax, **kwargs)
        return p
    
    
    @property
    def pyresample_area(self):
        return get_pyresample_area(self._obj)
    
    
    @property
    def is_orbit(self):
        return is_orbit(self._obj)
    
    
    @property
    def is_grid(self):
        return is_grid(self._obj)
    
        
@xr.register_dataarray_accessor("gpm_api")
class GPM_DataArray_Accessor:
    
    def __init__(self, xarray_obj):
        if not isinstance(xarray_obj, (xr.DataArray, xr.Dataset)):
            raise TypeError("The 'gpm_api' accessor is available only for xr.Datasets and xr.DataArray.")
        self._obj = xarray_obj


    def crop(self, bbox):
        """Crop dataset."""
        return crop_dataset(self._obj, bbox)


    def title(self, time_idx=0, resolution="m", timezone="UTC",
              prefix_product=True, add_timestep=True):
        return get_dataset_title(self._obj, time_idx=time_idx,
                                 resolution=resolution, 
                                 timezone=timezone,
                                 prefix_product=prefix_product,
                                 add_timestep=add_timestep)
    
    
    def plot(self, ax=None, add_colorbar=True):
        da = self._obj
        p = _plot_map(ax=ax, da=da, add_colorbar=add_colorbar)
        return p 
    
    
    def plot_swath_lines(self, ax, **kwargs):
        p = _plot_swath_lines(self._obj, ax=ax, **kwargs)
        return p 
    
    
    def plot_transect_line(self, ax, color="black"):
        p = plot_transect_line(self._obj, ax=ax, color=color)
        return p 
    
    
    @property
    def pyresample_area(self):
        return get_pyresample_area(self._obj)
    
    
    @property
    def is_orbit(self):
        return is_orbit(self._obj)
    
    
    @property
    def is_grid(self):
        return is_grid(self._obj)