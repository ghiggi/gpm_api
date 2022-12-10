#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 09:31:39 2022

@author: ghiggi
"""
import numpy as np
import xarray as xr


class GPM_Base_Accessor:
    def __init__(self, xarray_obj):
        if not isinstance(xarray_obj, (xr.DataArray, xr.Dataset)):
            raise TypeError(
                "The 'gpm_api' accessor is available only for xr.Dataset and xr.DataArray."
            )
        self._obj = xarray_obj

    def crop(self, bbox):
        """Crop xarray object by bounding box."""
        from gpm_api.utils.geospatial import crop
        return crop(self._obj, bbox)
    
    def crop_by_country(self, name):
        """Crop xarray object by country name."""
        from gpm_api.utils.geospatial import crop_by_country
        return crop_by_country(self._obj, name)

    @property
    def pyresample_area(self):
        from gpm_api.utils.geospatial import get_pyresample_area

        return get_pyresample_area(self._obj)

    @property
    def is_orbit(self):
        from gpm_api.utils.geospatial import is_orbit

        return is_orbit(self._obj)

    @property
    def is_grid(self):
        from gpm_api.utils.geospatial import is_grid

        return is_grid(self._obj)

    @property
    def is_spatial_2D_field(self):
        from gpm_api.utils.geospatial import is_spatial_2D_field

        return is_spatial_2D_field(self._obj)

    @property
    def has_regular_timesteps(self):
        from gpm_api.utils.time import has_regular_timesteps

        return has_regular_timesteps(self._obj)

    def subset_by_time(self, start_time=None, end_time=None):
        from gpm_api.utils.time import subset_by_time

        return subset_by_time(self._obj, start_time=start_time, end_time=end_time)

    def subset_by_time_slice(self, slice):
        from gpm_api.utils.time import subset_by_time_slice

        return subset_by_time_slice(self._obj, slice=slice)

    def get_regular_time_slices(self, tolerance=None):
        from gpm_api.utils.time import get_regular_time_slices

        return get_regular_time_slices(self._obj, tolerance=tolerance)
    
    def plot_transect_line(self, ax, color="black"):
        from gpm_api.visualization.profile import plot_transect_line

        p = plot_transect_line(self._obj, ax=ax, color=color)
        return p

    def plot_swath_lines(self, ax, **kwargs):
        from gpm_api.visualization.orbit import plot_swath_lines

        p = plot_swath_lines(self._obj, ax=ax, **kwargs)
        return p


@xr.register_dataset_accessor("gpm_api")
class GPM_Dataset_Accessor(GPM_Base_Accessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
    
    def title(
        self,
        add_timestep=True,
        time_idx=None,
        resolution="m",
        timezone="UTC",
    ):
        from gpm_api.visualization.title import get_dataset_title
        title = get_dataset_title(
            self._obj,
            add_timestep=add_timestep,
            time_idx=time_idx,
            resolution=resolution,
            timezone=timezone,
        )
        return title
    
    def plot_map(self, variable, ax=None, add_colorbar=True):
        from gpm_api.visualization.plot import plot_map
        da = self._obj[variable]
        p = plot_map(ax=ax, da=da, add_colorbar=add_colorbar)
        return p

    def patch_generator(
        self,
        variable,
        min_value_threshold=0.1,
        max_value_threshold=np.inf,
        min_area_threshold=10,
        max_area_threshold=np.inf,
        footprint_buffer=None,
        sort_by="max",
        sort_decreasing=True,
        label_name="label",
        n_patches=None,
        patch_margin=(48, 20),
    ):
        from gpm_api.patch.generator import get_ds_patch_generator

        gen = get_ds_patch_generator(
            self._obj,
            variable=variable,
            # Labels options
            min_value_threshold=min_value_threshold,
            max_value_threshold=max_value_threshold,
            min_area_threshold=min_area_threshold,
            max_area_threshold=max_area_threshold,
            footprint_buffer=footprint_buffer,
            sort_by=sort_by,
            sort_decreasing=sort_decreasing,
            # Patch options
            n_patches=n_patches,
            patch_margin=patch_margin,
        )
        return gen

    def plot_patches(
        self,
        variable,
        min_value_threshold=-np.inf,
        max_value_threshold=np.inf,
        min_area_threshold=1,
        max_area_threshold=np.inf,
        footprint_buffer=None,
        sort_by="area",
        sort_decreasing=True,
        n_patches=None,
        patch_margin=None,
        interpolation="nearest",
    ):
        from gpm_api.visualization.patches import plot_patches

        data_array = self._obj[variable]
        plot_patches(
            data_array=data_array,
            min_value_threshold=min_value_threshold,
            max_value_threshold=max_value_threshold,
            min_area_threshold=min_area_threshold,
            max_area_threshold=max_area_threshold,
            footprint_buffer=footprint_buffer,
            sort_by=sort_by,
            sort_decreasing=sort_decreasing,
            n_patches=n_patches,
            patch_margin=patch_margin,
        )


@xr.register_dataarray_accessor("gpm_api")
class GPM_DataArray_Accessor(GPM_Base_Accessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    def title(
        self,
        prefix_product=True,
        add_timestep=True,
        time_idx=None,
        resolution="m",
        timezone="UTC",
    ):
        from gpm_api.visualization.title import get_dataarray_title
        title = get_dataarray_title(
            self._obj,
            prefix_product=prefix_product,
            add_timestep=add_timestep,
            time_idx=time_idx,
            resolution=resolution,
            timezone=timezone,
        )
        return title

    def plot_map(self, ax=None, add_colorbar=True):
        from gpm_api.visualization.plot import plot_map

        da = self._obj
        p = plot_map(ax=ax, da=da, add_colorbar=add_colorbar)
        return p

    def plot_image(self, ax=None, add_colorbar=True, interpolation="nearest"):
        from gpm_api.visualization.plot import plot_image

        da = self._obj
        p = plot_image(
            da, ax=ax, add_colorbar=add_colorbar, interpolation=interpolation
        )
        return p

    def patch_generator(
        self,
        min_value_threshold=0.1,
        max_value_threshold=np.inf,
        min_area_threshold=10,
        max_area_threshold=np.inf,
        footprint_buffer=None,
        sort_by="max",
        sort_decreasing=True,
        label_name="label",
        n_patches=None,
        patch_margin=(48, 20),
    ):
        from gpm_api.patch.generator import get_da_patch_generator

        gen = get_da_patch_generator(
            self._obj,
            # Labels options
            min_value_threshold=min_value_threshold,
            max_value_threshold=max_value_threshold,
            min_area_threshold=min_area_threshold,
            max_area_threshold=max_area_threshold,
            footprint_buffer=footprint_buffer,
            sort_by=sort_by,
            sort_decreasing=sort_decreasing,
            # Patch options
            n_patches=n_patches,
            patch_margin=patch_margin,
        )
        return gen

    def plot_patches(
        self,
        min_value_threshold=-np.inf,
        max_value_threshold=np.inf,
        min_area_threshold=1,
        max_area_threshold=np.inf,
        footprint_buffer=None,
        sort_by="area",
        sort_decreasing=True,
        label_name="label",
        n_patches=None,
        patch_margin=None,
        interpolation="nearest",
    ):
        from gpm_api.visualization.patches import plot_patches

        plot_patches(
            data_array=self._obj,
            min_value_threshold=min_value_threshold,
            max_value_threshold=max_value_threshold,
            min_area_threshold=min_area_threshold,
            max_area_threshold=max_area_threshold,
            footprint_buffer=footprint_buffer,
            sort_by=sort_by,
            sort_decreasing=sort_decreasing,
            label_name=label_name,
            n_patches=n_patches,
            patch_margin=patch_margin,
        )
