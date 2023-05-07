#!/usr/bin/env python3
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

    def crop(self, extent):
        """Crop xarray object by bounding box."""
        from gpm_api.utils.geospatial import crop

        return crop(self._obj, extent)

    def crop_by_country(self, name):
        """Crop xarray object by country name."""
        from gpm_api.utils.geospatial import crop_by_country

        return crop_by_country(self._obj, name)

    def crop_by_continent(self, name):
        """Crop xarray object by continent name."""
        from gpm_api.utils.geospatial import crop_by_continent

        return crop_by_continent(self._obj, name)

    def get_crop_slices_by_extent(self, extent):
        """Get subsetting slices given the extent."""
        from gpm_api.utils.geospatial import get_crop_slices_by_extent

        return get_crop_slices_by_extent(self._obj, extent)

    def get_crop_slices_by_country(self, name):
        """Get subsetting slices given the country name."""
        from gpm_api.utils.geospatial import get_crop_slices_by_country

        return get_crop_slices_by_country(self._obj, name)

    def get_crop_slices_by_continent(self, name):
        """Get subsetting slices given the continent name."""
        from gpm_api.utils.geospatial import get_crop_slices_by_continent

        return get_crop_slices_by_continent(self._obj, name)

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
    def is_spatial_2d(self):
        from gpm_api.utils.geospatial import is_spatial_2d

        return is_spatial_2d(self._obj)

    @property
    def is_regular(self):
        from gpm_api.utils.checks import is_regular

        return is_regular(self._obj)

    @property
    def has_regular_time(self):
        from gpm_api.utils.checks import has_regular_time

        return has_regular_time(self._obj)

    @property
    def has_contiguous_scans(self):
        from gpm_api.utils.checks import has_contiguous_scans

        return has_contiguous_scans(self._obj)

    @property
    def has_missing_granules(self):
        from gpm_api.utils.checks import has_missing_granules

        return has_missing_granules(self._obj)

    @property
    def has_valid_geolocation(self):
        from gpm_api.utils.checks import has_valid_geolocation

        return has_valid_geolocation(self._obj)

    @property
    def start_time(self):
        if "time" in self._obj.coords:
            start_time = self._obj["time"].values[0]
        elif "gpm_time" in self._obj.coords:
            start_time = self._obj["gpm_time"].values[0]
        else:
            raise ValueError("Time coordinate not found")
        return start_time

    @property
    def end_time(self):
        if "time" in self._obj.coords:
            end_time = self._obj["time"].values[-1]
        elif "gpm_time" in self._obj.coords:
            end_time = self._obj["gpm_time"].values[-1]
        else:
            raise ValueError("Time coordinate not found")
        return end_time

    def subset_by_time(self, start_time=None, end_time=None):
        from gpm_api.utils.time import subset_by_time

        return subset_by_time(self._obj, start_time=start_time, end_time=end_time)

    def subset_by_time_slice(self, slice):
        from gpm_api.utils.time import subset_by_time_slice

        return subset_by_time_slice(self._obj, slice=slice)

    def get_slices_regular_time(self, tolerance=None, min_size=1):
        from gpm_api.utils.checks import get_slices_regular_time

        return get_slices_regular_time(self._obj, tolerance=tolerance, min_size=min_size)

    def get_slices_contiguous_scans(self, min_size=2):
        from gpm_api.utils.checks import get_slices_contiguous_scans

        return get_slices_contiguous_scans(self._obj, min_size=min_size)

    def get_slices_contiguous_granules(self, min_size=2):
        from gpm_api.utils.checks import get_slices_contiguous_granules

        return get_slices_contiguous_granules(self._obj, min_size=min_size)

    def get_slices_valid_geolocation(self, min_size=2):
        from gpm_api.utils.checks import get_slices_valid_geolocation

        return get_slices_valid_geolocation(self._obj, min_size=min_size)

    def get_slices_regular(self, min_size=2):
        from gpm_api.utils.checks import get_slices_regular

        return get_slices_regular(self._obj, min_size=min_size)

    def plot_transect_line(self, ax=None, color="black"):
        from gpm_api.visualization.profile import plot_transect_line

        p = plot_transect_line(self._obj, ax=ax, color=color)
        return p

    def plot_swath_lines(self, ax=None, **kwargs):
        from gpm_api.visualization.orbit import plot_swath_lines

        p = plot_swath_lines(self._obj, ax=ax, **kwargs)
        return p

    def label_object(
        self,
        variable=None,
        min_value_threshold=0.1,
        max_value_threshold=np.inf,
        min_area_threshold=1,
        max_area_threshold=np.inf,
        footprint=None,
        sort_by="area",
        sort_decreasing=True,
        label_name="label",
    ):
        from gpm_api.patch.labels import label_xarray_object

        xr_obj = label_xarray_object(
            self._obj,
            variable=variable,
            # Labels options
            min_value_threshold=min_value_threshold,
            max_value_threshold=max_value_threshold,
            min_area_threshold=min_area_threshold,
            max_area_threshold=max_area_threshold,
            footprint=footprint,
            sort_by=sort_by,
            sort_decreasing=sort_decreasing,
        )
        return xr_obj

    def labels_patch_generator(
        self,
        variable=None,
        min_value_threshold=0.1,
        max_value_threshold=np.inf,
        min_area_threshold=1,
        max_area_threshold=np.inf,
        footprint=None,
        sort_by="area",
        sort_decreasing=True,
        n_patches=None,
        padding=None,
        min_patch_size=None,
    ):
        from gpm_api.patch.labels_patch import labels_patch_generator

        gen = labels_patch_generator(
            self._obj,
            variable=variable,
            # Labels options
            min_value_threshold=min_value_threshold,
            max_value_threshold=max_value_threshold,
            min_area_threshold=min_area_threshold,
            max_area_threshold=max_area_threshold,
            footprint=footprint,
            sort_by=sort_by,
            sort_decreasing=sort_decreasing,
            # Patch options
            n_patches=n_patches,
            padding=padding,
            min_patch_size=min_patch_size,
        )
        return gen


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

    def plot_map(
        self,
        variable,
        ax=None,
        add_colorbar=True,
        add_swath_lines=True,
        interpolation="nearest",  # used only for GPM grid object
        fig_kwargs={},
        subplot_kwargs={},
        cbar_kwargs={},
        **plot_kwargs,
    ):
        from gpm_api.visualization.plot import plot_map

        da = self._obj[variable]
        p = plot_map(
            ax=ax,
            da=da,
            add_colorbar=add_colorbar,
            add_swath_lines=add_swath_lines,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
        return p

    def plot_image(
        self,
        variable,
        ax=None,
        add_colorbar=True,
        interpolation="nearest",
        fig_kwargs={},
        cbar_kwargs={},
        **plot_kwargs,
    ):
        from gpm_api.visualization.plot import plot_image

        da = self._obj[variable]
        p = plot_image(
            da,
            ax=ax,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
        return p

    def plot_patches(
        self,
        variable,
        min_value_threshold=-np.inf,
        max_value_threshold=np.inf,
        min_area_threshold=1,
        max_area_threshold=np.inf,
        footprint=None,
        sort_by="area",
        sort_decreasing=True,
        n_patches=None,
        min_patch_size=None,
        padding=None,
        add_colorbar=True,
        interpolation="nearest",
        fig_kwargs={},
        cbar_kwargs={},
        **plot_kwargs,
    ):
        from gpm_api.visualization.labels import plot_patches

        data_array = self._obj[variable]
        plot_patches(
            data_array=data_array,
            min_value_threshold=min_value_threshold,
            max_value_threshold=max_value_threshold,
            min_area_threshold=min_area_threshold,
            max_area_threshold=max_area_threshold,
            footprint=footprint,
            sort_by=sort_by,
            sort_decreasing=sort_decreasing,
            n_patches=n_patches,
            min_patch_size=min_patch_size,
            padding=padding,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )


@xr.register_dataarray_accessor("gpm_api")
class GPM_DataArray_Accessor(GPM_Base_Accessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    def get_slices_var_equals(self, dim, values, union=True, criteria="all"):
        from gpm_api.utils.checks import get_slices_var_equals

        return get_slices_var_equals(
            self._obj, dim=dim, values=values, union=union, criteria=criteria
        )

    def get_slices_var_between(self, dim, vmin=-np.inf, vmax=np.inf, criteria="all"):
        from gpm_api.utils.checks import get_slices_var_between

        return get_slices_var_between(self._obj, dim=dim, vmin=vmin, vmax=vmax, criteria=criteria)

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

    def plot_map(
        self,
        ax=None,
        add_colorbar=True,
        add_swath_lines=True,
        interpolation="nearest",  # used only for GPM grid object
        fig_kwargs={},
        subplot_kwargs={},
        cbar_kwargs={},
        **plot_kwargs,
    ):
        from gpm_api.visualization.plot import plot_map

        da = self._obj
        p = plot_map(
            ax=ax,
            da=da,
            add_colorbar=add_colorbar,
            add_swath_lines=add_swath_lines,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
        return p

    def plot_map_mesh(
        self,
        ax=None,
        edgecolors="k",
        linewidth=0.1,
        fig_kwargs={},
        subplot_kwargs={},
        **plot_kwargs,
    ):
        from gpm_api.visualization.plot import plot_map_mesh

        da = self._obj
        p = plot_map_mesh(
            ax=ax,
            da=da,
            edgecolors=edgecolors,
            linewidth=linewidth,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            **plot_kwargs,
        )
        return p

    def plot_image(
        self,
        ax=None,
        add_colorbar=True,
        interpolation="nearest",
        fig_kwargs={},
        cbar_kwargs={},
        **plot_kwargs,
    ):
        from gpm_api.visualization.plot import plot_image

        da = self._obj
        p = plot_image(
            da,
            ax=ax,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
        return p

    def plot_patches(
        self,
        min_value_threshold=0.1,
        max_value_threshold=np.inf,
        min_area_threshold=1,
        max_area_threshold=np.inf,
        footprint=None,
        sort_by="area",
        sort_decreasing=True,
        n_patches=None,
        min_patch_size=None,
        padding=None,
        add_colorbar=True,
        interpolation="nearest",
        fig_kwargs={},
        cbar_kwargs={},
        **plot_kwargs,
    ):
        from gpm_api.visualization.labels import plot_patches

        plot_patches(
            data_array=self._obj,
            min_value_threshold=min_value_threshold,
            max_value_threshold=max_value_threshold,
            min_area_threshold=min_area_threshold,
            max_area_threshold=max_area_threshold,
            footprint=footprint,
            sort_by=sort_by,
            sort_decreasing=sort_decreasing,
            n_patches=n_patches,
            min_patch_size=min_patch_size,
            padding=padding,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
