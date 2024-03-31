# -----------------------------------------------------------------------------.
# MIT License

# Copyright (c) 2024 GPM-API developers
#
# This file is part of GPM-API.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -----------------------------------------------------------------------------.
"""This module defines GPM-API xarray accessors."""
import numpy as np
import xarray as xr


class GPM_Base_Accessor:
    def __init__(self, xarray_obj):
        if not isinstance(xarray_obj, (xr.DataArray, xr.Dataset)):
            raise TypeError("The 'gpm' accessor is available only for xr.Dataset and xr.DataArray.")
        self._obj = xarray_obj

    def extent(self, padding=0):
        """Return the geographic extent (bbox) of the object."""
        from gpm.utils.geospatial import get_extent

        return get_extent(self._obj, padding=padding)

    def crop(self, extent):
        """Crop xarray object by bounding box."""
        from gpm.utils.geospatial import crop

        return crop(self._obj, extent)

    def crop_by_country(self, name):
        """Crop xarray object by country name."""
        from gpm.utils.geospatial import crop_by_country

        return crop_by_country(self._obj, name)

    def crop_by_continent(self, name):
        """Crop xarray object by continent name."""
        from gpm.utils.geospatial import crop_by_continent

        return crop_by_continent(self._obj, name)

    def get_crop_slices_by_extent(self, extent):
        """Get subsetting slices given the extent."""
        from gpm.utils.geospatial import get_crop_slices_by_extent

        return get_crop_slices_by_extent(self._obj, extent)

    def get_crop_slices_by_country(self, name):
        """Get subsetting slices given the country name."""
        from gpm.utils.geospatial import get_crop_slices_by_country

        return get_crop_slices_by_country(self._obj, name)

    def get_crop_slices_by_continent(self, name):
        """Get subsetting slices given the continent name."""
        from gpm.utils.geospatial import get_crop_slices_by_continent

        return get_crop_slices_by_continent(self._obj, name)

    @property
    def pyresample_area(self):
        from gpm.utils.pyresample import get_pyresample_area

        return get_pyresample_area(self._obj)

    def remap_on(self, dst_ds, radius_of_influence=20000, fill_value=np.nan):
        """Remap data from one dataset to another one."""
        from gpm.utils.pyresample import remap

        return remap(
            self._obj,
            dst_ds=dst_ds,
            radius_of_influence=radius_of_influence,
            fill_value=fill_value,
        )

    def collocate(
        self,
        product,
        product_type="RS",
        version=None,
        scan_modes=None,
        variables=None,
        groups=None,
        verbose=True,
        decode_cf=True,
        chunks={},
    ):
        """Collocate another product on the dataset.

        It assumes that along all the input dataset, there is an approximate collocated product.
        """
        from gpm.utils.collocation import collocate_product

        return collocate_product(
            self._obj,
            product=product,
            product_type=product_type,
            version=version,
            scan_modes=scan_modes,
            variables=variables,
            groups=groups,
            verbose=verbose,
            chunks=chunks,
            decode_cf=decode_cf,
        )

    #### Transect utility
    def define_transect_slices(
        self,
        direction="cross_track",
        lon=None,
        lat=None,
        variable=None,
        transect_kwargs={},
    ):
        from gpm.visualization.profile import get_transect_slices

        return get_transect_slices(
            self._obj,
            direction=direction,
            variable=variable,
            lon=lon,
            lat=lat,
            transect_kwargs=transect_kwargs,
        )

    def select_transect(
        self,
        direction="cross_track",
        lon=None,
        lat=None,
        variable=None,
        transect_kwargs={},
        keep_only_valid_variables=True,
    ):
        from gpm.visualization.profile import select_transect

        return select_transect(
            self._obj,
            direction=direction,
            variable=variable,
            lon=lon,
            lat=lat,
            transect_kwargs=transect_kwargs,
            keep_only_valid_variables=keep_only_valid_variables,
        )

    #### Profile utility
    def get_variable_at_bin(self, bin, variable=None):
        """Retrieve variable values at specific range bins."""
        from gpm.utils.manipulations import get_variable_at_bin

        return get_variable_at_bin(self._obj, bin=bin, variable=variable)

    def get_height_at_bin(self, bin):
        """Retrieve height values at specific range bins."""
        from gpm.utils.manipulations import get_height_at_bin

        return get_height_at_bin(self._obj, bin=bin)

    def slice_range_with_valid_data(self, variable=None):
        """Select the 'range' interval with valid data."""
        from gpm.utils.manipulations import slice_range_with_valid_data

        return slice_range_with_valid_data(self._obj, variable=variable)

    def slice_range_where_values(self, variable=None, vmin=-np.inf, vmax=np.inf):
        """Select the 'range' interval where values are within the [vmin, vmax] interval."""
        from gpm.utils.manipulations import slice_range_where_values

        return slice_range_where_values(self._obj, variable=variable, vmin=vmin, vmax=vmax)

    def slice_range_at_height(self, height):
        """Slice the 3D array at a given height."""
        from gpm.utils.manipulations import slice_range_at_height

        return slice_range_at_height(self._obj, height=height)

    def slice_range_at_value(self, value, variable=None):
        """Slice the 3D arrays where the variable values are close to value."""
        from gpm.utils.manipulations import slice_range_at_value

        return slice_range_at_value(self._obj, variable=variable, value=value)

    def slice_range_at_max_value(self, variable=None):
        """Slice the 3D arrays where the variable values are at maximum."""
        from gpm.utils.manipulations import slice_range_at_max_value

        return slice_range_at_max_value(self._obj, variable=variable)

    def slice_range_at_min_value(self, variable=None):
        """Slice the 3D arrays where the variable values are at minimum."""
        from gpm.utils.manipulations import slice_range_at_min_value

        return slice_range_at_min_value(self._obj, variable=variable)

    #### Dataset utility
    @property
    def is_orbit(self):
        from gpm.checks import is_orbit

        return is_orbit(self._obj)

    @property
    def is_grid(self):
        from gpm.checks import is_grid

        return is_grid(self._obj)

    @property
    def is_spatial_2d(self):
        from gpm.checks import is_spatial_2d

        return is_spatial_2d(self._obj)

    @property
    def is_spatial_3d(self):
        from gpm.checks import is_spatial_3d

        return is_spatial_3d(self._obj)

    @property
    def start_time(self):
        from gpm.io.checks import check_time

        if "time" in self._obj.coords:
            start_time = self._obj["time"].to_numpy()[0]
        elif "gpm_time" in self._obj.coords:
            start_time = self._obj["gpm_time"].to_numpy()[0]
        else:
            raise ValueError("Time coordinate not found")
        return check_time(start_time)

    @property
    def end_time(self):
        from gpm.io.checks import check_time

        if "time" in self._obj.coords:
            end_time = self._obj["time"].to_numpy()[-1]
        elif "gpm_time" in self._obj.coords:
            end_time = self._obj["gpm_time"].to_numpy()[-1]
        else:
            raise ValueError("Time coordinate not found")
        return check_time(end_time)

    @property
    def vertical_dimension(self):
        from gpm.checks import get_vertical_dimension

        return get_vertical_dimension(self._obj)

    @property
    def spatial_dimensions(self):
        from gpm.checks import get_spatial_dimensions

        return get_spatial_dimensions(self._obj)

    #### Dataset Quality Checks
    @property
    def is_regular(self):
        from gpm.utils.checks import is_regular

        return is_regular(self._obj)

    @property
    def has_regular_time(self):
        from gpm.utils.checks import has_regular_time

        return has_regular_time(self._obj)

    @property
    def has_contiguous_scans(self):
        from gpm.utils.checks import has_contiguous_scans

        return has_contiguous_scans(self._obj)

    @property
    def has_missing_granules(self):
        from gpm.utils.checks import has_missing_granules

        return has_missing_granules(self._obj)

    @property
    def has_valid_geolocation(self):
        from gpm.utils.checks import has_valid_geolocation

        return has_valid_geolocation(self._obj)

    #### Subsetting utility
    def subset_by_time(self, start_time=None, end_time=None):
        from gpm.utils.time import subset_by_time

        return subset_by_time(self._obj, start_time=start_time, end_time=end_time)

    def subset_by_time_slice(self, slice):
        from gpm.utils.time import subset_by_time_slice

        return subset_by_time_slice(self._obj, slice=slice)

    def get_slices_regular_time(self, tolerance=None, min_size=1):
        from gpm.utils.checks import get_slices_regular_time

        return get_slices_regular_time(self._obj, tolerance=tolerance, min_size=min_size)

    def get_slices_contiguous_scans(self, min_size=2, min_n_scans=3):
        from gpm.utils.checks import get_slices_contiguous_scans

        return get_slices_contiguous_scans(self._obj, min_size=min_size, min_n_scans=min_n_scans)

    def get_slices_contiguous_granules(self, min_size=2):
        from gpm.utils.checks import get_slices_contiguous_granules

        return get_slices_contiguous_granules(self._obj, min_size=min_size)

    def get_slices_valid_geolocation(self, min_size=2):
        from gpm.utils.checks import get_slices_valid_geolocation

        return get_slices_valid_geolocation(self._obj, min_size=min_size)

    def get_slices_regular(self, min_size=None, min_n_scans=3):
        from gpm.utils.checks import get_slices_regular

        return get_slices_regular(self._obj, min_size=min_size, min_n_scans=min_n_scans)

    #### Plotting utility
    def plot_transect_line(
        self,
        ax,
        add_direction=True,
        text_kwargs={},
        line_kwargs={},
        **common_kwargs,
    ):
        from gpm.visualization.profile import plot_transect_line

        return plot_transect_line(
            self._obj,
            ax=ax,
            add_direction=add_direction,
            text_kwargs=text_kwargs,
            line_kwargs=line_kwargs,
            **common_kwargs,
        )

    def plot_swath(
        self,
        ax=None,
        facecolor="orange",
        edgecolor="black",
        alpha=0.4,
        fig_kwargs=None,
        subplot_kwargs=None,
        add_background=True,
        **plot_kwargs,
    ):
        from gpm.visualization.orbit import plot_swath

        return plot_swath(
            self._obj,
            ax=ax,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            add_background=add_background,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            **plot_kwargs,
        )

    def plot_swath_lines(
        self,
        ax=None,
        x="lon",
        y="lat",
        linestyle="--",
        color="k",
        add_background=True,
        fig_kwargs=None,
        subplot_kwargs=None,
        **plot_kwargs,
    ):
        from gpm.visualization.orbit import plot_swath_lines

        return plot_swath_lines(
            self._obj,
            ax=ax,
            x=x,
            y=y,
            linestyle=linestyle,
            color=color,
            add_background=add_background,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            **plot_kwargs,
        )

    def plot_map_mesh(
        self,
        x="lon",
        y="lat",
        ax=None,
        edgecolors="k",
        linewidth=0.1,
        add_background=True,
        fig_kwargs=None,
        subplot_kwargs=None,
        **plot_kwargs,
    ):
        from gpm.visualization.plot import plot_map_mesh

        return plot_map_mesh(
            xr_obj=self._obj,
            x=x,
            y=y,
            ax=ax,
            edgecolors=edgecolors,
            linewidth=linewidth,
            add_background=add_background,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            **plot_kwargs,
        )

    def plot_map_mesh_centroids(
        self,
        x="lon",
        y="lat",
        ax=None,
        c="r",
        s=1,
        add_background=True,
        fig_kwargs=None,
        subplot_kwargs=None,
        **plot_kwargs,
    ):
        from gpm.visualization.plot import plot_map_mesh_centroids

        return plot_map_mesh_centroids(
            self._obj,
            x=x,
            y=y,
            ax=ax,
            c=c,
            s=s,
            add_background=add_background,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            **plot_kwargs,
        )


@xr.register_dataset_accessor("gpm")
class GPM_Dataset_Accessor(GPM_Base_Accessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    @property
    def variables(self):
        from gpm.checks import get_dataset_variables

        return get_dataset_variables(self._obj, sort=True)

    @property
    def spatial_2d_variables(self):
        from gpm.checks import get_spatial_2d_variables

        return get_spatial_2d_variables(self._obj)

    @property
    def spatial_3d_variables(self):
        from gpm.checks import get_spatial_3d_variables

        return get_spatial_3d_variables(self._obj)

    @property
    def frequency_variables(self):
        from gpm.checks import get_frequency_variables

        return get_frequency_variables(self._obj)

    def select_spatial_3d_variables(self, strict=False, squeeze=True):
        from gpm.utils.manipulations import select_spatial_3d_variables

        return select_spatial_3d_variables(self._obj, strict=strict, squeeze=squeeze)

    def select_spatial_2d_variables(self, strict=False, squeeze=True):
        from gpm.utils.manipulations import select_spatial_2d_variables

        return select_spatial_2d_variables(self._obj, strict=strict, squeeze=squeeze)

    def set_encoding(self, encoding_dict=None):
        from gpm.encoding.routines import set_encoding

        return set_encoding(self._obj, encoding_dict=encoding_dict)

    def title(
        self,
        add_timestep=True,
        time_idx=None,
        resolution="m",
        timezone="UTC",
    ):
        from gpm.visualization.title import get_dataset_title

        return get_dataset_title(
            self._obj,
            add_timestep=add_timestep,
            time_idx=time_idx,
            resolution=resolution,
            timezone=timezone,
        )

    def plot_map(
        self,
        variable,
        ax=None,
        x="lon",
        y="lat",
        rgb=False,
        add_colorbar=True,
        add_swath_lines=True,
        add_background=True,
        interpolation="nearest",  # used only for GPM grid object
        fig_kwargs=None,
        subplot_kwargs=None,
        cbar_kwargs=None,
        **plot_kwargs,
    ):
        from gpm.visualization.plot import plot_map

        da = self._obj[variable]
        return plot_map(
            da,
            ax=ax,
            x=x,
            y=y,
            add_colorbar=add_colorbar,
            rgb=rgb,
            add_swath_lines=add_swath_lines,
            add_background=add_background,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )

    def plot_image(
        self,
        variable,
        ax=None,
        x=None,
        y=None,
        add_colorbar=True,
        interpolation="nearest",
        fig_kwargs=None,
        cbar_kwargs=None,
        **plot_kwargs,
    ):
        from gpm.visualization.plot import plot_image

        da = self._obj[variable]
        return plot_image(
            da,
            ax=ax,
            x=x,
            y=y,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )

    def plot_transect(
        self,
        variable,
        ax=None,
        add_colorbar=True,
        zoom=True,
        fig_kwargs=None,
        cbar_kwargs=None,
        **plot_kwargs,
    ):
        from gpm.visualization.profile import plot_transect

        da = self._obj[variable]
        return plot_transect(
            da,
            ax=ax,
            add_colorbar=add_colorbar,
            zoom=zoom,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )

    def available_retrievals(self):
        """Available GPM-API retrievals for that GPM product."""
        from gpm.retrievals.routines import available_retrievals

        return available_retrievals(self._obj)

    def retrieve(self, name, **kwargs):
        """Retrieve a GPM-API variable."""
        from gpm.retrievals.routines import get_retrieval_variable

        return get_retrieval_variable(self._obj, name=name, **kwargs)

    def slice_range_at_temperature(self, temperature, variable_temperature="airTemperature"):
        """Slice the 3D arrays along a specific isotherm."""
        from gpm.utils.manipulations import slice_range_at_temperature

        return slice_range_at_temperature(
            self._obj,
            temperature=temperature,
            variable_temperature=variable_temperature,
        )

    def to_pandas_dataframe(self):
        """Convert xr.Dataset to Pandas Dataframe. Expects xr.Dataset with only 2D spatial DataArrays."""
        from gpm.bucket.processing import ds_to_pd_df_function

        return ds_to_pd_df_function(self._obj)

    def to_dask_dataframe(self):
        """Convert xr.Dataset to Dask Dataframe. Expects xr.Dataset with only 2D spatial DataArrays."""
        from gpm.bucket.processing import ds_to_dask_df_function

        return ds_to_dask_df_function(self._obj)


@xr.register_dataarray_accessor("gpm")
class GPM_DataArray_Accessor(GPM_Base_Accessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    def get_slices_var_equals(self, dim, values, union=True, criteria="all"):
        from gpm.utils.checks import get_slices_var_equals

        return get_slices_var_equals(
            self._obj,
            dim=dim,
            values=values,
            union=union,
            criteria=criteria,
        )

    def get_slices_var_between(self, dim, vmin=-np.inf, vmax=np.inf, criteria="all"):
        from gpm.utils.checks import get_slices_var_between

        return get_slices_var_between(self._obj, dim=dim, vmin=vmin, vmax=vmax, criteria=criteria)

    def title(
        self,
        prefix_product=True,
        add_timestep=True,
        time_idx=None,
        resolution="m",
        timezone="UTC",
    ):
        from gpm.visualization.title import get_dataarray_title

        return get_dataarray_title(
            self._obj,
            prefix_product=prefix_product,
            add_timestep=add_timestep,
            time_idx=time_idx,
            resolution=resolution,
            timezone=timezone,
        )

    def plot_map(
        self,
        ax=None,
        x="lon",
        y="lat",
        add_colorbar=True,
        add_swath_lines=True,
        add_background=True,
        interpolation="nearest",  # used only for GPM grid object
        rgb=False,
        fig_kwargs=None,
        subplot_kwargs=None,
        cbar_kwargs=None,
        **plot_kwargs,
    ):
        from gpm.visualization.plot import plot_map

        da = self._obj
        return plot_map(
            da,
            ax=ax,
            x=x,
            y=y,
            add_colorbar=add_colorbar,
            add_swath_lines=add_swath_lines,
            add_background=add_background,
            rgb=rgb,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )

    def plot_image(
        self,
        ax=None,
        x=None,
        y=None,
        add_colorbar=True,
        interpolation="nearest",
        fig_kwargs=None,
        cbar_kwargs=None,
        **plot_kwargs,
    ):
        from gpm.visualization.plot import plot_image

        da = self._obj
        return plot_image(
            da,
            ax=ax,
            x=x,
            y=y,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )

    def plot_transect(
        self,
        ax=None,
        add_colorbar=True,
        zoom=True,
        fig_kwargs=None,
        cbar_kwargs=None,
        **plot_kwargs,
    ):
        from gpm.visualization.profile import plot_transect

        da = self._obj
        return plot_transect(
            da,
            ax=ax,
            add_colorbar=add_colorbar,
            zoom=zoom,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )

    def integrate_profile_concentration(self, name, scale_factor=None, units=None):
        from gpm.utils.manipulations import integrate_profile_concentration

        return integrate_profile_concentration(
            self._obj,
            name=name,
            scale_factor=scale_factor,
            units=units,
        )
