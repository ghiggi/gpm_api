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
import functools
import importlib
import inspect
import re
import sys
from typing import Callable

import numpy as np
import xarray as xr


def get_imported_gpm_method_path(function: Callable) -> tuple[str, str]:
    """Get path of imported gpm method in accessor method source code (format is "module.method"))."""
    source = inspect.getsource(function)
    import_pattern = re.compile(r"from (\S+) import (\S+)")
    match = import_pattern.search(source)
    if match:
        module = match.group(1)
        method_name = match.group(2)
        return module, method_name
    raise ValueError(f"No import statement found in accessor {function} method.")


def get_imported_gpm_method(accessor_method: Callable) -> Callable:
    """Return the source function called by the accessor method."""
    try:
        imported_module, imported_method_name = get_imported_gpm_method_path(accessor_method)
        module = importlib.import_module(imported_module)
        function = getattr(module, imported_method_name)
    except Exception:
        function = None
    return function


def auto_wrap_docstring(func):
    """Decorator to add the source function docstring to the accessor method."""
    # If running pytest units, return the original accessor method
    # - This is required for testing accessor arguments against source functions
    if "pytest" in sys.modules:
        return func
    # Else find the source function
    source_function = get_imported_gpm_method(func)
    # Retrieve current accessor docstring
    accessor_doc = func.__doc__
    # If source function found and accessor docstring not specified
    if callable(source_function) and accessor_doc is None:
        # Set docstring of the source function
        return functools.wraps(source_function, assigned=("__doc__",))(func)
    # If no import statement is found or docstring already specified, return the original function
    return func


class GPM_Base_Accessor:
    def __init__(self, xarray_obj):
        if not isinstance(xarray_obj, (xr.DataArray, xr.Dataset)):
            raise TypeError("The 'gpm' accessor is available only for xarray.Dataset and xarray.DataArray.")
        self._obj = xarray_obj

    @auto_wrap_docstring
    def sel(self, indexers=None, drop=False, **indexers_kwargs):
        from gpm.utils.subsetting import sel

        return sel(self._obj, indexers=indexers, drop=drop, **indexers_kwargs)

    @auto_wrap_docstring
    def extent(self, padding=0, size=None):
        from gpm.utils.geospatial import get_geographic_extent_from_xarray

        return get_geographic_extent_from_xarray(self._obj, padding=padding, size=size)

    @auto_wrap_docstring
    def crop(self, extent):
        from gpm.utils.geospatial import crop

        return crop(self._obj, extent)

    @auto_wrap_docstring
    def crop_by_country(self, name):
        from gpm.utils.geospatial import crop_by_country

        return crop_by_country(self._obj, name)

    @auto_wrap_docstring
    def crop_by_continent(self, name):
        from gpm.utils.geospatial import crop_by_continent

        return crop_by_continent(self._obj, name)

    @auto_wrap_docstring
    def crop_around_point(self, lon, lat, distance=None, size=None):
        from gpm.utils.geospatial import crop_around_point

        return crop_around_point(self._obj, lon=lon, lat=lat, distance=distance, size=size)

    @auto_wrap_docstring
    def get_crop_slices_by_extent(self, extent):
        from gpm.utils.geospatial import get_crop_slices_by_extent

        return get_crop_slices_by_extent(self._obj, extent)

    @auto_wrap_docstring
    def get_crop_slices_by_country(self, name):
        from gpm.utils.geospatial import get_crop_slices_by_country

        return get_crop_slices_by_country(self._obj, name)

    @auto_wrap_docstring
    def get_crop_slices_by_continent(self, name):
        from gpm.utils.geospatial import get_crop_slices_by_continent

        return get_crop_slices_by_continent(self._obj, name)

    @auto_wrap_docstring
    def get_crop_slices_around_point(self, lon, lat, distance=None, size=None):
        from gpm.utils.geospatial import get_crop_slices_around_point

        return get_crop_slices_around_point(self._obj, lon=lon, lat=lat, distance=distance, size=size)

    @property
    def pyresample_area(self):
        from gpm.utils.pyresample import get_pyresample_area

        return get_pyresample_area(self._obj)

    @auto_wrap_docstring
    def remap_on(self, dst_ds, radius_of_influence=20000, fill_value=np.nan):
        from gpm.utils.pyresample import remap

        return remap(
            self._obj,
            dst_ds=dst_ds,
            radius_of_influence=radius_of_influence,
            fill_value=fill_value,
        )

    @auto_wrap_docstring
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
    @auto_wrap_docstring
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

    @auto_wrap_docstring
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
    @auto_wrap_docstring
    def slice_range_at_bin(self, bins):
        from gpm.utils.manipulations import slice_range_at_bin

        return slice_range_at_bin(self._obj, bins=bins)

    @auto_wrap_docstring
    def get_height_at_bin(self, bins):
        from gpm.utils.manipulations import get_height_at_bin

        return get_height_at_bin(self._obj, bins=bins)

    @auto_wrap_docstring
    def subset_range_with_valid_data(self, variable=None):
        from gpm.utils.manipulations import subset_range_with_valid_data

        return subset_range_with_valid_data(self._obj, variable=variable)

    @auto_wrap_docstring
    def subset_range_where_values(self, variable=None, vmin=-np.inf, vmax=np.inf):
        from gpm.utils.manipulations import subset_range_where_values

        return subset_range_where_values(self._obj, variable=variable, vmin=vmin, vmax=vmax)

    @auto_wrap_docstring
    def slice_range_at_height(self, value):
        from gpm.utils.manipulations import slice_range_at_height

        return slice_range_at_height(self._obj, value=value)

    @auto_wrap_docstring
    def slice_range_at_value(self, value, variable=None):
        from gpm.utils.manipulations import slice_range_at_value

        return slice_range_at_value(self._obj, variable=variable, value=value)

    @auto_wrap_docstring
    def slice_range_at_max_value(self, variable=None):
        from gpm.utils.manipulations import slice_range_at_max_value

        return slice_range_at_max_value(self._obj, variable=variable)

    @auto_wrap_docstring
    def slice_range_at_min_value(self, variable=None):
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
    @auto_wrap_docstring
    def subset_by_time(self, start_time=None, end_time=None):
        from gpm.utils.time import subset_by_time

        return subset_by_time(self._obj, start_time=start_time, end_time=end_time)

    @auto_wrap_docstring
    def subset_by_time_slice(self, slice):
        from gpm.utils.time import subset_by_time_slice

        return subset_by_time_slice(self._obj, slice=slice)

    @auto_wrap_docstring
    def get_slices_regular_time(self, tolerance=None, min_size=1):
        from gpm.utils.checks import get_slices_regular_time

        return get_slices_regular_time(self._obj, tolerance=tolerance, min_size=min_size)

    @auto_wrap_docstring
    def get_slices_contiguous_scans(
        self,
        min_size=2,
        min_n_scans=3,
        x="lon",
        y="lat",
        along_track_dim="along_track",
        cross_track_dim="cross_track",
    ):
        from gpm.utils.checks import get_slices_contiguous_scans

        return get_slices_contiguous_scans(
            self._obj,
            min_size=min_size,
            min_n_scans=min_n_scans,
            x=x,
            y=y,
            cross_track_dim=cross_track_dim,
            along_track_dim=along_track_dim,
        )

    @auto_wrap_docstring
    def get_slices_contiguous_granules(self, min_size=2):
        from gpm.utils.checks import get_slices_contiguous_granules

        return get_slices_contiguous_granules(self._obj, min_size=min_size)

    @auto_wrap_docstring
    def get_slices_valid_geolocation(
        self,
        min_size=2,
        x="lon",
        y="lat",
        along_track_dim="along_track",
        cross_track_dim="cross_track",
    ):
        from gpm.utils.checks import get_slices_valid_geolocation

        return get_slices_valid_geolocation(
            self._obj,
            min_size=min_size,
            x=x,
            y=y,
            cross_track_dim=cross_track_dim,
            along_track_dim=along_track_dim,
        )

    @auto_wrap_docstring
    def get_slices_regular(
        self,
        min_size=None,
        min_n_scans=3,
        x="lon",
        y="lat",
        along_track_dim="along_track",
        cross_track_dim="cross_track",
    ):
        from gpm.utils.checks import get_slices_regular

        return get_slices_regular(
            self._obj,
            min_size=min_size,
            min_n_scans=min_n_scans,
            x=x,
            y=y,
            cross_track_dim=cross_track_dim,
            along_track_dim=along_track_dim,
        )

    #### Plotting utility
    @auto_wrap_docstring
    def plot_transect_line(
        self,
        ax=None,
        add_direction=True,
        add_background=True,
        fig_kwargs=None,
        subplot_kwargs=None,
        text_kwargs=None,
        line_kwargs=None,
        **common_kwargs,
    ):
        from gpm.visualization.profile import plot_transect_line

        return plot_transect_line(
            self._obj,
            ax=ax,
            add_direction=add_direction,
            add_background=add_background,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            text_kwargs=text_kwargs,
            line_kwargs=line_kwargs,
            **common_kwargs,
        )

    @auto_wrap_docstring
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

    @auto_wrap_docstring
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

    @auto_wrap_docstring
    def plot_map_mesh(
        self,
        x=None,
        y=None,
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

    @auto_wrap_docstring
    def plot_map_mesh_centroids(
        self,
        x=None,
        y=None,
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
        from gpm.utils.xarray import get_dataset_variables

        return get_dataset_variables(self._obj, sort=True)

    @property
    def vertical_variables(self):
        from gpm.checks import get_vertical_variables

        return get_vertical_variables(self._obj)

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

    @property
    def bin_variables(self):
        from gpm.checks import get_bin_variables

        return get_bin_variables(self._obj)

    @auto_wrap_docstring
    def select_spatial_3d_variables(self, strict=False, squeeze=True):
        from gpm.utils.manipulations import select_spatial_3d_variables

        return select_spatial_3d_variables(self._obj, strict=strict, squeeze=squeeze)

    @auto_wrap_docstring
    def select_spatial_2d_variables(self, strict=False, squeeze=True):
        from gpm.utils.manipulations import select_spatial_2d_variables

        return select_spatial_2d_variables(self._obj, strict=strict, squeeze=squeeze)

    @auto_wrap_docstring
    def select_vertical_variables(self):
        from gpm.utils.manipulations import select_vertical_variables

        return select_vertical_variables(self._obj)

    @auto_wrap_docstring
    def select_frequency_variables(self):
        from gpm.utils.manipulations import select_frequency_variables

        return select_frequency_variables(self._obj)

    @auto_wrap_docstring
    def select_bin_variables(self):
        from gpm.utils.manipulations import select_bin_variables

        return select_bin_variables(self._obj)

    @auto_wrap_docstring
    def set_encoding(self, encoding_dict=None):
        from gpm.encoding.routines import set_encoding

        return set_encoding(self._obj, encoding_dict=encoding_dict)

    @auto_wrap_docstring
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

    @auto_wrap_docstring
    def plot_map(
        self,
        variable,
        ax=None,
        x=None,
        y=None,
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

        return plot_map(
            self._obj[variable],
            ax=ax,
            x=x,
            y=y,
            add_colorbar=add_colorbar,
            add_swath_lines=add_swath_lines,
            add_background=add_background,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )

    @auto_wrap_docstring
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

        return plot_image(
            self._obj[variable],
            ax=ax,
            x=x,
            y=y,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )

    @auto_wrap_docstring
    def plot_transect(
        self,
        variable,
        ax=None,
        x=None,
        y=None,
        add_colorbar=True,
        interpolation="nearest",
        zoom=True,
        fig_kwargs=None,
        cbar_kwargs=None,
        **plot_kwargs,
    ):
        from gpm.visualization.profile import plot_transect

        return plot_transect(
            self._obj[variable],
            ax=ax,
            x=x,
            y=y,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            zoom=zoom,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )

    @auto_wrap_docstring
    def available_retrievals(self):
        from gpm.retrievals.routines import available_retrievals

        return available_retrievals(self._obj)

    @auto_wrap_docstring
    def retrieve(self, name, **kwargs):
        from gpm.retrievals.routines import get_retrieval_variable

        return get_retrieval_variable(self._obj, name=name, **kwargs)

    @auto_wrap_docstring
    def slice_range_at_temperature(self, temperature, variable_temperature="airTemperature"):
        from gpm.utils.manipulations import slice_range_at_temperature

        return slice_range_at_temperature(
            self._obj,
            temperature=temperature,
            variable_temperature=variable_temperature,
        )

    @auto_wrap_docstring
    def extract_dataset_above_bin(self, bins, new_range_size=None, strict=False, reverse=False):
        from gpm.utils.manipulations import extract_dataset_above_bin

        return extract_dataset_above_bin(
            self._obj,
            bins=bins,
            new_range_size=new_range_size,
            strict=strict,
            reverse=reverse,
        )

    @auto_wrap_docstring
    def extract_dataset_below_bin(self, bins, new_range_size=None, strict=False, reverse=False):
        from gpm.utils.manipulations import extract_dataset_below_bin

        return extract_dataset_below_bin(
            self._obj,
            bins=bins,
            new_range_size=new_range_size,
            strict=strict,
            reverse=reverse,
        )

    @auto_wrap_docstring
    def extract_l2_dataset(self, bin_ellipsoid="binEllipsoid", new_range_size=None, shortened_range=True):
        from gpm.utils.manipulations import extract_l2_dataset

        return extract_l2_dataset(
            self._obj,
            bin_ellipsoid=bin_ellipsoid,
            new_range_size=new_range_size,
            shortened_range=shortened_range,
        )

    @auto_wrap_docstring
    def to_pandas_dataframe(self):
        from gpm.utils.dataframe import to_pandas_dataframe

        return to_pandas_dataframe(self._obj)

    @auto_wrap_docstring
    def to_dask_dataframe(self):
        from gpm.utils.dataframe import to_dask_dataframe

        return to_dask_dataframe(self._obj)


@xr.register_dataarray_accessor("gpm")
class GPM_DataArray_Accessor(GPM_Base_Accessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    @auto_wrap_docstring
    def get_slices_var_equals(self, dim, values, union=True, criteria="all"):
        from gpm.utils.checks import get_slices_var_equals

        return get_slices_var_equals(
            self._obj,
            dim=dim,
            values=values,
            union=union,
            criteria=criteria,
        )

    @auto_wrap_docstring
    def get_slices_var_between(self, dim, vmin=-np.inf, vmax=np.inf, criteria="all"):
        from gpm.utils.checks import get_slices_var_between

        return get_slices_var_between(self._obj, dim=dim, vmin=vmin, vmax=vmax, criteria=criteria)

    @auto_wrap_docstring
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

    @auto_wrap_docstring
    def plot_map(
        self,
        ax=None,
        x=None,
        y=None,
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

        return plot_map(
            self._obj,
            ax=ax,
            x=x,
            y=y,
            add_colorbar=add_colorbar,
            add_swath_lines=add_swath_lines,
            add_background=add_background,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )

    @auto_wrap_docstring
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

        return plot_image(
            self._obj,
            ax=ax,
            x=x,
            y=y,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )

    @auto_wrap_docstring
    def plot_transect(
        self,
        ax=None,
        x=None,
        y=None,
        add_colorbar=True,
        interpolation="nearest",
        zoom=True,
        fig_kwargs=None,
        cbar_kwargs=None,
        **plot_kwargs,
    ):
        from gpm.visualization.profile import plot_transect

        return plot_transect(
            self._obj,
            ax=ax,
            x=x,
            y=y,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            zoom=zoom,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )

    @auto_wrap_docstring
    def integrate_profile_concentration(self, name, scale_factor=None, units=None):
        from gpm.utils.manipulations import integrate_profile_concentration

        return integrate_profile_concentration(
            self._obj,
            name=name,
            scale_factor=scale_factor,
            units=units,
        )
