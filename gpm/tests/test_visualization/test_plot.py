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
import platform

import cartopy.crs as ccrs
import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt

import gpm.configs
from gpm.tests.test_visualization.utils import (
    expand_dims,
    get_test_name,
    save_and_check_figure,
    skip_tests_if_no_data,
)
from gpm.visualization import plot
from gpm.visualization.plot import add_map_inset

# Fixtures imported from gpm.tests.conftest:
# - orbit_dataarray
# - orbit_antimeridian_dataarray
# - orbit_pole_dataarray
# - orbit_nan_cross_track_dataarray
# - orbit_nan_along_track_dataarray
# - orbit_nan_lon_cross_track_dataarray
# - orbit_nan_lon_along_track_dataarray
# - grid_dataarray
# - grid_nan_lon_dataarray


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Figure comparison is skipped because of minor differences against Linux.",
)
skip_tests_if_no_data()


def test_is_generator() -> None:
    """Test the _is_generator function."""

    def generator():
        yield 1

    assert plot.is_generator(generator())
    assert plot.is_generator(i for i in range(10))
    assert not plot.is_generator([1, 2, 3])


def test_preprocess_figure_args() -> None:
    """Test the preprocess_figure_args function."""
    nothing = {}
    something = {"": 0}

    # Test with ax None
    _ = plot.preprocess_figure_args(None, fig_kwargs=nothing, subplot_kwargs=nothing)
    _ = plot.preprocess_figure_args(None, fig_kwargs=something, subplot_kwargs=nothing)
    _ = plot.preprocess_figure_args(None, fig_kwargs=nothing, subplot_kwargs=something)

    # Test with ax not None
    ax = plt.subplot()
    _ = plot.preprocess_figure_args(ax, fig_kwargs=nothing, subplot_kwargs=nothing)

    with pytest.raises(ValueError):
        plot.preprocess_figure_args(ax, fig_kwargs=something, subplot_kwargs=nothing)

    with pytest.raises(ValueError):
        plot.preprocess_figure_args(ax, fig_kwargs=nothing, subplot_kwargs=something)


def test_get_antimeridian_mask(
    orbit_antimeridian_dataarray: xr.DataArray,
) -> None:
    """Test the get_antimeridian_mask function."""
    lon = orbit_antimeridian_dataarray["lon"].data
    returned_mask = plot.get_antimeridian_mask(lon)
    # fmt: off
    expected_mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    ], dtype=bool)
    np.testing.assert_array_equal(returned_mask, expected_mask)


class TestPlotMap:
    """Test the plot_map function."""

    def test_orbit(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data."""
        p = plot.plot_map(orbit_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_antimeridian_not_recentered(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the antimeridian."""
        p = plot.plot_map(orbit_antimeridian_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_antimeridian(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the antimeridian with recentering."""
        crs_proj = ccrs.PlateCarree(central_longitude=180)
        cmap = "Spectral"  # check that bad is set to 0 to avoid cartopy bug
        p = plot.plot_map(orbit_antimeridian_dataarray, cmap=cmap, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_antimeridian_with_nan_coordinates(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the antimeridian and with masked values due to nan coordinates."""
        orbit_antimeridian_dataarray["lon"].data[1, 2:6] = np.nan
        crs_proj = ccrs.PlateCarree(central_longitude=180)
        p = plot.plot_map(orbit_antimeridian_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_antimeridian_projection(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the antimeridian on orthographic projection."""
        crs_proj = ccrs.Orthographic(180, 0)
        p = plot.plot_map(orbit_antimeridian_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_antimeridian_not_masked(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the antimeridian without masking (recentered)."""
        crs_proj = ccrs.PlateCarree(central_longitude=180)
        with gpm.config.set({"viz_hide_antimeridian_data": False}):
            p = plot.plot_map(orbit_antimeridian_dataarray, subplot_kwargs={"projection": crs_proj})
            save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_pole(
        self,
        orbit_pole_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the south pole."""
        p = plot.plot_map(orbit_pole_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_pole_projection(
        self,
        orbit_pole_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the south pole on orthographic projection."""
        crs_proj = ccrs.Orthographic(0, -90)
        p = plot.plot_map(orbit_pole_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_xy_dim(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with x and y dimensions."""
        p = plot.plot_map(orbit_dataarray.rename({"cross_track": "y", "along_track": "x"}))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_longitude_latitude_coords(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with longitude and latitude  coordinates."""
        p = plot.plot_map(orbit_dataarray.rename({"lon": "longitude", "lat": "latitude"}))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    ####------------------------------------------------------------------------
    #### - Test with NaN in the data

    def test_orbit_data_nan_cross_track(
        self,
        orbit_data_nan_cross_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit with NaN values in the data at cross-track edges."""
        p = plot.plot_map(orbit_data_nan_cross_track_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_data_nan_along_track(
        self,
        orbit_data_nan_along_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit with NaN values in the data at along-track edges."""
        p = plot.plot_map(orbit_data_nan_along_track_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_data_nan_values(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit with NaN values in the data at one cell."""
        orbit_dataarray.data[2, 2] = np.nan
        p = plot.plot_map(orbit_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_data_all_nan_values(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with some invalid values."""
        orbit_dataarray.data[1:4, 1:4] = np.nan
        p = plot.plot_map(orbit_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    ####------------------------------------------------------------------------
    #### - Test with NaN coordinates

    def test_orbit_nan_coordinate_at_one_cell(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with NaN coordinates at one cell."""
        # NOTE: here we test linear interpolation of coordinates
        orbit_dataarray["lon"].data[1, 3] = np.nan
        orbit_dataarray["lat"].data[1, 3] = np.nan
        p = plot.plot_map(orbit_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    @pytest.mark.skipif(platform.system() == "Windows", reason="Minor figure difference on Windows")
    def test_orbit_nan_coordinate_at_corners(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with NaN coordinates at the swath corners."""
        # NOTE: here we test nearest neighbour interpolation of coordinates
        orbit_dataarray["lon"].data[0:2, 0:2] = np.nan
        orbit_dataarray["lat"].data[0:2, 0:2] = np.nan
        p = plot.plot_map(orbit_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_nan_coordinate_at_cross_track_center(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with NaN coordinates at one cell on the cross-track centerline."""
        # NOTE: centerline used to identify contiguoues slices. So here assumed not contiguous.
        orbit_dataarray["lon"].data[2, 3] = np.nan
        orbit_dataarray["lat"].data[2, 3] = np.nan
        p = plot.plot_map(orbit_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_nan_outer_cross_track(
        self,
        orbit_nan_outer_cross_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with some NaN coordinates on the outer cross-track cells."""
        p = plot.plot_map(orbit_nan_outer_cross_track_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_nan_inner_cross_track(
        self,
        orbit_nan_inner_cross_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with all NaN coordinates on some inner cross-track cells."""
        p = plot.plot_map(orbit_nan_inner_cross_track_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_nan_slice_along_track(
        self,
        orbit_nan_slice_along_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with some NaN latitudes along-track."""
        p = plot.plot_map(orbit_nan_slice_along_track_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    ####------------------------------------------------------------------------
    #### - Test cross-track/along-track view

    def test_orbit_cross_track(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data."""
        p = plot.plot_map(orbit_dataarray.isel(along_track=0))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_along_track(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data."""
        p = plot.plot_map(orbit_dataarray.isel(cross_track=0))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_cross_track_pole_projection(
        self,
        orbit_pole_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit cross-track view going over the south pole on orthographic projection."""
        crs_proj = ccrs.Orthographic(0, -90)
        p = plot.plot_map(orbit_pole_dataarray.isel(cross_track=0), subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_along_track_pole_projection(
        self,
        orbit_pole_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit along-track view going over the south pole on orthographic projection."""
        crs_proj = ccrs.Orthographic(0, -90)
        p = plot.plot_map(orbit_pole_dataarray.isel(along_track=0), subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_cross_track_nan_outer_cross_track(
        self,
        orbit_nan_outer_cross_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit cross-track view with some NaN coordinates on the outer cross-track cells."""
        p = plot.plot_map(orbit_nan_outer_cross_track_dataarray.isel(along_track=0))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_cross_track_nan_inner_cross_track(
        self,
        orbit_nan_inner_cross_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit cross-track view with all NaN coordinates on some inner cross-track cells."""
        p = plot.plot_map(orbit_nan_inner_cross_track_dataarray.isel(along_track=0))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_along_track_nan_slice_along_track(
        self,
        orbit_nan_slice_along_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit along-track view with some NaN latitudes along-track."""
        p = plot.plot_map(orbit_nan_slice_along_track_dataarray.isel(cross_track=0))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    ####------------------------------------------------------------------------
    #### - Test RGB options

    def test_orbit_rgb(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit RGB data."""
        orbit_dataarray = expand_dims(orbit_dataarray, 3, dim="rgb", axis=2)
        p = plot.plot_map(orbit_dataarray, rgb="rgb")
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_rgba(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit RGBA data."""
        orbit_dataarray = expand_dims(orbit_dataarray, 4, dim="rgb", axis=2)
        p = plot.plot_map(orbit_dataarray, rgb="rgb")
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_rgb_alpha_value(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit RGB data with alpha value."""
        orbit_dataarray = expand_dims(orbit_dataarray, 3, dim="rgb", axis=2)
        p = plot.plot_map(orbit_dataarray, rgb="rgb", alpha=0.4)
        save_and_check_figure(figure=p.figure, name=get_test_name())

        # NOTE: alpha array not allowed currently

    def test_orbit_rgb_with_nan_coordinates(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit RGB data with some NaN coordinates (and thus masked pixels)."""
        orbit_dataarray_rgb = expand_dims(orbit_dataarray, 3, dim="rgb", axis=2)
        orbit_dataarray_rgb["lon"].data[3, 5:15] = np.nan
        p = plot.plot_map(orbit_dataarray_rgb, rgb="rgb")
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_rgb_antimeridian(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit RGB data going over the antimeridian without masking (recentered)."""
        orbit_dataarray_rgb = expand_dims(orbit_antimeridian_dataarray, 3, dim="rgb", axis=2)
        crs_proj = ccrs.PlateCarree(central_longitude=180)
        p = plot.plot_map(orbit_dataarray_rgb, subplot_kwargs={"projection": crs_proj}, rgb="rgb")
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_rgb_antimeridian_with_nan_coordinates(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit RGB data going over the antimeridian and with masked values due to nan coordinates."""
        orbit_dataarray_rgb = expand_dims(orbit_antimeridian_dataarray, 3, dim="rgb", axis=2)
        orbit_dataarray_rgb["lon"].data[1, 2:6] = np.nan
        crs_proj = ccrs.PlateCarree(central_longitude=180)
        p = plot.plot_map(orbit_dataarray_rgb, subplot_kwargs={"projection": crs_proj}, rgb="rgb")
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_rgb_antimeridian_not_masked(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit RGB data going over the antimeridian without masking (recentered)."""
        orbit_dataarray_rgb = expand_dims(orbit_antimeridian_dataarray, 3, dim="rgb", axis=2)
        crs_proj = ccrs.PlateCarree(central_longitude=180)
        with gpm.config.set({"viz_hide_antimeridian_data": False}):
            p = plot.plot_map(orbit_dataarray_rgb, subplot_kwargs={"projection": crs_proj}, rgb="rgb")
            save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_rgb_invalid(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with RGB flag on non RGB data."""
        with pytest.raises(ValueError):
            plot.plot_map(orbit_dataarray, rgb="rgb")

    ####------------------------------------------------------------------------
    #### Test colorbar kwargs

    def test_orbit_alpha_value(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit with alpha value."""
        p = plot.plot_map(orbit_dataarray, alpha=0.4)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_alpha_array(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit with alpha array."""
        alpha_arr = np.ones(orbit_dataarray.shape) * np.linspace(0, 1, orbit_dataarray.shape[1])
        p = plot.plot_map(orbit_dataarray, alpha=alpha_arr)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_alpha_nan_outer_cross_track(
        self,
        orbit_nan_outer_cross_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with some NaN coordinates on the outer cross-track cells and alpha array."""
        alpha_arr = np.ones(orbit_nan_outer_cross_track_dataarray.shape)
        alpha_arr = alpha_arr * np.linspace(0, 1, orbit_nan_outer_cross_track_dataarray.shape[1])
        p = plot.plot_map(orbit_nan_outer_cross_track_dataarray, alpha=alpha_arr)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_cbar_kwargs(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with colorbar keyword arguments."""
        cbar_kwargs = {"ticks": [0.1, 0.2, 0.4, 0.6, 0.8], "ticklabels": [42.5, 43, 44, 45, 46]}

        p = plot.plot_map(orbit_dataarray, cbar_kwargs=cbar_kwargs)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_horizontal_colorbar(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with a horizontal colorbar."""
        cbar_kwargs = {"orientation": "horizontal"}
        p = plot.plot_map(orbit_dataarray, cbar_kwargs=cbar_kwargs)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    ####------------------------------------------------------------------------
    #### Test GRID options

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data."""
        p = plot.plot_map(grid_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid_nan_lon(
        self,
        grid_nan_lon_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data with NaN longitudes."""
        p = plot.plot_map(grid_nan_lon_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid_time_dim(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data with time dimension."""
        grid_dataarray = expand_dims(grid_dataarray, 4, "time")
        with pytest.raises(ValueError):  # Expecting a 2D GPM field
            plot.plot_map(grid_dataarray)

    def test_grid_xy_dim(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data with x and y dimensions."""
        p = plot.plot_map(grid_dataarray.rename({"lat": "y", "lon": "x"}))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid_longitude_latitude_coords(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data with longitude and latitude  coordinates."""
        p = plot.plot_map(grid_dataarray.rename({"lon": "longitude", "lat": "latitude"}))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_invalid(
        self,
    ) -> None:
        """Test invalid data."""
        da = xr.DataArray()
        with pytest.raises(ValueError):
            plot.plot_map(da)

    ####------------------------------------------------------------------------
    #### Test map inset options
    def test_add_map_inset(self, orbit_dataarray: xr.DataArray):
        """Test the add_map_inset function."""
        p = plot.plot_map(orbit_dataarray)
        add_map_inset(
            ax=p.axes,
            loc="upper left",
            inset_height=0.2,
            projection=None,
            inside_figure=True,
            border_pad=0.02,
        )
        save_and_check_figure(figure=p.figure, name=get_test_name())


class TestPlotImage:
    """Test the plot_image function."""

    def test_orbit(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data."""
        p = plot.plot_image(orbit_dataarray.drop_vars(["lon", "lat"]))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_without_coords(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data without coordinates."""
        p = plot.plot_image(orbit_dataarray.drop_vars(["lon", "lat"]))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_with_xy_dims(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with x and y dimensions."""
        p = plot.plot_image(orbit_dataarray.rename({"cross_track": "y", "along_track": "x"}))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_alpha_array(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit with alpha array."""
        alpha_arr = np.ones(orbit_dataarray.shape) * np.linspace(0, 1, orbit_dataarray.shape[1])
        p = plot.plot_image(orbit_dataarray, alpha=alpha_arr)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_with_fully_transparent_cbar(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with invisible colorbar."""
        p = plot.plot_image(orbit_dataarray, visible_colorbar=False)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_without_cbar(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data without colorbar."""
        p = plot.plot_image(orbit_dataarray, add_colorbar=False)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_cbar_kwargs(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with colorbar keyword arguments."""
        cbar_kwargs = {"ticks": [0.1, 0.2, 0.4, 0.6, 0.8], "ticklabels": [42.5, 43, 44, 45, 46]}

        p = plot.plot_image(orbit_dataarray, cbar_kwargs=cbar_kwargs)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_horizontal_colorbar(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with a horizontal colorbar."""
        cbar_kwargs = {"orientation": "horizontal"}
        p = plot.plot_image(orbit_dataarray, cbar_kwargs=cbar_kwargs)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_no_cbar(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data without colorbar."""
        p = plot.plot_image(orbit_dataarray, add_colorbar=False)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data."""
        p = plot.plot_image(grid_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid_without_coords(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data without coordinates."""
        p = plot.plot_image(grid_dataarray.drop_vars(["lon", "lat"]))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid_with_xy_dims(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data with x and y dimensions."""
        p = plot.plot_image(grid_dataarray.rename({"lat": "y", "lon": "x"}))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid_with_longitude_latitude_coords(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data with x and y dimensions."""
        p = plot.plot_image(grid_dataarray.rename({"lat": "latitude", "lon": "longitude"}))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_invalid(
        self,
    ) -> None:
        """Test invalid data."""
        da = xr.DataArray()
        with pytest.raises(ValueError):
            plot.plot_image(da)


class TestPlotMapMesh:
    """Test the plot_map_mesh function."""

    def test_orbit(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data."""
        p = plot.plot_map_mesh(orbit_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    # def test_orbit_antimeridian(  # Does not work, issue in cartopy
    #     self,
    #     orbit_antimeridian_dataarray: xr.DataArray,
    # ) -> None:
    #     """Test plotting orbit data going over the antimeridian"""

    #     p = plot.plot_map_mesh(orbit_antimeridian_dataarray)
    #     save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_antimeridian_projection(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the antimeridian on orthographic projection."""
        crs_proj = ccrs.Orthographic(180, 0)
        p = plot.plot_map_mesh(
            orbit_antimeridian_dataarray,
            subplot_kwargs={"projection": crs_proj},
        )
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_pole(
        self,
        orbit_pole_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the south pole."""
        p = plot.plot_map_mesh(orbit_pole_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_pole_projection(
        self,
        orbit_pole_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the south pole on orthographic projection."""
        crs_proj = ccrs.Orthographic(0, -90)
        p = plot.plot_map_mesh(orbit_pole_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data."""
        p = plot.plot_map_mesh(grid_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid_nan_lon(
        self,
        grid_nan_lon_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data with NaN longitudes."""
        p = plot.plot_map_mesh(grid_nan_lon_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())


class TestPlotMapMeshCentroids:
    """Test the plot_map_mesh_centroids function."""

    def test_orbit(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data."""
        p = plot.plot_map_mesh_centroids(orbit_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data."""
        p = plot.plot_map_mesh_centroids(grid_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())


class TestPlotLabels:
    """Test the plot_labels function."""

    label_name = "label"

    @pytest.fixture
    def orbit_labels_dataarray(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> xr.DataArray:
        """Create an orbit data array with label coordinates."""
        rng = np.random.default_rng(seed=0)
        labels = rng.integers(0, 10, size=orbit_dataarray.shape)
        return orbit_dataarray.assign_coords(
            {self.label_name: (("cross_track", "along_track"), labels)},
        )

    @pytest.fixture
    def grid_labels_dataarray(
        self,
        grid_dataarray: xr.DataArray,
    ) -> xr.DataArray:
        """Create a grid data array with label coordinates."""
        rng = np.random.default_rng(seed=0)
        labels = rng.integers(0, 10, size=grid_dataarray.shape)
        return grid_dataarray.assign_coords({self.label_name: (("lat", "lon"), labels)})

    def test_orbit(
        self,
        orbit_labels_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data."""
        p = plot.plot_labels(orbit_labels_dataarray, label_name=self.label_name)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_dataset(
        self,
        orbit_labels_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data from a dataset."""
        ds = xr.Dataset({self.label_name: orbit_labels_dataarray[self.label_name]})
        p = plot.plot_labels(ds, label_name=self.label_name)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_labels_dataarray(
        self,
        orbit_labels_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data from labels data array directly."""
        p = plot.plot_labels(orbit_labels_dataarray[self.label_name])
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_exceed_labels(
        self,
        orbit_labels_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with too many labels for colorbar."""
        p = plot.plot_labels(orbit_labels_dataarray, label_name=self.label_name, max_n_labels=5)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid(
        self,
        grid_labels_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data."""
        p = plot.plot_labels(grid_labels_dataarray, label_name=self.label_name)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    @pytest.mark.usefixtures("_prevent_pyplot_show")
    def test_generator(
        self,
        orbit_labels_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data form a generator."""
        da_list = [
            (0, orbit_labels_dataarray),
            (1, orbit_labels_dataarray),
        ]
        generator = (t for t in da_list)

        p = plot.plot_labels(generator, label_name=self.label_name)  # only last plot is returned
        save_and_check_figure(figure=p.figure, name=get_test_name())


class TestPlotPatches:
    """Test the plot_patches function."""

    @pytest.mark.usefixtures("_prevent_pyplot_show")
    def test_orbit(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data."""
        da_list = [
            (0, orbit_dataarray),
            (1, orbit_dataarray),
        ]
        generator = (t for t in da_list)
        plot.plot_patches(generator)  # does not return plotter
        save_and_check_figure(name=get_test_name())

    @pytest.mark.usefixtures("_prevent_pyplot_show")
    def test_orbit_dataset(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data from a dataset."""
        variable_name = "variable"
        ds = xr.Dataset({variable_name: orbit_dataarray})
        ds_list = [
            (0, ds),
            (1, ds),
        ]
        generator = (t for t in ds_list)

        # Test with missing variable
        with pytest.raises(ValueError):
            plot.plot_patches(generator)

        plot.plot_patches(generator, variable=variable_name)  # does not return plotter
        save_and_check_figure(name=get_test_name())

    @pytest.mark.usefixtures("_prevent_pyplot_show")
    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data."""
        da_list = [
            (0, grid_dataarray),
            (1, grid_dataarray),
        ]
        generator = (t for t in da_list)
        plot.plot_patches(generator)  # does not return plotter
        save_and_check_figure(name=get_test_name())

    def test_invalid(
        self,
    ) -> None:
        """Test invalid data."""
        invalid_list = [
            (0, xr.DataArray()),
            (1, xr.DataArray()),
        ]
        generator = (t for t in invalid_list)
        plot.plot_patches(generator)  # passes without error
