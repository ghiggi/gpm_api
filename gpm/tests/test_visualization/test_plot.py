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
import cartopy.crs as ccrs
import pytest
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr


import gpm.configs
from gpm.visualization import plot
from gpm.tests.test_visualization.utils import (
    expand_dims,
    get_test_name,
    save_and_check_figure,
)


def test_is_generator() -> None:
    """Test the _is_generator function"""

    def generator():
        yield 1

    assert plot.is_generator(generator())
    assert plot.is_generator((i for i in range(10)))
    assert not plot.is_generator([1, 2, 3])


def test_preprocess_figure_args() -> None:
    """Test the _preprocess_figure_args function"""

    nothing = {}
    something = {"": 0}

    # Test with ax None
    plot._preprocess_figure_args(None, fig_kwargs=nothing, subplot_kwargs=nothing)
    plot._preprocess_figure_args(None, fig_kwargs=something, subplot_kwargs=nothing)
    plot._preprocess_figure_args(None, fig_kwargs=nothing, subplot_kwargs=something)

    # Test with ax not None
    ax = plt.subplot()
    plot._preprocess_figure_args(ax, fig_kwargs=nothing, subplot_kwargs=nothing)

    with pytest.raises(ValueError):
        plot._preprocess_figure_args(ax, fig_kwargs=something, subplot_kwargs=nothing)

    with pytest.raises(ValueError):
        plot._preprocess_figure_args(ax, fig_kwargs=nothing, subplot_kwargs=something)


class TestGetExtent:
    """Test the get_extent function"""

    def test_orbit(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test getting the extent of orbit data"""

        returned_extent = plot.get_extent(orbit_dataarray)
        expected_extent = (-2.77663454, 22.65579744, -3.53830585, 18.64709521)
        np.testing.assert_allclose(returned_extent, expected_extent, rtol=1e-9)

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test getting the extent of grid data"""

        returned_extent = plot.get_extent(grid_dataarray)
        expected_extent = (-5, 20, -5, 15)
        assert returned_extent == expected_extent


def test_get_antimeridian_mask(
    orbit_antimeridian_dataarray: xr.DataArray,
) -> None:
    """Test the get_antimeridian_mask function"""

    lon = orbit_antimeridian_dataarray["lon"].data
    returned_mask = plot.get_antimeridian_mask(lon)
    # fmt: off
    expected_mask = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0,],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,],
    ], dtype=bool)
    np.testing.assert_array_equal(returned_mask, expected_mask)


class TestPlotMap:
    """Test the plot_map function"""

    def test_orbit(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data"""

        p = plot.plot_map(orbit_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_antimeridian(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the antimeridian"""

        p = plot.plot_map(orbit_antimeridian_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_antimeridian_recentered(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the antimeridian with recentering"""

        crs_proj = ccrs.PlateCarree(central_longitude=180)
        p = plot.plot_map(orbit_antimeridian_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_antimeridian_projection(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the antimeridian on orthographic projection"""

        crs_proj = ccrs.Orthographic(180, 0)
        p = plot.plot_map(orbit_antimeridian_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_antimeridian_not_masked_recentered(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the antimeridian without masking (recentered)"""

        crs_proj = ccrs.PlateCarree(central_longitude=180)
        with gpm.config.set({"viz_hide_antimeridian_data": False}):
            p = plot.plot_map(orbit_antimeridian_dataarray, subplot_kwargs={"projection": crs_proj})
            save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_pole(
        self,
        orbit_pole_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the south pole"""

        p = plot.plot_map(orbit_pole_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_pole_projection(
        self,
        orbit_pole_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the south pole on orthographic projection"""

        crs_proj = ccrs.Orthographic(0, -90)
        p = plot.plot_map(orbit_pole_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_nan_cross_track(
        self,
        orbit_nan_cross_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with NaN values at cross-track edges"""

        p = plot.plot_map(orbit_nan_cross_track_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_nan_along_track(
        self,
        orbit_nan_along_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with NaN values at along-track edges"""

        p = plot.plot_map(orbit_nan_along_track_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_nan_values(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with NaN values at one cell"""

        orbit_dataarray.data[2, 2] = np.nan
        p = plot.plot_map(orbit_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_nan_lon_outer_cross_track(
        self,
        orbit_nan_lon_cross_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with some NaN longitudes on the outer cross-track cells"""

        p = plot.plot_map(orbit_nan_lon_cross_track_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_nan_lon_along_track(
        self,
        orbit_nan_lon_along_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with some NaN latitudes along-track"""

        p = plot.plot_map(orbit_nan_lon_along_track_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_nan_lon_one_cell(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with NaN longitude at one cell"""

        orbit_dataarray["lon"].data[1, 3] = np.nan
        p = plot.plot_map(orbit_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_nan_lon_one_cell_centerline(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with NaN longitude at one cell on the cross-track centerline"""

        orbit_dataarray["lon"].data[2, 3] = np.nan
        p = plot.plot_map(orbit_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_cbar_kwargs(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with colorbar keyword arguments"""

        cbar_kwargs = {"ticklabels": [42, 43, 44, 45, 46]}

        p = plot.plot_map(orbit_dataarray, cbar_kwargs=cbar_kwargs)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_horizontal_colorbar(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with a horizontal colorbar"""

        cbar_kwargs = {"orientation": "horizontal"}
        p = plot.plot_map(orbit_dataarray, cbar_kwargs=cbar_kwargs)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_rgb(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit RGB data"""

        orbit_dataarray = expand_dims(orbit_dataarray, 3, dim="rgb", axis=2)
        p = plot.plot_map(orbit_dataarray, rgb=True)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_rgba(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit RGBA data"""

        orbit_dataarray = expand_dims(orbit_dataarray, 4, dim="rgb", axis=2)
        p = plot.plot_map(orbit_dataarray, rgb=True)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_rgb_antimeridian_recentered(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit RGB data going over the antimeridian without masking (recentered)"""

        orbit_dataarray = expand_dims(orbit_antimeridian_dataarray, 3, dim="rgb", axis=2)
        crs_proj = ccrs.PlateCarree(central_longitude=180)
        p = plot.plot_map(orbit_dataarray, subplot_kwargs={"projection": crs_proj}, rgb=True)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_rgb_antimeridian_not_masked_recentered(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit RGB data going over the antimeridian without masking (recentered)"""

        orbit_dataarray = expand_dims(orbit_antimeridian_dataarray, 3, dim="rgb", axis=2)
        crs_proj = ccrs.PlateCarree(central_longitude=180)
        with gpm.config.set({"viz_hide_antimeridian_data": False}):
            p = plot.plot_map(orbit_dataarray, subplot_kwargs={"projection": crs_proj}, rgb=True)
            save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_rgb_invalid(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with RGB flag on non RGB data"""

        with pytest.raises(ValueError):
            plot.plot_map(orbit_dataarray, rgb=True)

    def test_orbit_invalid_values(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with some invalid values"""

        orbit_dataarray.data[1:4, 1:4] = np.nan
        p = plot.plot_map(orbit_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data"""

        p = plot.plot_map(grid_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid_nan_lon(
        self,
        grid_nan_lon_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data with NaN longitudes"""

        p = plot.plot_map(grid_nan_lon_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid_time_dim(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data with time dimension"""

        grid_dataarray = expand_dims(grid_dataarray, 4, "time")
        with pytest.raises(ValueError):  # Expecting a 2D GPM field
            plot.plot_map(grid_dataarray)

    def test_invalid(
        self,
    ) -> None:
        """Test invalid data"""

        da = xr.DataArray()
        with pytest.raises(ValueError):
            plot.plot_map(da)


class TestPlotImage:
    """Test the plot_image function"""

    def test_orbit(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data"""

        p = plot.plot_image(orbit_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_cbar_kwargs(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with colorbar keyword arguments"""

        cbar_kwargs = {"ticklabels": [42, 43, 44, 45, 46]}

        p = plot.plot_image(orbit_dataarray, cbar_kwargs=cbar_kwargs)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_horizontal_colorbar(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with a horizontal colorbar"""

        cbar_kwargs = {"orientation": "horizontal"}
        p = plot.plot_image(orbit_dataarray, cbar_kwargs=cbar_kwargs)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_no_cbar(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data without colorbar"""

        p = plot.plot_image(orbit_dataarray, add_colorbar=False)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data"""

        p = plot.plot_image(grid_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_invalid(
        self,
    ) -> None:
        """Test invalid data"""

        da = xr.DataArray()
        with pytest.raises(ValueError):
            plot.plot_image(da)


class TestPlotMapMesh:
    """Test the plot_map_mesh function"""

    def test_orbit(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data"""

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
        """Test plotting orbit data going over the antimeridian on orthographic projection"""

        crs_proj = ccrs.Orthographic(180, 0)
        p = plot.plot_map_mesh(
            orbit_antimeridian_dataarray, subplot_kwargs={"projection": crs_proj}
        )
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_pole(
        self,
        orbit_pole_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the south pole"""

        p = plot.plot_map_mesh(orbit_pole_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_pole_projection(
        self,
        orbit_pole_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the south pole on orthographic projection"""

        crs_proj = ccrs.Orthographic(0, -90)
        p = plot.plot_map_mesh(orbit_pole_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data"""

        p = plot.plot_map_mesh(grid_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid_nan_lon(
        self,
        grid_nan_lon_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data with NaN longitudes"""

        p = plot.plot_map_mesh(grid_nan_lon_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())


class TestPlotMapMeshCentroids:
    """Test the plot_map_mesh_centroids function"""

    def test_orbit(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data"""

        p = plot.plot_map_mesh_centroids(orbit_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data"""

        p = plot.plot_map_mesh_centroids(grid_dataarray)
        save_and_check_figure(figure=p.figure, name=get_test_name())


class TestPlotLabels:
    """Test the plot_labels function"""

    label_name = "label"

    @pytest.fixture
    def orbit_labels_dataarray(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> xr.DataArray:
        """Create an orbit data array with label coordinates"""

        labels = np.random.randint(0, 10, orbit_dataarray.shape)
        orbit_dataarray = orbit_dataarray.assign_coords(
            {self.label_name: (("cross_track", "along_track"), labels)}
        )
        return orbit_dataarray

    @pytest.fixture
    def grid_labels_dataarray(
        self,
        grid_dataarray: xr.DataArray,
    ) -> xr.DataArray:
        """Create a grid data array with label coordinates"""

        labels = np.random.randint(0, 10, grid_dataarray.shape)
        grid_dataarray = grid_dataarray.assign_coords({self.label_name: (("lat", "lon"), labels)})
        return grid_dataarray

    def test_orbit(
        self,
        orbit_labels_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data"""

        p = plot.plot_labels(orbit_labels_dataarray, label_name=self.label_name)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_dataset(
        self,
        orbit_labels_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data from a dataset"""

        ds = xr.Dataset({self.label_name: orbit_labels_dataarray[self.label_name]})
        p = plot.plot_labels(ds, label_name=self.label_name)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_labels_dataarray(
        self,
        orbit_labels_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data from labels data array directly"""

        p = plot.plot_labels(orbit_labels_dataarray[self.label_name])
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_orbit_exceed_labels(
        self,
        orbit_labels_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with too many labels for colorbar"""

        p = plot.plot_labels(orbit_labels_dataarray, label_name=self.label_name, max_n_labels=5)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_grid(
        self,
        grid_labels_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data"""

        p = plot.plot_labels(grid_labels_dataarray, label_name=self.label_name)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_generator(
        self,
        orbit_labels_dataarray: xr.DataArray,
        prevent_pyplot_show: None,
    ) -> None:
        """Test plotting orbit data form a generator"""

        da_list = [
            (0, orbit_labels_dataarray),
            (1, orbit_labels_dataarray),
        ]
        generator = (t for t in da_list)

        p = plot.plot_labels(generator, label_name=self.label_name)  # only last plot is returned
        save_and_check_figure(figure=p.figure, name=get_test_name())


class TestPlotPatches:
    """Test the plot_patches function"""

    def test_orbit(
        self,
        orbit_dataarray: xr.DataArray,
        prevent_pyplot_show: None,
    ) -> None:
        """Test plotting orbit data"""

        da_list = [
            (0, orbit_dataarray),
            (1, orbit_dataarray),
        ]
        generator = (t for t in da_list)
        plot.plot_patches(generator)  # does not return plotter
        save_and_check_figure(name=get_test_name())

    def test_orbit_dataset(
        self,
        orbit_dataarray: xr.DataArray,
        prevent_pyplot_show: None,
    ) -> None:
        """Test plotting orbit data from a dataset"""

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

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
        prevent_pyplot_show: None,
    ) -> None:
        """Test plotting grid data"""

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
        """Test invalid data"""

        invalid_list = [
            (0, xr.DataArray()),
            (1, xr.DataArray()),
        ]
        generator = (t for t in invalid_list)
        plot.plot_patches(generator)  # passes without error
