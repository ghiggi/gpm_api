import cartopy.crs as ccrs
import inspect
import os
import pytest
from matplotlib import image as mpl_image, pyplot as plt
import numpy as np
import tempfile
import xarray as xr


from gpm_api import _root_path
from gpm_api.visualization import plot


plots_dir_path = os.path.join(_root_path, "gpm_api", "tests", "data", "plots")
image_extension = ".png"
mse_tolerance = 5e-3


# Utils functions ##############################################################


def save_and_check_figure(
    name: str,
) -> None:
    """Save the current figure to a temporary location and compare it to the reference figure

    If the reference figure does not exist, it is created and the test is skipped.
    """

    # Save reference figure if it does not exist
    reference_path = os.path.join(plots_dir_path, name + image_extension)

    if not os.path.exists(reference_path):
        os.makedirs(plots_dir_path, exist_ok=True)
        plt.savefig(reference_path)
        pytest.skip(
            "Reference figure did not exist. Created it. To clone existing test data instead, run `git submodule update --init`."
        )

    # Save current figure to temporary file
    tmp_file = tempfile.NamedTemporaryFile(suffix=image_extension, delete=False)
    plt.savefig(tmp_file.name)

    # Compare reference and temporary file
    reference = mpl_image.imread(reference_path)
    tmp = mpl_image.imread(tmp_file.name)

    mse = np.mean((reference - tmp) ** 2)
    assert (
        mse < mse_tolerance
    ), f"Figure {tmp_file.name} is not the same as {name}{image_extension}. MSE {mse} > {mse_tolerance}"

    # Remove temporary file if comparison was successful
    tmp_file.close()
    os.remove(tmp_file.name)
    plt.close()


def get_test_name(
    class_instance=None,
) -> str:
    """Get a unique name for the calling function

    If the function is a method of a class, pass the class instance as argument (self).
    """

    # Add module name
    name_parts = [inspect.getmodulename(__file__)]

    # Add class name
    if class_instance is not None:
        name_parts.append(class_instance.__class__.__name__)

    # Add function name
    name_parts.append(inspect.stack()[1][3])

    return "-".join(name_parts)


# Tests ########################################################################


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

        plot.plot_map(orbit_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_antimeridian(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the antimeridian"""

        plot.plot_map(orbit_antimeridian_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_antimeridian_recenter(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the antimeridian with recentering"""

        crs_proj = ccrs.PlateCarree(central_longitude=180)
        plot.plot_map(orbit_antimeridian_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(get_test_name(self))

    def test_orbit_antimeridian_projection(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the antimeridian on orthographic projection"""

        crs_proj = ccrs.Orthographic(180, 0)
        plot.plot_map(orbit_antimeridian_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(get_test_name(self))

    def test_orbit_pole(
        self,
        orbit_pole_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the south pole"""

        plot.plot_map(orbit_pole_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_pole_projection(
        self,
        orbit_pole_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the south pole on orthographic projection"""

        crs_proj = ccrs.Orthographic(0, -30)
        plot.plot_map(orbit_pole_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(get_test_name(self))

    def test_orbit_nan_cross_track(
        self,
        orbit_nan_cross_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with NaN values at cross-track edges"""

        plot.plot_map(orbit_nan_cross_track_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_nan_along_track(
        self,
        orbit_nan_along_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with NaN values at along-track edges"""

        plot.plot_map(orbit_nan_along_track_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_nan_lon_cross_track(
        self,
        orbit_nan_lon_cross_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with some NaN longitudes cross-track"""

        with pytest.raises(ValueError):
            p = plot.plot_map(orbit_nan_lon_cross_track_dataarray)

    def test_orbit_nan_lon_along_track(
        self,
        orbit_nan_lon_along_track_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with some NaN latitudes along-track"""

        plot.plot_map(orbit_nan_lon_along_track_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_cbar_kwargs(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with colorbar keyword arguments"""

        cbar_kwargs = {"ticklabels": [42, 43, 44, 45, 46]}

        plot.plot_map(orbit_dataarray, cbar_kwargs=cbar_kwargs)
        save_and_check_figure(get_test_name(self))

    def test_orbit_rgb(
        self,
        orbit_rgb_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with RGB flag"""

        plot.plot_map(orbit_rgb_dataarray, rgb=True)
        save_and_check_figure(get_test_name(self))

    def test_orbit_rgb_invalid(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with RGB flag on non RGB data"""

        with pytest.raises(ValueError):
            plot.plot_map(orbit_dataarray, rgb=True)

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data"""

        plot.plot_map(grid_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_grid_antimeridian(
        self,
        grid_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data going over the antimeridian"""

        plot.plot_map(grid_antimeridian_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_grid_nan_lon(
        self,
        grid_nan_lon_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data with NaN longitudes"""

        plot.plot_map(grid_nan_lon_dataarray)
        save_and_check_figure(get_test_name(self))

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

        plot.plot_image(orbit_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data"""

        plot.plot_image(grid_dataarray)
        save_and_check_figure(get_test_name(self))

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

        plot.plot_map_mesh(orbit_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_antimeridian(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the antimeridian"""

        plot.plot_map_mesh(orbit_antimeridian_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_antimeridian_projection(
        self,
        orbit_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the antimeridian on orthographic projection"""

        crs_proj = ccrs.Orthographic(180, 0)
        plot.plot_map_mesh(orbit_antimeridian_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(get_test_name(self))

    def test_orbit_pole(
        self,
        orbit_pole_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the south pole"""

        plot.plot_map_mesh(orbit_pole_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_pole_projection(
        self,
        orbit_pole_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data going over the south pole on orthographic projection"""

        crs_proj = ccrs.Orthographic(0, -30)
        plot.plot_map_mesh(orbit_pole_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(get_test_name(self))

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data"""

        plot.plot_map_mesh(grid_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_grid_antimeridian(
        self,
        grid_antimeridian_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data going over the antimeridian"""

        plot.plot_map_mesh(grid_antimeridian_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_grid_nan_lon(
        self,
        grid_nan_lon_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting grid data with NaN longitudes"""

        plot.plot_map_mesh(grid_nan_lon_dataarray)
        save_and_check_figure(get_test_name(self))


def test_plot_map_mesh_centroids(
    orbit_dataarray: xr.DataArray,
) -> None:
    plot.plot_map_mesh_centroids(orbit_dataarray)
    save_and_check_figure(get_test_name())
