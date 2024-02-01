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
        pytest.skip("Reference figure did not exist. Created it.")

    # Save current figure to temporary file
    with tempfile.NamedTemporaryFile(suffix=image_extension) as tmp_file:
        plt.savefig(tmp_file.name)

        # Compare reference and temporary file
        reference = mpl_image.imread(reference_path)
        tmp = mpl_image.imread(tmp_file.name)

        assert np.allclose(reference, tmp), f"Figure {name}{image_extension} is not the same"


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


class TestPlotMap:
    """Test the plot_map function"""

    def test_orbit(
        self,
        orbit_dataarray: xr.Dataset,
    ) -> None:
        """Test plotting orbit data"""

        plot.plot_map(orbit_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_antimeridian(
        self,
        orbit_antimeridian_dataarray: xr.Dataset,
    ) -> None:
        """Test plotting orbit data going over the antimeridian"""

        plot.plot_map(orbit_antimeridian_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_antimeridian_projection(
        self,
        orbit_antimeridian_dataarray: xr.Dataset,
    ) -> None:
        """Test plotting orbit data going over the antimeridian on orthographic projection"""

        crs_proj = ccrs.Orthographic(180, 0)
        plot.plot_map(orbit_antimeridian_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(get_test_name(self))

    def test_orbit_pole(
        self,
        orbit_pole_dataarray: xr.Dataset,
    ) -> None:
        """Test plotting orbit data going over the south pole"""

        plot.plot_map(orbit_pole_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_pole_projection(
        self,
        orbit_pole_dataarray: xr.Dataset,
    ) -> None:
        """Test plotting orbit data going over the south pole on orthographic projection"""

        crs_proj = ccrs.Orthographic(0, -30)
        plot.plot_map(orbit_pole_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(get_test_name(self))

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
        orbit_dataarray: xr.Dataset,
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
        orbit_dataarray: xr.Dataset,
    ) -> None:
        """Test plotting orbit data"""

        plot.plot_map_mesh(orbit_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_antimeridian(
        self,
        orbit_antimeridian_dataarray: xr.Dataset,
    ) -> None:
        """Test plotting orbit data going over the antimeridian"""

        plot.plot_map_mesh(orbit_antimeridian_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_antimeridian_projection(
        self,
        orbit_antimeridian_dataarray: xr.Dataset,
    ) -> None:
        """Test plotting orbit data going over the antimeridian on orthographic projection"""

        crs_proj = ccrs.Orthographic(180, 0)
        plot.plot_map_mesh(orbit_antimeridian_dataarray, subplot_kwargs={"projection": crs_proj})
        save_and_check_figure(get_test_name(self))

    def test_orbit_pole(
        self,
        orbit_pole_dataarray: xr.Dataset,
    ) -> None:
        """Test plotting orbit data going over the south pole"""

        plot.plot_map_mesh(orbit_pole_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_pole_projection(
        self,
        orbit_pole_dataarray: xr.Dataset,
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


def test_plot_map_mesh_centroids(
    orbit_dataarray: xr.Dataset,
) -> None:
    plot.plot_map_mesh_centroids(orbit_dataarray)
    save_and_check_figure(get_test_name())
