import inspect
import os
import pytest
from matplotlib import image as mpl_image, pyplot as plt
import numpy as np
import tempfile
import traceback
import xarray as xr


from gpm_api import _root_path
from gpm_api.visualization import plot


plots_dir_path = os.path.join(_root_path, "gpm_api", "tests", "data", "plots")
image_extension = ".png"
np.random.seed(0)


# Utils function and fixtures ##################################################


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


@pytest.fixture(scope="function")
def orbit_dataarray() -> xr.DataArray:
    n_cross_track = 5
    n_along_track = 10
    cross_track = np.arange(n_cross_track)
    along_track = np.arange(n_along_track)
    data = np.random.rand(n_cross_track, n_along_track)
    granule_id = np.zeros(n_along_track)

    # Coordinates
    lon = np.linspace(-50, 50, n_along_track)
    lat = np.linspace(-30, 30, n_along_track)

    # Add cross track dimension
    lon = np.tile(lon, (n_cross_track, 1))
    lat = np.tile(lat, (n_cross_track, 1))

    # Create data array
    da = xr.DataArray(data, coords={"cross_track": cross_track, "along_track": along_track})
    da.coords["lat"] = (("cross_track", "along_track"), lat)
    da.coords["lon"] = (("cross_track", "along_track"), lon)
    da.coords["gpm_granule_id"] = ("along_track", granule_id)

    return da


@pytest.fixture(scope="function")
def grid_dataarray() -> xr.DataArray:
    lon = np.arange(-50, 51, 10)
    lat = np.arange(-50, 51, 10)
    data = np.random.rand(len(lat), len(lon))

    # Create data array
    da = xr.DataArray(data, coords={"lat": lat, "lon": lon})

    return da


# Tests ########################################################################


def test_plot_map() -> None:
    pass  # TODO


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


def test_plot_map_mesh() -> None:
    pass  # TODO


def test_plot_map_mesh_centroids() -> None:
    pass  # TODO
