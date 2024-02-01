import cartopy.crs as ccrs
import inspect
import os
import pyproj
import pytest
from matplotlib import image as mpl_image, pyplot as plt
import numpy as np
import tempfile
from typing import Tuple
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


def get_geodesic_path(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    n_points: int,
    offset_distance: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute geodesic path between starting and ending coordinates"""

    geod = pyproj.Geod(ellps="sphere")
    r = geod.inv_intermediate(
        start_lon,
        start_lat,
        end_lon,
        end_lat,
        n_points,
        initial_idx=0,
        terminus_idx=0,
        return_back_azimuth=False,
        flags=pyproj.enums.GeodIntermediateFlag.AZIS_KEEP,
    )

    orthogonal_directions = np.array(r.azis) + 90

    if offset_distance != 0:
        geod.fwd(r.lons, r.lats, orthogonal_directions, [offset_distance] * n_points, inplace=True)

    # Convert into numpy arrays
    lon = np.array(r.lons)
    lat = np.array(r.lats)

    return lon, lat


def get_geodesic_band(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    width: float,
    n_along_track: int,
    n_cross_track: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute coordinates of geodesic band"""

    lon_lines = []
    lat_lines = []
    offsets = np.linspace(-width / 2, width / 2, n_cross_track)

    for offset in offsets:
        lon_line, lat_line = get_geodesic_path(
            start_lon=start_lon,
            start_lat=start_lat,
            end_lon=end_lon,
            end_lat=end_lat,
            n_points=n_along_track,
            offset_distance=offset,
        )
        lon_lines.append(lon_line)
        lat_lines.append(lat_line)

    lon = np.stack(lon_lines)
    lat = np.stack(lat_lines)

    return lon, lat


def get_orbit_dataarray(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    width: float,
    n_along_track: int,
    n_cross_track: int,
) -> xr.DataArray:
    """Create orbit data array on geodesic band"""

    np.random.seed(0)
    cross_track = np.arange(n_cross_track)
    along_track = np.arange(n_along_track)
    data = np.random.rand(n_cross_track, n_along_track)
    granule_id = np.zeros(n_along_track)

    # Coordinates
    lon, lat = get_geodesic_band(
        start_lon=start_lon,
        start_lat=start_lat,
        end_lon=end_lon,
        end_lat=end_lat,
        width=width,
        n_along_track=n_along_track,
        n_cross_track=n_cross_track,
    )

    # Create data array
    da = xr.DataArray(data, coords={"cross_track": cross_track, "along_track": along_track})
    da.coords["lat"] = (("cross_track", "along_track"), lat)
    da.coords["lon"] = (("cross_track", "along_track"), lon)
    da.coords["gpm_granule_id"] = ("along_track", granule_id)

    return da


def get_grid_dataarray(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    n_lon: int,
    n_lat: int,
) -> xr.DataArray:
    """Create grid data array"""

    np.random.seed(0)
    lon = np.linspace(start_lon, end_lon, n_lon)
    lat = np.linspace(start_lat, end_lat, n_lat)
    data = np.random.rand(n_lat, n_lon)

    # Create data array
    da = xr.DataArray(data, coords={"lat": lat, "lon": lon})

    return da


# Fixtures #####################################################################


@pytest.fixture(scope="function")
def orbit_dataarray() -> xr.DataArray:
    """Create orbit data array near 0 longitude and latitude"""

    return get_orbit_dataarray(
        start_lon=0,
        start_lat=0,
        end_lon=20,
        end_lat=15,
        width=1e6,
        n_along_track=20,
        n_cross_track=5,
    )


@pytest.fixture(scope="function")
def orbit_antimeridian_dataarray() -> xr.DataArray:
    """Create orbit data array going over the antimeridian"""

    return get_orbit_dataarray(
        start_lon=160,
        start_lat=0,
        end_lon=-170,
        end_lat=15,
        width=1e6,
        n_along_track=20,
        n_cross_track=5,
    )


@pytest.fixture(scope="function")
def orbit_pole_dataarray() -> xr.DataArray:
    """Create orbit data array going over the pole"""

    return get_orbit_dataarray(
        start_lon=-30,
        start_lat=70,
        end_lon=150,
        end_lat=75,
        width=1e6,
        n_along_track=20,
        n_cross_track=5,
    )


@pytest.fixture(scope="function")
def grid_dataarray() -> xr.DataArray:
    """Create grid data array near 0 longitude and latitude"""

    return get_grid_dataarray(
        start_lon=-5,
        start_lat=-5,
        end_lon=20,
        end_lat=15,
        n_lon=20,
        n_lat=15,
    )


@pytest.fixture(scope="function")
def grid_antimeridian_dataarray() -> xr.DataArray:
    """Create grid data array going over the antimeridian"""

    return get_grid_dataarray(
        start_lon=160,
        start_lat=-5,
        end_lon=-170,
        end_lat=15,
        n_lon=20,
        n_lat=15,
    )


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
        """Test plotting orbit data going over the pole"""

        plot.plot_map(orbit_pole_dataarray)
        save_and_check_figure(get_test_name(self))

    def test_orbit_pole_projection(
        self,
        orbit_pole_dataarray: xr.Dataset,
    ) -> None:
        """Test plotting orbit data going over the pole on orthographic projection"""

        crs_proj = ccrs.Orthographic(0, 30)
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


def test_plot_map_mesh() -> None:
    pass  # TODO


def test_plot_map_mesh_centroids() -> None:
    pass  # TODO
