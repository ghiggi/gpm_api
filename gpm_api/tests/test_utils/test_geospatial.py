import numpy as np
import pytest
from pytest_mock import MockFixture
from typing import Dict, Tuple
import xarray as xr

from gpm_api.utils import geospatial

ExtentDictionary = Dict[str, Tuple[float, float, float, float]]


def test_get_country_extent(
    country_extent_dictionary: ExtentDictionary,
) -> None:
    """Test get_country_extent"""

    # Test valid country
    country = "Afghanistan"
    e = country_extent_dictionary[country]
    expected_extent = (e[0] - 0.2, e[1] + 0.2, e[2] - 0.2, e[3] + 0.2)
    returned_extent = geospatial.get_country_extent(country)
    assert returned_extent == expected_extent

    # Test invalid country
    country = "Invalid"
    with pytest.raises(ValueError):
        geospatial.get_country_extent(country)

    # Test typo in country name
    country = "Afganistan"
    with pytest.raises(ValueError) as exception_info:
        geospatial.get_country_extent(country)

    assert "Afghanistan" in str(exception_info.value.args[0])

    # Test invalid country type
    country = 123
    with pytest.raises(TypeError):
        geospatial.get_country_extent(country)


def test_get_continent_extent(
    continent_extent_dictionary: ExtentDictionary,
) -> None:
    """Test get_continent_extent"""

    # Test valid continent
    continent = "Africa"
    e = continent_extent_dictionary[continent]
    expected_extent = (e[0], e[1], e[2], e[3])
    returned_extent = geospatial.get_continent_extent(continent)
    assert returned_extent == expected_extent

    # Test invalid continent
    continent = "Invalid"
    with pytest.raises(ValueError):
        geospatial.get_continent_extent(continent)

    # Test typo in continent name
    continent = "Arica"
    with pytest.raises(ValueError) as exception_info:
        geospatial.get_continent_extent(continent)

    assert "Africa" in str(exception_info.value.args[0])

    # Test invalid continent type
    continent = 123
    with pytest.raises(TypeError):
        geospatial.get_continent_extent(continent)


def test_get_extent() -> None:
    """Test get_extent"""

    ds = xr.Dataset(
        {
            "lon": [-10, 0, 20],
            "lat": [-30, 0, 40],
        }
    )

    # Test without padding
    expected_extent = (-10, 20, -30, 40)
    returned_extent = geospatial.get_extent(ds)
    assert returned_extent == expected_extent

    # Test with float padding
    padding = 0.1
    expected_extent = (-10.1, 20.1, -30.1, 40.1)
    returned_extent = geospatial.get_extent(ds, padding=padding)
    assert returned_extent == expected_extent

    # Test with padding exceeding bounds
    padding = 180
    expected_extent = (-180, 180, -90, 90)
    returned_extent = geospatial.get_extent(ds, padding=padding)
    assert returned_extent == expected_extent

    # Test with tuple padding
    padding = (0.1, 0.2)
    expected_extent = (-10.1, 20.1, -30.2, 40.2)
    returned_extent = geospatial.get_extent(ds, padding=padding)
    assert returned_extent == expected_extent

    # Test with invalid padding
    with pytest.raises(TypeError):
        geospatial.get_extent(ds, padding="invalid")
    with pytest.raises(ValueError):
        geospatial.get_extent(ds, padding=(0.1,))
    with pytest.raises(ValueError):
        geospatial.get_extent(ds, padding=(0.1, 0.2, 0.3))

    # Test with object crossing dateline
    ds = xr.Dataset(
        {
            "lon": [170, 180, -160],
            "lat": [-30, 0, 40],
        }
    )
    with pytest.raises(NotImplementedError):
        geospatial.get_extent(ds)


@pytest.fixture
def orbit_dataset() -> xr.Dataset:
    # Values along track
    lon = np.arange(-50, 51, 10)
    lat = np.zeros_like(lon)

    # Add cross track dimension
    lat = lat[np.newaxis, :]
    lon = lon[np.newaxis, :]

    # Create dataset
    ds = xr.Dataset()
    ds["lat"] = (("cross_track", "along_track"), lat)
    ds["lon"] = (("cross_track", "along_track"), lon)

    return ds


@pytest.fixture
def orbit_dataset_multiple_prime_meridian_crossings() -> xr.Dataset:
    """Orbit dataset that crosses the prime meridian multiple times"""

    # Values along track
    lon = np.arange(-50, 51, 10)
    lon = np.tile(lon, 2)
    lat = np.zeros_like(lon)

    # Add cross track dimension
    lat = lat[np.newaxis, :]
    lon = lon[np.newaxis, :]

    # Create dataset
    ds = xr.Dataset()
    ds["lat"] = (("cross_track", "along_track"), lat)
    ds["lon"] = (("cross_track", "along_track"), lon)

    return ds


@pytest.fixture
def grid_dataarray() -> xr.DataArray:
    lon = np.arange(-50, 51, 10)
    lat = np.arange(-50, 51, 10)
    data = np.zeros((len(lat), len(lon)))

    # Create data array
    da = xr.DataArray(data, coords={"lat": lat, "lon": lon})

    return da


class TestCrop:
    """Test crop"""

    extent = (-10, 20, -30, 40)

    def test_crop_orbit(
        self,
        orbit_dataset: xr.Dataset,
    ) -> None:
        ds = geospatial.crop(orbit_dataset, self.extent)
        expected_lon = [-10, 0, 10, 20]
        np.testing.assert_array_equal(ds.lon.values[0], expected_lon)

    def test_orbit_multiple_crossings(
        self,
        orbit_dataset_multiple_prime_meridian_crossings: xr.Dataset,
    ) -> None:
        """Test with multiple crosses of extent"""

        with pytest.raises(ValueError):
            geospatial.crop(orbit_dataset_multiple_prime_meridian_crossings, self.extent)

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        da = geospatial.crop(grid_dataarray, self.extent)
        expected_lon = np.arange(-10, 21, 10)
        expected_lat = np.arange(-30, 41, 10)
        np.testing.assert_array_equal(da.lon.values, expected_lon)
        np.testing.assert_array_equal(da.lat.values, expected_lat)

    def test_invalid(self) -> None:
        da = xr.DataArray()
        with pytest.raises(ValueError):
            geospatial.crop(da, self.extent)


def test_crop_by_country(
    mocker: MockFixture,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test crop_by_country"""

    country = "Wakanda"
    extent = (-10, 20, -30, 40)

    # Mock _get_country_extent_dictionary
    mocker.patch(
        "gpm_api.utils.geospatial._get_country_extent_dictionary",
        return_value={country: extent},
    )

    # Crop
    da = geospatial.crop_by_country(grid_dataarray, country)
    expected_lon = np.arange(-10, 21, 10)
    expected_lat = np.arange(-30, 41, 10)
    np.testing.assert_array_equal(da.lon.values, expected_lon)
    np.testing.assert_array_equal(da.lat.values, expected_lat)


def test_crop_by_continent(
    mocker: MockFixture,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test crop_by_continent"""

    continent = "Middle Earth"
    extent = (-10, 20, -30, 40)

    # Mock _get_continent_extent_dictionary
    mocker.patch(
        "gpm_api.utils.geospatial._get_continent_extent_dictionary",
        return_value={continent: extent},
    )

    # Crop
    da = geospatial.crop_by_continent(grid_dataarray, continent)
    expected_lon = np.arange(-10, 21, 10)
    expected_lat = np.arange(-30, 41, 10)
    np.testing.assert_array_equal(da.lon.values, expected_lon)
    np.testing.assert_array_equal(da.lat.values, expected_lat)


class TestCropSlicesByExtent:
    """Test crop_slices_by_extent.

    We know this function works correctly because it is used in crop.
    Still, they can be used directly by the user, so we test them here.
    """

    extent = (-10, 20, -30, 40)
    extent_outside_lon = (60, 70, -10, 10)
    extent_outside_lat = (-10, 10, 60, 70)

    def test_orbit(
        self,
        orbit_dataset: xr.Dataset,
    ) -> None:
        slices = geospatial.get_crop_slices_by_extent(orbit_dataset, self.extent)
        expected_slices = [{"along_track": slice(4, 8)}]
        assert slices == expected_slices

    def test_orbit_multiple_crossings(
        self,
        orbit_dataset_multiple_prime_meridian_crossings: xr.Dataset,
    ) -> None:
        slices = geospatial.get_crop_slices_by_extent(
            orbit_dataset_multiple_prime_meridian_crossings, self.extent
        )
        expected_slices = [{"along_track": slice(4, 8)}, {"along_track": slice(15, 19)}]
        assert slices == expected_slices

    def test_orbit_outside(
        self,
        orbit_dataset: xr.Dataset,
    ) -> None:
        with pytest.raises(ValueError):
            geospatial.get_crop_slices_by_extent(orbit_dataset, self.extent_outside_lon)
        with pytest.raises(ValueError):
            geospatial.get_crop_slices_by_extent(orbit_dataset, self.extent_outside_lat)

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        slices = geospatial.get_crop_slices_by_extent(grid_dataarray, self.extent)
        expected_slices = {"lon": slice(4, 8), "lat": slice(2, 10)}
        assert slices == expected_slices

    def test_grid_outside(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        with pytest.raises(ValueError):
            geospatial.get_crop_slices_by_extent(grid_dataarray, self.extent_outside_lon)
        with pytest.raises(ValueError):
            geospatial.get_crop_slices_by_extent(grid_dataarray, self.extent_outside_lat)

    def test_invalid(self) -> None:
        da = xr.DataArray()
        with pytest.raises(NotImplementedError):
            geospatial.get_crop_slices_by_extent(da, self.extent)


def test_get_crop_slices_by_country(
    mocker: MockFixture,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test get_crop_slices_by_country"""

    country = "Froopyland"
    extent = (-10, 20, -30, 40)

    # Mock _get_country_extent_dictionary
    mocker.patch(
        "gpm_api.utils.geospatial._get_country_extent_dictionary",
        return_value={country: extent},
    )

    # Get slices
    slices = geospatial.get_crop_slices_by_country(grid_dataarray, country)
    expected_slices = {"lon": slice(4, 8), "lat": slice(2, 10)}
    assert slices == expected_slices


def test_get_crop_slices_by_continent(
    mocker: MockFixture,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test get_crop_slices_by_continent"""

    continent = "Atlantis"
    extent = (-10, 20, -30, 40)

    # Mock _get_continent_extent_dictionary
    mocker.patch(
        "gpm_api.utils.geospatial._get_continent_extent_dictionary",
        return_value={continent: extent},
    )

    # Get slices
    slices = geospatial.get_crop_slices_by_continent(grid_dataarray, continent)
    expected_slices = {"lon": slice(4, 8), "lat": slice(2, 10)}
    assert slices == expected_slices
