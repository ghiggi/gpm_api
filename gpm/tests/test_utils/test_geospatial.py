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
"""This module test the geospatial utilities."""

import numpy as np
import pytest
import xarray as xr
from pytest_mock import MockFixture

from gpm.tests.utils.fake_datasets import get_grid_dataarray, get_orbit_dataarray
from gpm.utils import geospatial

ExtentDictionary = dict[str, tuple[float, float, float, float]]


# Fixtures #####################################################################


@pytest.fixture()
def orbit_dataarray() -> xr.DataArray:
    return get_orbit_dataarray(
        start_lon=-50,
        start_lat=-1,
        end_lon=50,
        end_lat=1,
        width=1e5,
        n_along_track=11,
        n_cross_track=3,
    )


@pytest.fixture()
def orbit_dataarray_multiple_prime_meridian_crossings() -> xr.DataArray:
    """Orbit dataset that crosses the prime meridian multiple times"""

    da = get_orbit_dataarray(
        start_lon=-50,
        start_lat=-1,
        end_lon=50,
        end_lat=1,
        width=1e5,
        n_along_track=11,
        n_cross_track=3,
    )

    cross_track_tiled = da.cross_track.data
    along_track_tiled = np.arange(len(da.along_track) * 2)
    data_tiled = np.tile(da.data, (1, 2))
    lon_tiled = np.tile(da.lon.data, (1, 2))
    lat_tiled = np.tile(da.lat.data, (1, 2))
    granule_id_tiled = np.tile(da.gpm_granule_id.data, 2)

    da_tiled = xr.DataArray(
        data_tiled,
        coords={"cross_track": cross_track_tiled, "along_track": along_track_tiled},
    )
    da_tiled.coords["lat"] = (("cross_track", "along_track"), lat_tiled)
    da_tiled.coords["lon"] = (("cross_track", "along_track"), lon_tiled)
    da_tiled.coords["gpm_granule_id"] = ("along_track", granule_id_tiled)

    return da_tiled


@pytest.fixture()
def grid_dataarray() -> xr.DataArray:
    return get_grid_dataarray(
        start_lon=-50,
        start_lat=-50,
        end_lon=50,
        end_lat=50,
        n_lon=11,
        n_lat=11,
    )


# Tests ########################################################################


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

    padding = (0.1, 0.1, 0.2, 0.2)
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


class TestCrop:
    """Test crop"""

    extent = (-10, 20, -30, 40)

    def test_crop_orbit(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        ds = geospatial.crop(orbit_dataarray, self.extent)
        expected_lon = [-10, 0, 10, 20]
        np.testing.assert_array_almost_equal(ds.lon.values[0], expected_lon, decimal=2)

    def test_orbit_multiple_crossings(
        self,
        orbit_dataarray_multiple_prime_meridian_crossings: xr.DataArray,
    ) -> None:
        """Test with multiple crosses of extent"""

        with pytest.raises(ValueError):
            geospatial.crop(orbit_dataarray_multiple_prime_meridian_crossings, self.extent)

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

    # Mock read_countries_extent_dictionary
    mocker.patch(
        "gpm.utils.geospatial.read_countries_extent_dictionary",
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

    # Mock read_continents_extent_dictionary
    mocker.patch(
        "gpm.utils.geospatial.read_continents_extent_dictionary",
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
        orbit_dataarray: xr.DataArray,
    ) -> None:
        slices = geospatial.get_crop_slices_by_extent(orbit_dataarray, self.extent)
        expected_slices = [{"along_track": slice(4, 8)}]
        assert slices == expected_slices

    def test_orbit_multiple_crossings(
        self,
        orbit_dataarray_multiple_prime_meridian_crossings: xr.DataArray,
    ) -> None:
        slices = geospatial.get_crop_slices_by_extent(
            orbit_dataarray_multiple_prime_meridian_crossings, self.extent
        )
        expected_slices = [{"along_track": slice(4, 8)}, {"along_track": slice(15, 19)}]
        assert slices == expected_slices

    def test_orbit_outside(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        with pytest.raises(ValueError):
            geospatial.get_crop_slices_by_extent(orbit_dataarray, self.extent_outside_lon)
        with pytest.raises(ValueError):
            geospatial.get_crop_slices_by_extent(orbit_dataarray, self.extent_outside_lat)

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
        with pytest.raises(ValueError):
            geospatial.get_crop_slices_by_extent(da, self.extent)


def test_get_crop_slices_by_country(
    mocker: MockFixture,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test get_crop_slices_by_country"""

    country = "Froopyland"
    extent = (-10, 20, -30, 40)

    # Mock read_countries_extent_dictionary
    mocker.patch(
        "gpm.utils.geospatial.read_countries_extent_dictionary",
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

    # Mock read_continents_extent_dictionary
    mocker.patch(
        "gpm.utils.geospatial.read_continents_extent_dictionary",
        return_value={continent: extent},
    )

    # Get slices
    slices = geospatial.get_crop_slices_by_continent(grid_dataarray, continent)
    expected_slices = {"lon": slice(4, 8), "lat": slice(2, 10)}
    assert slices == expected_slices
