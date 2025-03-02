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
from gpm.utils.geospatial import (
    adjust_geographic_extent,
    check_extent,
    crop,
    crop_around_point,
    crop_by_continent,
    crop_by_country,
    extend_geographic_extent,
    get_circle_coordinates_around_point,
    get_continent_extent,
    get_country_extent,
    get_crop_slices_around_point,
    get_crop_slices_by_continent,
    get_crop_slices_by_country,
    get_crop_slices_by_extent,
    get_geographic_extent_around_point,
    get_geographic_extent_from_xarray,
    unwrap_longitude_degree,
)

ExtentDictionary = dict[str, tuple[float, float, float, float]]


# Fixtures #####################################################################


@pytest.fixture
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


@pytest.fixture
def orbit_dataarray_multiple_prime_meridian_crossings() -> xr.DataArray:
    """Orbit dataset that crosses the prime meridian multiple times."""
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
    lon_tiled = np.tile(da["lon"].data, (1, 2))
    lat_tiled = np.tile(da["lat"].data, (1, 2))
    granule_id_tiled = np.tile(da.gpm_granule_id.data, 2)

    da_tiled = xr.DataArray(
        data_tiled,
        coords={"cross_track": cross_track_tiled, "along_track": along_track_tiled},
    )
    da_tiled.coords["lat"] = (("cross_track", "along_track"), lat_tiled)
    da_tiled.coords["lon"] = (("cross_track", "along_track"), lon_tiled)
    da_tiled.coords["gpm_granule_id"] = ("along_track", granule_id_tiled)

    return da_tiled


@pytest.fixture
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


class TestCheckExtent:
    """Tests for the check_extent function."""

    def test_valid_extent(self):
        """Test that a valid extent passes without error."""
        assert list(check_extent([-180, 180, -90, 90])) == [-180, 180, -90, 90]
        assert list(check_extent([-360, 180, -98, 90])) == [-360, 180, -98, 90]  # should not assume lat/lon coords

    def test_invalid_extent_length(self):
        """Test that an error is raised when the extent does not contain exactly four elements."""
        with pytest.raises(ValueError) as excinfo:
            check_extent([-180, 180, -90])
        assert "four elements" in str(excinfo.value)

    def test_invalid_xmin_greater_than_xmax(self):
        """Test that an error is raised when xmin is not less than xmax."""
        with pytest.raises(ValueError) as excinfo:
            check_extent([180, -180, -90, 90])
        assert "xmin must be less than xmax" in str(excinfo.value)

    def test_invalid_ymin_greater_than_ymax(self):
        """Test that an error is raised when ymin is not less than ymax."""
        with pytest.raises(ValueError) as excinfo:
            check_extent([-180, 180, 90, -90])
        assert "ymin must be less than ymax" in str(excinfo.value)

    def test_invalid_numerical_values(self):
        """Test that the function handles non-numerical inputs."""
        with pytest.raises(ValueError):
            check_extent(["west", "east", "south", "north"])
        with pytest.raises(ValueError):
            check_extent([None, None, None, None])


class TestAdjustGeographicExtent:
    """Tests for adjust_geographic_extent function."""

    def test_adjust_extent_tuple_case(self):
        """Test extent adjustment with size tuple."""
        extent = (0, 10, 0, 10)
        size = (5, 6)
        expected = (2.5, 7.5, 2.0, 8.0)
        assert adjust_geographic_extent(extent, size) == expected

    def test_adjust_extent_single_number_size(self):
        """Test with single size  number (for symmetrical extent adjustment)."""
        extent = (0, 10, 0, 10)
        size = 5
        expected = (2.5, 7.5, 2.5, 7.5)
        assert adjust_geographic_extent(extent, size) == expected

    def test_adjust_extent_and_clip_lon_lats(self):
        """Test with single size  number (for symmetrical extent adjustment)."""
        extent = (0, 10, 0, 10)
        size = (360, 180)
        expected = (-180, 180, -90, 90)
        assert adjust_geographic_extent(extent, size) == expected

        size = (370, 190)
        expected = (-180, 180, -90, 90)
        assert adjust_geographic_extent(extent, size) == expected

    def test_adjust_extent_improper_size(self):
        """Ensure ValueError is raised with an improper size input."""
        with pytest.raises(ValueError):
            adjust_geographic_extent((0, 10, 0, 10), (5,))

    def test_adjust_extent_negative_size(self):
        """Check for handling of negative size values."""
        with pytest.raises(ValueError):
            adjust_geographic_extent((0, 10, 0, 10), -5)

    def test_adjust_extent_improper_type(self):
        """Ensure TypeError is raised with an improper size type."""
        with pytest.raises(TypeError):
            adjust_geographic_extent((0, 10, 0, 10), "dummy")


class TestExtendGeographicExtent:
    """Tests for extend_geographic_extent function."""

    def test_extend_extent_tuple_case(self):
        """Test extent extension with full tuple."""
        extent = (0, 10, 0, 10)
        padding = (1, 1, 2, 2)
        expected = (-1, 11, -2, 12)
        assert extend_geographic_extent(extent, padding) == expected

    def test_extend_extent_tuple_xy_case(self):
        """Test extent extension with (x,y) tuple."""
        extent = (0, 10, 0, 10)
        padding = (1, 2)
        expected = (-1, 11, -2, 12)
        assert extend_geographic_extent(extent, padding) == expected

    def test_extend_extent_single_number_padding(self):
        """Test extent extension  with single number padding."""
        extent = (0, 10, 0, 10)
        padding = 1
        expected = (-1, 11, -1, 11)
        assert extend_geographic_extent(extent, padding) == expected

    def test_shirnk_extent_single_number_padding(self):
        """Test extent shrinkage with negative padding."""
        extent = (0, 10, 0, 10)
        padding = -1
        expected = (1, 9, 1, 9)
        assert extend_geographic_extent(extent, padding) == expected

    def test_extend_extent_improper_padding(self):
        """Ensure ValueError is raised with improper padding input."""
        with pytest.raises(ValueError):
            extend_geographic_extent((0, 10, 0, 10), (1, 2, 3))

    def test_extend_extent_improper_type(self):
        """Ensure ValueError is raised with improper padding type."""
        with pytest.raises(TypeError):
            extend_geographic_extent((0, 10, 0, 10), "dummy")

    def test_extend_extent_boundary_limits(self):
        """Check how the function handles world boundaries."""
        extent = (-179, 179, -89, 89)
        padding = (5, 5, 5, 5)
        expected = (-180, 180, -90, 90)  # max/min limits for lon/lat
        assert extend_geographic_extent(extent, padding) == expected


class TestGetGeographicExtentAroundPoint:
    """Class to test get_geographic_extent_around_point function."""

    def test_with_valid_distance(self):
        """Test function with a valid distance and no size."""
        lon, lat, distance = -123.1207, 49.2827, 10000
        result = get_geographic_extent_around_point(lon, lat, distance=distance)
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 4, "Tuple should have four elements"
        np.testing.assert_almost_equal(result, [-123.258144, -122.983255, 49.1927835, 49.372615], decimal=6)

    def test_with_valid_size(self):
        """Test function with a valid size and no distance."""
        lon, lat = -123.1207, 49.2827
        size = (0.1, 0.1)
        result = get_geographic_extent_around_point(lon, lat, size=size)
        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 4, "Tuple should have four elements"
        np.testing.assert_almost_equal(result, [-123.1707, -123.0707, 49.2327, 49.332699], decimal=6)

    def test_with_both_distance_and_size(self):
        """Test function raises ValueError when both distance and size are provided."""
        lon, lat, distance, size = -123.1207, 49.2827, 10000, (0.1, 0.1)
        with pytest.raises(ValueError):
            get_geographic_extent_around_point(lon, lat, distance=distance, size=size)

    def test_with_neither_distance_nor_size(self):
        """Test function raises ValueError when neither distance nor size is provided."""
        lon, lat = -123.1207, 49.2827
        with pytest.raises(ValueError):
            get_geographic_extent_around_point(lon, lat)

    def test_with_invalid_size_type(self):
        """Test function raises TypeError when size is of an invalid type."""
        lon, lat = -123.1207, 49.2827
        with pytest.raises(TypeError):
            get_geographic_extent_around_point(lon, lat, size="invalid_size_type")


def test_get_country_extent(
    country_extent_dictionary: ExtentDictionary,
) -> None:
    """Test get_country_extent."""
    # Test valid country
    country = "Afghanistan"
    e = country_extent_dictionary[country]
    expected_extent = (e[0] - 0.2, e[1] + 0.2, e[2] - 0.2, e[3] + 0.2)
    returned_extent = get_country_extent(country)
    assert returned_extent == expected_extent

    # Test invalid country
    country = "Invalid"
    with pytest.raises(ValueError):
        get_country_extent(country)

    # Test typo in country name
    country = "Afganista"  # Instead of Afghanistan
    with pytest.raises(ValueError) as exception_info:
        get_country_extent(country)

    assert "Afghanistan" in str(exception_info.value.args[0])

    # Test invalid country type
    country = 123
    with pytest.raises(TypeError):
        get_country_extent(country)


def test_get_continent_extent(
    continent_extent_dictionary: ExtentDictionary,
) -> None:
    """Test get_continent_extent."""
    # Test valid continent
    continent = "Africa"
    e = continent_extent_dictionary[continent]
    expected_extent = (e[0], e[1], e[2], e[3])
    returned_extent = get_continent_extent(continent)
    assert returned_extent == expected_extent

    # Test invalid continent
    continent = "Invalid"
    with pytest.raises(ValueError):
        get_continent_extent(continent)

    # Test typo in continent name
    continent = "Arica"
    with pytest.raises(ValueError) as exception_info:
        get_continent_extent(continent)

    assert "Africa" in str(exception_info.value.args[0])

    # Test invalid continent type
    continent = 123
    with pytest.raises(TypeError):
        get_continent_extent(continent)


def test_get_geographic_extent_from_xarray() -> None:
    """Test get_geographic_extent_from_xarray."""
    ds = xr.Dataset(
        {
            "lon": [-10, 0, 20],
            "lat": [-30, 0, 40],
        },
    )

    # Test without padding
    expected_extent = (-10, 20, -30, 40)
    returned_extent = get_geographic_extent_from_xarray(ds)
    assert returned_extent == expected_extent

    # Test with float padding
    padding = 0.1
    expected_extent = (-10.1, 20.1, -30.1, 40.1)
    returned_extent = get_geographic_extent_from_xarray(ds, padding=padding)
    assert returned_extent == expected_extent

    # Test with size
    expected_extent = (0, 10, 0, 10)
    returned_extent = get_geographic_extent_from_xarray(ds, size=10)
    assert returned_extent == expected_extent

    # Test with padding exceeding bounds
    padding = 180
    expected_extent = (-180, 180, -90, 90)
    returned_extent = get_geographic_extent_from_xarray(ds, padding=padding)
    assert returned_extent == expected_extent

    # Test with tuple padding
    padding = (0.1, 0.2)
    expected_extent = (-10.1, 20.1, -30.2, 40.2)
    returned_extent = get_geographic_extent_from_xarray(ds, padding=padding)
    assert returned_extent == expected_extent

    padding = (0.1, 0.1, 0.2, 0.2)
    expected_extent = (-10.1, 20.1, -30.2, 40.2)
    returned_extent = get_geographic_extent_from_xarray(ds, padding=padding)
    assert returned_extent == expected_extent

    # Test with invalid padding
    with pytest.raises(TypeError):
        get_geographic_extent_from_xarray(ds, padding="invalid")
    with pytest.raises(ValueError):
        get_geographic_extent_from_xarray(ds, padding=(0.1,))
    with pytest.raises(ValueError):
        get_geographic_extent_from_xarray(ds, padding=(0.1, 0.2, 0.3))

    # Test with object crossing dateline
    ds = xr.Dataset(
        {
            "lon": [170, 180, -160],
            "lat": [-30, 0, 40],
        },
    )
    with pytest.raises(NotImplementedError):
        get_geographic_extent_from_xarray(ds)


def test_get_circle_coordinates_around_point():
    """Test the function get_circle_coordinates_around_point."""
    lon, lat, radius = -123.1207, 88.2827, 100_000
    num_vertices = 360
    lons, lats = get_circle_coordinates_around_point(lon, lat, radius=radius, num_vertices=num_vertices)
    assert lons.shape == (num_vertices,), "Output arrays should match the number of vertices."
    assert lats.shape == (num_vertices,), "Output arrays should match the number of vertices."


def test_unwrap_longitude_degree():
    """Test unwrap_longitude_degree."""
    # Longitudes larger than 180
    longitudes = np.array([181, 359, 360, 361, 539, 540, 541, 720])
    expected = np.array([-179, -1, 0, 1, 179, -180, -179, 0])
    assert np.array_equal(unwrap_longitude_degree(longitudes), expected), "Should wrap positive longitudes correctly."

    # Longitudes smaller than -180
    longitudes = np.array([-181, -360, -539, -540, -541, -720])
    expected = np.array([179, 0, -179, -180, 179, 0])
    assert np.array_equal(unwrap_longitude_degree(longitudes), expected), "Should wrap negative longitudes correctly."

    # Longitudes within [-180, 180]
    longitudes = np.array([-179, 0, 179])
    expected = longitudes
    assert np.array_equal(
        unwrap_longitude_degree(longitudes),
        expected,
    ), "Should not alter longitudes within [-180, 180)."

    # Edge cases
    longitudes = np.array([180, -180])
    expected = np.array([-180, -180])
    assert np.array_equal(unwrap_longitude_degree(longitudes), expected), "At -180/180, should return -180."


class TestCrop:
    """Test crop."""

    extent = (-10, 20, -30, 40)

    def test_orbit(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        ds = crop(orbit_dataarray, self.extent)
        expected_lon = [-10, 0, 10, 20]
        np.testing.assert_array_almost_equal(ds.lon.to_numpy()[0], expected_lon, decimal=2)

    def test_orbit_multiple_crossings(
        self,
        orbit_dataarray_multiple_prime_meridian_crossings: xr.DataArray,
    ) -> None:
        """Test with multiple crosses of extent."""
        with pytest.raises(ValueError):
            crop(orbit_dataarray_multiple_prime_meridian_crossings, self.extent)

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        da = crop(grid_dataarray, self.extent)
        expected_lon = np.arange(-10, 21, 10)
        expected_lat = np.arange(-30, 41, 10)
        np.testing.assert_array_equal(da["lon"].to_numpy(), expected_lon)
        np.testing.assert_array_equal(da["lat"].to_numpy(), expected_lat)

    def test_invalid(self) -> None:
        da = xr.DataArray()
        with pytest.raises(ValueError):
            crop(da, self.extent)


def test_crop_by_country(
    mocker: MockFixture,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test crop_by_country."""
    country = "Wakanda"
    extent = (-10, 20, -30, 40)

    # Mock read_countries_extent_dictionary
    mocker.patch(
        "gpm.utils.geospatial.read_countries_extent_dictionary",
        return_value={country: extent},
    )

    # Crop
    da = crop_by_country(grid_dataarray, country)
    expected_lon = np.arange(-10, 21, 10)
    expected_lat = np.arange(-30, 41, 10)
    np.testing.assert_array_equal(da["lon"].to_numpy(), expected_lon)
    np.testing.assert_array_equal(da["lat"].to_numpy(), expected_lat)


def test_crop_by_continent(
    mocker: MockFixture,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test crop_by_continent."""
    continent = "Middle Earth"
    extent = (-10, 20, -30, 40)

    # Mock read_continents_extent_dictionary
    mocker.patch(
        "gpm.utils.geospatial.read_continents_extent_dictionary",
        return_value={continent: extent},
    )

    # Crop
    da = crop_by_continent(grid_dataarray, continent)
    expected_lon = np.arange(-10, 21, 10)
    expected_lat = np.arange(-30, 41, 10)
    np.testing.assert_array_equal(da["lon"].to_numpy(), expected_lon)
    np.testing.assert_array_equal(da["lat"].to_numpy(), expected_lat)


def test_crop_around_point(
    grid_dataarray: xr.DataArray,
) -> None:
    """Test crop_around_point."""
    lon = 0
    lat = 0
    distance = 5000_000
    size = 20
    # Crop by distance around point
    da = crop_around_point(grid_dataarray, lon=lon, lat=lat, distance=distance)
    expected_lon = np.arange(-40, 50, 10)  # [-40, 40]
    expected_lat = np.arange(-40, 50, 10)  # [-40, 40]
    np.testing.assert_array_equal(da["lon"].to_numpy(), expected_lon)
    np.testing.assert_array_equal(da["lat"].to_numpy(), expected_lat)

    # Crop by size around point
    da = crop_around_point(grid_dataarray, lon=lon, lat=lat, size=size)
    expected_lon = np.arange(-10, 20, 10)  # [-10, 10]
    expected_lat = np.arange(-10, 20, 10)  # [-10, 10]
    np.testing.assert_array_equal(da["lon"].to_numpy(), expected_lon)
    np.testing.assert_array_equal(da["lat"].to_numpy(), expected_lat)


class TestGetCropSlicesByExtent:
    """Test get_crop_slices_by_extent.

    We know this function works correctly because it is used in crop.
    Still, it can be used directly by the user, so we test them here.
    """

    extent = (-10, 20, -30, 40)
    extent_outside_lon = (60, 70, -10, 10)
    extent_outside_lat = (-10, 10, 60, 70)

    def test_orbit(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        slices = get_crop_slices_by_extent(orbit_dataarray, self.extent)
        expected_slices = [{"along_track": slice(4, 8)}]
        assert slices == expected_slices

    def test_orbit_multiple_crossings(
        self,
        orbit_dataarray_multiple_prime_meridian_crossings: xr.DataArray,
    ) -> None:
        slices = get_crop_slices_by_extent(
            orbit_dataarray_multiple_prime_meridian_crossings,
            self.extent,
        )
        expected_slices = [{"along_track": slice(4, 8)}, {"along_track": slice(15, 19)}]
        assert slices == expected_slices

    def test_orbit_outside(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        with pytest.raises(ValueError):
            get_crop_slices_by_extent(orbit_dataarray, self.extent_outside_lon)
        with pytest.raises(ValueError):
            get_crop_slices_by_extent(orbit_dataarray, self.extent_outside_lat)

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        slices = get_crop_slices_by_extent(grid_dataarray, self.extent)
        expected_slices = {"lon": slice(4, 8), "lat": slice(2, 10)}
        assert slices == expected_slices

    def test_grid_outside(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        with pytest.raises(ValueError):
            get_crop_slices_by_extent(grid_dataarray, self.extent_outside_lon)
        with pytest.raises(ValueError):
            get_crop_slices_by_extent(grid_dataarray, self.extent_outside_lat)

    def test_invalid(self) -> None:
        da = xr.DataArray()
        with pytest.raises(ValueError):
            get_crop_slices_by_extent(da, self.extent)


def test_get_crop_slices_by_country(
    mocker: MockFixture,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test get_crop_slices_by_country."""
    country = "Froopyland"
    extent = (-10, 20, -30, 40)

    # Mock read_countries_extent_dictionary
    mocker.patch(
        "gpm.utils.geospatial.read_countries_extent_dictionary",
        return_value={country: extent},
    )

    # Get slices
    slices = get_crop_slices_by_country(grid_dataarray, country)
    expected_slices = {"lon": slice(4, 8), "lat": slice(2, 10)}
    assert slices == expected_slices


def test_get_crop_slices_by_continent(
    mocker: MockFixture,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test get_crop_slices_by_continent."""
    continent = "Atlantis"
    extent = (-10, 20, -30, 40)

    # Mock read_continents_extent_dictionary
    mocker.patch(
        "gpm.utils.geospatial.read_continents_extent_dictionary",
        return_value={continent: extent},
    )

    # Get slices
    slices = get_crop_slices_by_continent(grid_dataarray, continent)
    expected_slices = {"lon": slice(4, 8), "lat": slice(2, 10)}
    assert slices == expected_slices


def test_get_crop_slices_around_point(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test get_crop_slices_around_point."""
    lon = 0
    lat = 0
    distance = 2000_000  # 2000 km

    # Get grid slices
    slices = get_crop_slices_around_point(grid_dataarray, lon=lon, lat=lat, distance=distance)
    expected_slices = {"lon": slice(4, 7), "lat": slice(4, 7)}
    assert slices == expected_slices

    # Get orbit slices
    slices = get_crop_slices_around_point(orbit_dataarray, lon=lon, lat=lat, distance=distance)
    expected_slices = [{"along_track": slice(4, 7)}]
    assert slices == expected_slices
