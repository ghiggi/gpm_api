import numpy as np
from typing import Union
import xarray as xr

import pytest
from pytest_mock import MockerFixture

from gpm_api.utils import checks
from utils import (
    create_dataset_with_coordinate,
    create_fake_datetime_array_from_hours_list,
    create_orbit_time_array,
)


def test_get_missing_granule_numbers() -> None:
    """Test get_missing_granule_numbers"""

    # Test without missing granules
    granule_ids = np.arange(10)
    ds = create_dataset_with_coordinate("gpm_granule_id", granule_ids)
    returned_missing_ids = checks.get_missing_granule_numbers(ds)
    assert len(returned_missing_ids) == 0

    # Test with missing granules
    granule_ids = np.array([0, 1, 2, 7, 9])
    ds = create_dataset_with_coordinate("gpm_granule_id", granule_ids)
    expected_missing_ids = np.array([3, 4, 5, 6, 8])
    returned_missing_ids = checks.get_missing_granule_numbers(ds)
    assert np.all(returned_missing_ids == expected_missing_ids)


def test_is_contiguous_granule() -> None:
    """Test _is_contiguous_granule"""

    # Test expected behavior (True added at the end)
    granule_ids = np.array([0, 1, 2, 7, 8, 9])
    expected_bool_array = np.array([True, True, False, True, True, True])
    returned_bool_array = checks._is_contiguous_granule(granule_ids)
    np.testing.assert_array_equal(returned_bool_array, expected_bool_array)


class TestGetSlicesContiguousGranules:
    """Test get_slices_contiguous_granules"""

    def test_grid(
        self,
        set_is_grid_to_true: None,
    ) -> None:
        time = create_fake_datetime_array_from_hours_list([0, 1, 2, 7, 8, 9])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = [slice(0, 3), slice(3, 6)]
        returned_slices = checks.get_slices_contiguous_granules(ds)
        assert returned_slices == expected_slices

    def test_orbit(
        self,
        set_is_orbit_to_true: None,
    ) -> None:
        granule_ids = np.array([0, 1, 2, 7, 8, 9])
        ds = create_dataset_with_coordinate("gpm_granule_id", granule_ids)
        expected_slices = [slice(0, 3), slice(3, 6)]
        returned_slices = checks.get_slices_contiguous_granules(ds)
        assert returned_slices == expected_slices

        # Test 0 or 1 granules
        granule_ids = np.array([])
        ds = create_dataset_with_coordinate("gpm_granule_id", granule_ids)
        expected_slices = []
        returned_slices = checks.get_slices_contiguous_granules(ds)
        assert returned_slices == expected_slices

        granule_ids = np.array([0])
        ds = create_dataset_with_coordinate("gpm_granule_id", granule_ids)
        expected_slices = []
        returned_slices = checks.get_slices_contiguous_granules(ds)
        assert returned_slices == expected_slices

    def test_unknown(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test neither grid nor orbit"""

        mocker.patch("gpm_api.utils.checks.is_grid", return_value=False)
        mocker.patch("gpm_api.utils.checks.is_orbit", return_value=False)

        time = create_fake_datetime_array_from_hours_list([0, 1, 2, 7, 8, 9])
        ds = create_dataset_with_coordinate("time", time)
        with pytest.raises(ValueError):
            checks.get_slices_contiguous_granules(ds)


def test_check_missing_granules() -> None:
    """Test check_missing_granules"""

    # Test without missing granules
    granule_ids = np.arange(10)
    ds = create_dataset_with_coordinate("gpm_granule_id", granule_ids)
    checks.check_missing_granules(ds)
    # No error raised

    # Test with missing granules
    granule_ids = np.array([0, 1, 2, 7, 9])
    ds = create_dataset_with_coordinate("gpm_granule_id", granule_ids)
    with pytest.raises(ValueError):
        checks.check_missing_granules(ds)


class TestHasMissingGranules:
    """Test has_missing_granules"""

    def test_grid(
        self,
        set_is_grid_to_true: None,
    ) -> None:
        # Test without missing granules
        time = np.arange(10)
        ds = create_dataset_with_coordinate("time", time)
        assert not checks.has_missing_granules(ds)

        # Test with missing granules
        time = np.array([0, 1, 2, 7, 9])
        ds = create_dataset_with_coordinate("time", time)
        assert checks.has_missing_granules(ds)

    def test_orbit(
        self,
        set_is_orbit_to_true: None,
    ) -> None:
        # Test without missing granules
        granule_ids = np.arange(10)
        ds = create_dataset_with_coordinate("gpm_granule_id", granule_ids)
        assert not checks.has_missing_granules(ds)

        # Test with missing granules
        granule_ids = np.array([0, 1, 2, 7, 9])
        ds = create_dataset_with_coordinate("gpm_granule_id", granule_ids)
        assert checks.has_missing_granules(ds)

    def test_unknown(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Test neither grid nor orbit"""

        mocker.patch("gpm_api.utils.checks.is_grid", return_value=False)
        mocker.patch("gpm_api.utils.checks.is_orbit", return_value=False)

        time = np.arange(10)
        ds = create_dataset_with_coordinate("time", time)
        with pytest.raises(ValueError):
            checks.has_missing_granules(ds)


class TestGetSlicesRegularTime:
    """Test get_slices_regular_time"""

    def test_tolerance_provided(self) -> None:
        # Test regular time
        time = create_fake_datetime_array_from_hours_list(np.arange(0, 10))
        tolerance = time[1] - time[0]
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = [slice(0, 10)]
        returned_slices = checks.get_slices_regular_time(ds, tolerance=tolerance)
        assert returned_slices == expected_slices

        # Test irregular time
        time = create_fake_datetime_array_from_hours_list([0, 1, 2, 7, 8, 9])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = [slice(0, 3), slice(3, 6)]
        returned_slices = checks.get_slices_regular_time(ds, tolerance=tolerance)
        assert returned_slices == expected_slices

        # Test 0 or 1 timesteps
        time = create_fake_datetime_array_from_hours_list([])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = []
        returned_slices = checks.get_slices_regular_time(ds, tolerance=tolerance)
        assert returned_slices == expected_slices

        time = create_fake_datetime_array_from_hours_list([0])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = [slice(0, 1)]
        returned_slices = checks.get_slices_regular_time(ds, tolerance=tolerance)
        assert returned_slices == expected_slices

        # Only keep large enough slices
        time = create_fake_datetime_array_from_hours_list([0, 1, 2, 7, 8])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = [slice(0, 3)]
        returned_slices = checks.get_slices_regular_time(ds, tolerance=tolerance, min_size=3)
        assert returned_slices == expected_slices

    def test_grid(
        self,
        set_is_grid_to_true: None,
    ) -> None:
        # Tolerance not provided: inferred from first two values
        time = create_fake_datetime_array_from_hours_list([1, 2, 3, 7, 8, 9])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = [slice(0, 3), slice(3, 6)]
        returned_slices = checks.get_slices_regular_time(ds, tolerance=None)
        assert returned_slices == expected_slices

    def test_orbit(
        self,
        set_is_orbit_to_true: None,
    ) -> None:
        # Tolerance not provided: equal to gpm_api.utils.checks.ORBIT_TIME_TOLERANCE
        time = create_orbit_time_array([0, 1, 2, 7, 8, 9])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = [slice(0, 3), slice(3, 6)]
        returned_slices = checks.get_slices_regular_time(ds, tolerance=None)
        assert returned_slices == expected_slices


class TestGetSlicesNonRegularTime:
    """Test get_slices_non_regular_time"""

    def test_tolerance_provided(self) -> None:
        # Test regular time
        time = create_fake_datetime_array_from_hours_list(np.arange(0, 10))
        tolerance = time[1] - time[0]
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = []
        returned_slices = checks.get_slices_non_regular_time(ds, tolerance=tolerance)

        # Test irregular time
        #                                             0  1  2  3  4  5  6   7   8
        time = create_fake_datetime_array_from_hours_list([0, 1, 2, 4, 5, 6, 10, 11, 12])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = [slice(2, 4), slice(5, 7)]  # All slices have length 2
        returned_slices = checks.get_slices_non_regular_time(ds, tolerance=tolerance)
        assert returned_slices == expected_slices

        # Test 0 or 1 timesteps
        time = create_fake_datetime_array_from_hours_list([])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = []
        returned_slices = checks.get_slices_non_regular_time(ds, tolerance=tolerance)
        assert returned_slices == expected_slices

        time = create_fake_datetime_array_from_hours_list([0])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = []
        returned_slices = checks.get_slices_non_regular_time(ds, tolerance=tolerance)
        assert returned_slices == expected_slices

    def test_grid(
        self,
        set_is_grid_to_true: None,
    ) -> None:
        # Tolernace not provided: inferred from first two values
        #                                             0  1  2  3  4  5  6   7   8
        time = create_fake_datetime_array_from_hours_list([0, 1, 2, 4, 5, 6, 10, 11, 12])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = [slice(2, 4), slice(5, 7)]  # All slices have length 2
        returned_slices = checks.get_slices_non_regular_time(ds, tolerance=None)
        assert returned_slices == expected_slices

    def test_orbit(
        self,
        set_is_orbit_to_true: None,
    ) -> None:
        # Tolerance not provided: equal to gpm_api.utils.checks.ORBIT_TIME_TOLERANCE
        #                               0  1  2  3  4  5  6   7   8
        time = create_orbit_time_array([0, 1, 2, 4, 5, 6, 10, 11, 12])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = [slice(2, 4), slice(5, 7)]
        returned_slices = checks.get_slices_non_regular_time(ds, tolerance=None)
        assert returned_slices == expected_slices


class TestCheckRegularTime:
    """Test check_regular_time"""

    def test_grid(
        self,
        set_is_grid_to_true: None,
    ) -> None:
        # Test regular time
        time = create_fake_datetime_array_from_hours_list(np.arange(0, 10))
        ds = create_dataset_with_coordinate("time", time)
        checks.check_regular_time(ds)

        # Test irregular time
        time = create_fake_datetime_array_from_hours_list([0, 1, 2, 7, 8, 9])
        ds = create_dataset_with_coordinate("time", time)
        with pytest.raises(ValueError):
            checks.check_regular_time(ds)

    def test_orbit(
        self,
        set_is_orbit_to_true: None,
    ) -> None:
        # Test regular time
        time = create_orbit_time_array(np.arange(0, 10))
        ds = create_dataset_with_coordinate("time", time)
        checks.check_regular_time(ds)

        # Test irregular time
        time = create_orbit_time_array([0, 1, 2, 7, 8, 9])
        ds = create_dataset_with_coordinate("time", time)
        with pytest.raises(ValueError):
            checks.check_regular_time(ds)


def test_check_cross_track() -> None:
    """Test _check_cross_track decorator"""

    @checks._check_cross_track
    def identity(xr_obj: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
        return xr_obj

    # Test with cross_track
    da = xr.DataArray(np.arange(10), dims=["cross_track"])
    identity(da)
    # No error raised

    # Test without cross_track
    da = xr.DataArray(np.arange(10))
    with pytest.raises(ValueError):
        identity(da)


def test_get_along_track_scan_distance() -> None:
    """Test _get_along_track_scan_distance"""

    # Values along track
    lat = np.array([60, 60, 60])
    lon = np.array([0, 45, 90])

    # Stack values for cross track dimension
    lat = np.stack((np.random.rand(3), lat, np.random.rand(3)))
    lon = np.stack((np.random.rand(3), lon, np.random.rand(3)))

    # Create dataset
    ds = xr.Dataset()
    ds["lat"] = (("cross_track", "along_track"), lat)
    ds["lon"] = (("cross_track", "along_track"), lon)

    returned_distances = checks._get_along_track_scan_distance(ds)

    RADIUS_EARTH = 6357e3
    expected_distance = RADIUS_EARTH * np.pi / 8
    np.testing.assert_allclose(
        returned_distances, [expected_distance, expected_distance], rtol=0.02
    )


class TestContinuousScans:
    n_along_track = 10
    cut_idx = 5

    @pytest.fixture
    def ds_contiguous(self) -> xr.Dataset:
        # Values along track
        lat = np.array([60] * self.n_along_track)
        lon = np.arange(self.n_along_track)

        # Add cross track dimension
        lat = lat[np.newaxis, :]
        lon = lon[np.newaxis, :]

        # Add time dimension
        time = np.zeros(self.n_along_track)

        # Create dataset
        ds = xr.Dataset()
        ds["lat"] = (("cross_track", "along_track"), lat)
        ds["lon"] = (("cross_track", "along_track"), lon)
        ds["gpm_granule_id"] = np.ones(self.n_along_track)
        ds["time"] = time

        return ds

    @pytest.fixture
    def ds_non_contiguous_lon(
        self,
        ds_contiguous: xr.Dataset,
    ) -> xr.Dataset:
        ds = ds_contiguous.copy(deep=True)

        # Insert one gap
        ds["lon"][0, self.cut_idx :] = ds["lon"][0, self.cut_idx :] + 1

        return ds

    @pytest.fixture
    def ds_non_contiguous_granule_id(
        self,
        ds_contiguous: xr.Dataset,
    ) -> xr.Dataset:
        ds = ds_contiguous.copy(deep=True)

        # Insert one gap
        granule_id = np.ones(self.n_along_track)
        granule_id[self.cut_idx :] = granule_id[self.cut_idx :] + 2
        ds["gpm_granule_id"] = granule_id

        return ds

    @pytest.fixture
    def ds_non_contiguous_both(
        self,
        ds_non_contiguous_granule_id: xr.Dataset,
    ) -> xr.Dataset:
        """Create a dasaset with non contiguous lon and granule_id"""

        ds = ds_non_contiguous_granule_id.copy(deep=True)

        # Insert gap at same location as granule_id
        ds["lon"][0, self.cut_idx :] = ds["lon"][0, self.cut_idx :] + 1

        return ds

    def test_is_contiguous_scans(
        self,
        ds_contiguous: xr.Dataset,
        ds_non_contiguous_lon: xr.Dataset,
        ds_non_contiguous_granule_id: xr.Dataset,
        ds_non_contiguous_both: xr.Dataset,
    ) -> None:
        """Test _is_contiguous_scans"""

        # Test contiguous
        contiguous = checks._is_contiguous_scans(ds_contiguous)
        assert np.all(contiguous)

        contiguous = checks._is_contiguous_scans(ds_non_contiguous_granule_id)
        assert np.all(contiguous)  # lon is contiguous

        # Test non contiguous
        contiguous = checks._is_contiguous_scans(ds_non_contiguous_lon)
        assert np.sum(contiguous) == self.n_along_track - 1
        assert contiguous[self.cut_idx - 1] == False

        contiguous = checks._is_contiguous_scans(ds_non_contiguous_both)
        assert np.sum(contiguous) == self.n_along_track - 1
        assert contiguous[self.cut_idx - 1] == False

    def test_check_contiguous_scans(
        self,
        set_is_orbit_to_true: None,
        ds_contiguous: xr.Dataset,
        ds_non_contiguous_lon: xr.Dataset,
        ds_non_contiguous_granule_id: xr.Dataset,
        ds_non_contiguous_both: xr.Dataset,
    ) -> None:
        """Test check_contiguous_scans"""

        # Test contiguous
        checks.check_contiguous_scans(ds_contiguous)
        # No error raised

        # Test non contiguous
        with pytest.raises(ValueError):
            checks.check_contiguous_scans(ds_non_contiguous_lon)

        with pytest.raises(ValueError):
            checks.check_contiguous_scans(ds_non_contiguous_granule_id)

        with pytest.raises(ValueError):
            checks.check_contiguous_scans(ds_non_contiguous_both)

    def test_has_contiguous_scans(
        self,
        set_is_orbit_to_true: None,
        ds_contiguous: xr.Dataset,
        ds_non_contiguous_lon: xr.Dataset,
        ds_non_contiguous_granule_id: xr.Dataset,
        ds_non_contiguous_both: xr.Dataset,
    ) -> None:
        """Test has_contiguous_scans"""

        assert checks.has_contiguous_scans(ds_contiguous)
        assert not checks.has_contiguous_scans(ds_non_contiguous_lon)
        assert not checks.has_contiguous_scans(ds_non_contiguous_granule_id)
        assert not checks.has_contiguous_scans(ds_non_contiguous_both)


class TestValidGeolocation:
    n_along_track = 10
    invalid_idx = 5

    @pytest.fixture
    def ds_valid(self) -> xr.Dataset:
        # Values along track
        lon = np.arange(self.n_along_track, dtype=float)

        # Add cross-track dimension
        lon = lon[np.newaxis, :]
        time = np.zeros(self.n_along_track)

        ds = xr.Dataset()
        ds["lon"] = (("cross_track", "along_track"), lon)
        ds["time"] = time

        return ds

    @pytest.fixture
    def ds_invalid(
        self,
        ds_valid: xr.Dataset,
    ) -> xr.Dataset:
        ds = ds_valid.copy(deep=True)

        lon = ds["lon"].values.copy()
        lon[0, self.invalid_idx] = np.nan
        ds["lon"] = (("cross_track", "along_track"), lon)

        return ds

    def test_is_valid_geolocation(
        self,
        set_is_orbit_to_true: None,
        ds_valid: xr.Dataset,
        ds_invalid: xr.Dataset,
    ) -> None:
        """Test _is_valid_geolocation"""

        # Valid
        valid = checks._is_valid_geolocation(ds_valid)
        assert np.all(valid)

        # Invalid
        valid = checks._is_valid_geolocation(ds_invalid)
        assert np.sum(valid) == self.n_along_track - 1

    def test_check_valid_geolocation(
        self,
        set_is_orbit_to_true: None,
        ds_valid: xr.Dataset,
        ds_invalid: xr.Dataset,
    ) -> None:
        """Test check_valid_geolocation"""

        # Valid
        checks.check_valid_geolocation(ds_valid)
        # No error raised

        # Invalid
        with pytest.raises(ValueError):
            checks.check_valid_geolocation(ds_invalid)

    def test_has_valid_geolocation(
        self,
        set_is_orbit_to_true: None,
        ds_valid: xr.Dataset,
        ds_invalid: xr.Dataset,
    ) -> None:
        """Test has_valid_geolocation"""

        assert checks.has_valid_geolocation(ds_valid)
        assert not checks.has_valid_geolocation(ds_invalid)


class TestWobblingSwath:
    # Detect changes of direction. Repeated indices are not considered as changes
    #               0  1  2  3  4  5  6xx7xx8xx9  10 11 12
    lat = np.array([0, 1, 2, 2, 2, 3, 4, 3, 4, 3, 4, 5, 6])
    n_along_track = len(lat)
    lon = np.arange(n_along_track)

    # Add cross track dimension
    lat = lat[np.newaxis, :]
    lon = lon[np.newaxis, :]

    # Add time dimension
    time = np.zeros(n_along_track)

    # Create dataset
    ds = xr.Dataset()
    ds["lat"] = (("cross_track", "along_track"), lat)
    ds["lon"] = (("cross_track", "along_track"), lon)
    ds["gpm_granule_id"] = np.ones(n_along_track)
    ds["time"] = time

    # Threshold must be at least 3 to remove wobbling slices
    threshold = 3

    def test_get_slices_non_wobbling_swath(self) -> None:
        """Test get_slices_non_wobbling_swath"""

        returned_slices = checks.get_slices_non_wobbling_swath(self.ds, threshold=self.threshold)
        expected_slices = [slice(0, 6 + 1), slice(9, 12 + 1)]
        assert returned_slices == expected_slices

    def test_get_slices_wobbling_swath(self) -> None:
        """Test get_slices_wobbling_swath"""

        returned_slices = checks.get_slices_wobbling_swath(self.ds, threshold=self.threshold)
        expected_slices = [slice(7, 9)]
        assert returned_slices == expected_slices


class TestIsRegular:
    """Test is_regular"""

    def test_orbit(
        self,
        mocker: MockerFixture,
        set_is_orbit_to_true: None,
    ) -> None:
        # Mock has_contiguous_scans to return True
        mock_has_contiguous_scans = mocker.patch(
            "gpm_api.utils.checks.has_contiguous_scans", return_value=True
        )
        ds = xr.Dataset()
        assert checks.is_regular(ds)

        # Mock has_contiguous_scans to return False
        mock_has_contiguous_scans.return_value = False
        assert not checks.is_regular(ds)

    def test_grid(
        self,
        mocker: MockerFixture,
        set_is_grid_to_true: None,
    ) -> None:
        # Mock has_regular_time to return True
        mock_has_regular_time = mocker.patch(
            "gpm_api.utils.checks.has_regular_time", return_value=True
        )
        ds = xr.Dataset()
        assert checks.is_regular(ds)

        # Mock has_regular_time to return False
        mock_has_regular_time.return_value = False
        assert not checks.is_regular(ds)


def test_get_slices_var_equals() -> None:
    """Test get_slices_var_equals"""

    dim = "dimension"
    array = [0, 0, 1, 1, 2, 3]
    da = xr.DataArray(array, dims=[dim])

    # Test with single value
    values = 0
    returned_slices = checks.get_slices_var_equals(da, dim=dim, values=values)
    expected_slices = [slice(0, 2)]
    assert returned_slices == expected_slices

    # Test with union of slices
    values = [0, 1]
    returned_slices = checks.get_slices_var_equals(da, dim=dim, values=values, union=True)
    expected_slices = [slice(0, 4)]
    assert returned_slices == expected_slices

    # Test without union of slices
    returned_slices = checks.get_slices_var_equals(da, dim=dim, values=values, union=False)
    expected_slices = [slice(0, 2), slice(2, 4)]
    assert returned_slices == expected_slices

    # Test with multiple dimensions
    dim = ["x"]
    array = np.array([[0, 0, 1, 1], [0, 0, 0, 0], [3, 0, 1, 2]])
    da = xr.DataArray(array, dims=["x", "y"])
    returned_slices = checks.get_slices_var_equals(
        da, dim=dim, values=values, union=True, criteria="any"
    )
    expected_slices = [slice(0, 3)]  # All of sub arrays work
    assert returned_slices == expected_slices

    returned_slices = checks.get_slices_var_equals(
        da, dim=dim, values=values, union=True, criteria="all"
    )
    expected_slices = [slice(1, 2)]  # Only [0, 0, 0, 0] sub array works (all values are valid)
    assert returned_slices == expected_slices


def test_get_slices_var_between() -> None:
    """Test get_slices_var_between"""

    vmin = 5
    vmax = 10

    # Test with single dimension
    dim = "dimension"
    array = np.arange(20)
    da = xr.DataArray(array, dims=[dim])
    returned_slices = checks.get_slices_var_between(da, dim=dim, vmin=vmin, vmax=vmax)
    expected_slices = [slice(5, 11)]
    assert returned_slices == expected_slices

    # Test with multiple dimensions
    dim = ["x"]
    array = np.stack((np.arange(20), np.ones(20) * 7))
    da = xr.DataArray(array, dims=["x", "y"])
    returned_slices = checks.get_slices_var_between(
        da, dim=dim, vmin=vmin, vmax=vmax, criteria="any"
    )
    expected_slices = [slice(0, 2)]  # All of sub arrays work
    assert returned_slices == expected_slices

    returned_slices = checks.get_slices_var_between(
        da, dim=dim, vmin=vmin, vmax=vmax, criteria="all"
    )
    expected_slices = [slice(1, 2)]  # Only [7, 7] sub array works (all values are valid)
    assert returned_slices == expected_slices
