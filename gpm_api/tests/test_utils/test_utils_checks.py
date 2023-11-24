from typing import Union
import numpy as np
import xarray as xr

import pytest
from pytest_mock import MockerFixture

from gpm_api.utils import checks
from utils import convert_hours_array_to_datetime_array


# Utility functions ###########################################################


def create_dataset_with_coordinate(coord_name: str, coord_values: np.ndarray) -> xr.Dataset:
    """Create a dataset with a single coordinate"""

    ds = xr.Dataset()
    ds[coord_name] = coord_values
    return ds


def create_orbit_time_array(time_template: Union[list, np.ndarray]) -> np.ndarray:
    """Create a time array with ORBIT_TIME_TOLERANCE as unit"""

    start_time = np.datetime64("2020-12-31T00:00:00")
    time = np.array([start_time + checks.ORBIT_TIME_TOLERANCE * t for t in time_template])
    return time


@pytest.fixture
def set_is_orbit_to_true(
    mocker: MockerFixture,
) -> None:
    mocker.patch("gpm_api.checks.is_orbit", return_value=True)
    mocker.patch("gpm_api.checks.is_grid", return_value=False)
    mocker.patch("gpm_api.utils.checks.is_orbit", return_value=True)
    mocker.patch("gpm_api.utils.checks.is_grid", return_value=False)


@pytest.fixture
def set_is_grid_to_true(
    mocker: MockerFixture,
) -> None:
    mocker.patch("gpm_api.checks.is_grid", return_value=True)
    mocker.patch("gpm_api.checks.is_orbit", return_value=False)
    mocker.patch("gpm_api.utils.checks.is_grid", return_value=True)
    mocker.patch("gpm_api.utils.checks.is_orbit", return_value=False)


# Tests #######################################################################


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
        time = convert_hours_array_to_datetime_array([0, 1, 2, 7, 8, 9])
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

        time = convert_hours_array_to_datetime_array([0, 1, 2, 7, 8, 9])
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
        time = convert_hours_array_to_datetime_array(np.arange(0, 10))
        tolerance = time[1] - time[0]
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = [slice(0, 10)]
        returned_slices = checks.get_slices_regular_time(ds, tolerance=tolerance)
        assert returned_slices == expected_slices

        # Test irregular time
        time = convert_hours_array_to_datetime_array([0, 1, 2, 7, 8, 9])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = [slice(0, 3), slice(3, 6)]
        returned_slices = checks.get_slices_regular_time(ds, tolerance=tolerance)
        assert returned_slices == expected_slices

        # Test 0 or 1 timesteps
        time = convert_hours_array_to_datetime_array([])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = []
        returned_slices = checks.get_slices_regular_time(ds, tolerance=tolerance)
        assert returned_slices == expected_slices

        time = convert_hours_array_to_datetime_array([0])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = [slice(0, 1)]
        returned_slices = checks.get_slices_regular_time(ds, tolerance=tolerance)
        assert returned_slices == expected_slices

        # Only keep large enough slices
        time = convert_hours_array_to_datetime_array([0, 1, 2, 7, 8])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = [slice(0, 3)]
        returned_slices = checks.get_slices_regular_time(ds, tolerance=tolerance, min_size=3)
        assert returned_slices == expected_slices

    def test_grid(
        self,
        set_is_grid_to_true: None,
    ) -> None:
        # Tolerance not provided: inferred from first two values
        time = convert_hours_array_to_datetime_array([1, 2, 3, 7, 8, 9])
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
        time = convert_hours_array_to_datetime_array(np.arange(0, 10))
        tolerance = time[1] - time[0]
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = []
        returned_slices = checks.get_slices_non_regular_time(ds, tolerance=tolerance)

        # Test irregular time
        #                                             0  1  2  3  4  5  6   7   8
        time = convert_hours_array_to_datetime_array([0, 1, 2, 4, 5, 6, 10, 11, 12])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = [slice(2, 4), slice(5, 7)]  # All slices have length 2
        returned_slices = checks.get_slices_non_regular_time(ds, tolerance=tolerance)
        assert returned_slices == expected_slices

        # Test 0 or 1 timesteps
        time = convert_hours_array_to_datetime_array([])
        ds = create_dataset_with_coordinate("time", time)
        expected_slices = []
        returned_slices = checks.get_slices_non_regular_time(ds, tolerance=tolerance)
        assert returned_slices == expected_slices

        time = convert_hours_array_to_datetime_array([0])
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
        time = convert_hours_array_to_datetime_array([0, 1, 2, 4, 5, 6, 10, 11, 12])
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
        time = convert_hours_array_to_datetime_array(np.arange(0, 10))
        ds = create_dataset_with_coordinate("time", time)
        checks.check_regular_time(ds)

        # Test irregular time
        time = convert_hours_array_to_datetime_array([0, 1, 2, 7, 8, 9])
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


def test_get_slices_contiguous_scans(
    set_is_orbit_to_true: None,
) -> None:
    """Test get_slices_contiguous_scans"""

    # TODO
