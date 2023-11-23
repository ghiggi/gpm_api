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


@pytest.fixture
def set_is_orbit_to_true(
    mocker: MockerFixture,
) -> None:
    mocker.patch("gpm_api.checks.is_orbit", return_value=True)
    mocker.patch("gpm_api.checks.is_grid", return_value=False)


@pytest.fixture
def set_is_grid_to_true(
    mocker: MockerFixture,
) -> None:
    mocker.patch("gpm_api.checks.is_grid", return_value=True)
    mocker.patch("gpm_api.checks.is_orbit", return_value=False)


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
        pass


def test_get_slices_regular_time(
    mocker: MockerFixture,
    set_is_grid_to_true: None,
) -> None:
    """Test get_slices_regular_time"""

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

    # Tolerance not provided: inferred from first two values (for grid data)
    time = convert_hours_array_to_datetime_array([1, 2, 3, 7, 8, 9])
    ds = create_dataset_with_coordinate("time", time)
    expected_slices = [slice(0, 3), slice(3, 6)]
    returned_slices = checks.get_slices_regular_time(ds, tolerance=None)
    assert returned_slices == expected_slices
