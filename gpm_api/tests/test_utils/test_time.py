import datetime
import numpy as np
import xarray as xr

import pytest

from gpm_api.utils import time as gpm_time
from utils import create_fake_datetime_array_from_hours_list, get_time_range


N = float("nan")


class TestSubsetByTime:
    """Test subset_by_time"""

    time = get_time_range(0, 24)
    datetime_type_wrappers = [lambda x: x, str, np.datetime64]

    @pytest.fixture
    def data_array(self) -> xr.DataArray:
        return xr.DataArray(np.random.rand(len(self.time)), coords={"time": self.time})

    def test_no_subset(self, data_array: xr.DataArray) -> None:
        returned_da = gpm_time.subset_by_time(data_array, start_time=None, end_time=None)
        xr.testing.assert_equal(data_array["time"], returned_da["time"])

    @pytest.mark.parametrize("type_wrapper", datetime_type_wrappers)
    def test_subset_by_start_time(
        self,
        data_array: xr.DataArray,
        type_wrapper,
    ) -> None:
        start_time = type_wrapper(datetime.datetime(2020, 12, 31, 12, 0, 0))
        returned_da = gpm_time.subset_by_time(data_array, start_time=start_time, end_time=None)
        assert returned_da["time"].values[0] == np.datetime64(start_time)
        assert returned_da["time"].values[-1] == np.datetime64(self.time[-1])
        assert len(returned_da) == len(returned_da["time"])

    @pytest.mark.parametrize("type_wrapper", datetime_type_wrappers)
    def test_subset_by_end_time(
        self,
        data_array: xr.DataArray,
        type_wrapper,
    ) -> None:
        end_time = type_wrapper(datetime.datetime(2020, 12, 31, 12, 0, 0))
        returned_da = gpm_time.subset_by_time(data_array, start_time=None, end_time=end_time)
        assert returned_da["time"].values[0] == np.datetime64(self.time[0])
        assert returned_da["time"].values[-1] == np.datetime64(end_time)
        assert len(returned_da) == len(returned_da["time"])

    @pytest.mark.parametrize("type_wrapper", datetime_type_wrappers)
    def test_subset_by_start_and_end_time(
        self,
        data_array: xr.DataArray,
        type_wrapper,
    ) -> None:
        start_time = type_wrapper(datetime.datetime(2020, 12, 31, 6, 0, 0))
        end_time = type_wrapper(datetime.datetime(2020, 12, 31, 18, 0, 0))
        returned_da = gpm_time.subset_by_time(data_array, start_time=start_time, end_time=end_time)
        assert returned_da["time"].values[0] == np.datetime64(start_time)
        assert returned_da["time"].values[-1] == np.datetime64(end_time)
        assert len(returned_da) == len(returned_da["time"])

    @pytest.mark.parametrize("type_wrapper", datetime_type_wrappers)
    def test_dataset(
        self,
        type_wrapper,
    ) -> None:
        """Test dataset with "time" as variable"""
        ds = xr.Dataset(
            {
                "time": xr.DataArray(self.time, coords={"along_track": np.arange(len(self.time))}),
            }
        )
        start_time = type_wrapper(datetime.datetime(2020, 12, 31, 6, 0, 0))
        end_time = type_wrapper(datetime.datetime(2020, 12, 31, 18, 0, 0))
        returned_ds = gpm_time.subset_by_time(ds, start_time=start_time, end_time=end_time)
        assert returned_ds["time"].values[0] == np.datetime64(start_time)
        assert returned_ds["time"].values[-1] == np.datetime64(end_time)

    def test_no_dimension(self):
        da = xr.DataArray(42)  # Scalar value -> no dimension
        ds = xr.Dataset({"time": da})

        with pytest.raises(ValueError):
            gpm_time.subset_by_time(ds, start_time=None, end_time=None)

    def test_wrong_time_dimension(self):
        lat = np.arange(5)
        lon = np.arange(5)
        time = np.random.rand(len(lat), len(lon)) * 1e9
        time = np.array(time, dtype="datetime64[s]")
        da = xr.DataArray(time, coords=[("lat", lat), ("lon", lon)])
        ds = xr.Dataset({"time": da})

        with pytest.raises(ValueError):
            gpm_time.subset_by_time(ds, start_time=None, end_time=None)

    @pytest.mark.parametrize("type_wrapper", datetime_type_wrappers)
    def test_empty_subsets(
        self,
        data_array: xr.DataArray,
        type_wrapper,
    ) -> None:
        start_time = type_wrapper(datetime.datetime(2021, 1, 1, 0, 0, 0))
        with pytest.raises(ValueError):
            gpm_time.subset_by_time(data_array, start_time=start_time, end_time=None)

        end_time = datetime.datetime(2020, 12, 30, 0, 0, 0)
        with pytest.raises(ValueError):
            gpm_time.subset_by_time(data_array, start_time=None, end_time=end_time)


def test_subset_by_time_slice():
    """Test subset_by_time_slice"""

    time = get_time_range(0, 23)
    da = xr.DataArray(np.random.rand(len(time)), coords={"time": time})
    start_time = datetime.datetime(2020, 12, 31, 6, 0, 0)
    end_time = datetime.datetime(2020, 12, 31, 18, 0, 0)
    time_slice = slice(start_time, end_time)

    returned_da = gpm_time.subset_by_time_slice(da, time_slice)
    assert returned_da["time"].values[0] == np.datetime64(start_time)
    assert returned_da["time"].values[-1] == np.datetime64(end_time)
    assert len(returned_da) == len(returned_da["time"])


def test_is_nat():
    """Test is_nat"""

    assert gpm_time.is_nat(np.datetime64("NaT"))
    assert not gpm_time.is_nat(np.datetime64("2020-01-01"))


def test_has_nat():
    """Test has_nat"""

    time = datetime.datetime(2020, 12, 31, 12, 0, 0)
    nat = np.datetime64("NaT")

    assert gpm_time.has_nat(np.array([time, nat]))
    assert not gpm_time.has_nat(np.array([time, time]))


def test_interpolate_nat():
    """Test interpolate_nat.

    Only method="linear", limit_direction=None, limit_area="inside" are used in gpm_api and tested here.
    """

    kwargs = {"method": "linear", "limit": 5, "limit_direction": None, "limit_area": "inside"}

    # Test with no NaNs
    time = create_fake_datetime_array_from_hours_list(np.arange(0, 10))
    returned_time = gpm_time.interpolate_nat(time, **kwargs)
    np.testing.assert_equal(time, returned_time)

    # Test arrays too small to interpolate
    for hour_list in ([], [N], [1, N]):
        time = create_fake_datetime_array_from_hours_list(hour_list)
        returned_time = gpm_time.interpolate_nat(time, **kwargs)
        np.testing.assert_equal(time, returned_time)

    # Test with outside NaNs (not extrapolated)
    time = create_fake_datetime_array_from_hours_list([N, 1, 2, 3, N])
    returned_time = gpm_time.interpolate_nat(time, **kwargs)
    np.testing.assert_equal(time, returned_time)

    # Test linear interpolation
    time = create_fake_datetime_array_from_hours_list([N, 1, 2, N, N, N, 6, 7, N])
    expected_time = create_fake_datetime_array_from_hours_list([N, 1, 2, 3, 4, 5, 6, 7, N])
    returned_time = gpm_time.interpolate_nat(time, **kwargs)
    np.testing.assert_equal(expected_time, returned_time)

    # Test with gap too large: not all values are filled
    time = create_fake_datetime_array_from_hours_list([N, 1, 2, N, N, N, N, N, N, N, 10, 11, N])
    expected_time = create_fake_datetime_array_from_hours_list(
        [N, 1, 2, 3, 4, 5, 6, 7, N, N, 10, 11, N]
    )
    returned_time = gpm_time.interpolate_nat(time, **kwargs)
    np.testing.assert_equal(expected_time, returned_time)


def test_infill_timesteps():
    """Test infill_timesteps"""

    # Test with no NaNs
    time = create_fake_datetime_array_from_hours_list(np.arange(0, 10))
    returned_time = gpm_time.infill_timesteps(time, limit=5)
    np.testing.assert_equal(time, returned_time)

    # Test arrays too small to interpolate
    for hour_list in ([], [1], [1, 2]):
        time = create_fake_datetime_array_from_hours_list(hour_list)
        returned_time = gpm_time.infill_timesteps(time, limit=5)
        np.testing.assert_equal(time, returned_time)

    for hour_list in ([N], [1, N]):
        time = create_fake_datetime_array_from_hours_list(hour_list)
        with pytest.raises(ValueError):
            gpm_time.infill_timesteps(time, limit=5)

    # Test interpolation
    time = create_fake_datetime_array_from_hours_list([1, 2, N, N, N, 6, 7])
    expected_time = create_fake_datetime_array_from_hours_list([1, 2, 3, 4, 5, 6, 7])
    returned_time = gpm_time.infill_timesteps(time, limit=5)
    np.testing.assert_equal(expected_time, returned_time)

    # Test with gap too large: raise error
    time = create_fake_datetime_array_from_hours_list([1, 2, N, N, N, 6, 7])
    with pytest.raises(ValueError):
        gpm_time.infill_timesteps(time, limit=2)

    # Test with outside NaNs: raise error
    time = create_fake_datetime_array_from_hours_list([N, 1, 2, 3, N])
    with pytest.raises(ValueError):
        gpm_time.infill_timesteps(time, limit=5)

    # Test all NaNs: raise error
    time = create_fake_datetime_array_from_hours_list([N, N, N, N])
    with pytest.raises(ValueError):
        gpm_time.infill_timesteps(time, limit=5)


class TestEnsureTimeValidity:
    """Test ensure_time_validity"""

    time = create_fake_datetime_array_from_hours_list([1, 2, N, N, N, 6, 7])
    expected_time = create_fake_datetime_array_from_hours_list([1, 2, 3, 4, 5, 6, 7])

    def test_with_time_in_dims(self) -> None:
        da = xr.DataArray(np.random.rand(len(self.time)), coords={"time": self.time})
        returned_da = gpm_time.ensure_time_validity(da, limit=5)
        np.testing.assert_equal(self.expected_time, returned_da["time"])

    def test_without_time_in_dims(self) -> None:
        da = xr.DataArray(self.time, dims=["lat"])
        ds = xr.Dataset({"time": da})
        returned_ds = gpm_time.ensure_time_validity(ds, limit=5)
        np.testing.assert_equal(self.expected_time, returned_ds["time"])
