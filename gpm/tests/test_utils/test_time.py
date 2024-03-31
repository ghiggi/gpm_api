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
"""This module test the time utilities."""

import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gpm.tests.test_utils.utils import (
    create_fake_datetime_array_from_hours_list,
    get_time_range,
)
from gpm.utils.time import (
    ensure_time_validity,
    get_dataset_start_end_time,
    has_nat,
    infill_timesteps,
    interpolate_nat,
    is_nat,
    regularize_dataset,
    subset_by_time,
    subset_by_time_slice,
)

N = float("nan")


class TestSubsetByTime:
    """Test subset_by_time."""

    time = get_time_range(0, 24)
    datetime_type_wrappers = [lambda x: x, str, np.datetime64]

    @pytest.fixture()
    def data_array(self) -> xr.DataArray:
        return xr.DataArray(np.random.rand(len(self.time)), coords={"time": self.time})

    def test_no_subset(self, data_array: xr.DataArray) -> None:
        returned_da = subset_by_time(data_array, start_time=None, end_time=None)
        xr.testing.assert_equal(data_array["time"], returned_da["time"])

    @pytest.mark.parametrize("type_wrapper", datetime_type_wrappers)
    def test_subset_by_start_time(
        self,
        data_array: xr.DataArray,
        type_wrapper,
    ) -> None:
        start_time = type_wrapper(datetime.datetime(2020, 12, 31, 12, 0, 0))
        returned_da = subset_by_time(data_array, start_time=start_time, end_time=None)
        assert returned_da["time"].to_numpy()[0] == np.datetime64(start_time)
        assert returned_da["time"].to_numpy()[-1] == np.datetime64(self.time[-1])
        assert len(returned_da) == len(returned_da["time"])

    @pytest.mark.parametrize("type_wrapper", datetime_type_wrappers)
    def test_subset_by_end_time(
        self,
        data_array: xr.DataArray,
        type_wrapper,
    ) -> None:
        end_time = type_wrapper(datetime.datetime(2020, 12, 31, 12, 0, 0))
        returned_da = subset_by_time(data_array, start_time=None, end_time=end_time)
        assert returned_da["time"].to_numpy()[0] == np.datetime64(self.time[0])
        assert returned_da["time"].to_numpy()[-1] == np.datetime64(end_time)
        assert len(returned_da) == len(returned_da["time"])

    @pytest.mark.parametrize("type_wrapper", datetime_type_wrappers)
    def test_subset_by_start_and_end_time(
        self,
        data_array: xr.DataArray,
        type_wrapper,
    ) -> None:
        start_time = type_wrapper(datetime.datetime(2020, 12, 31, 6, 0, 0))
        end_time = type_wrapper(datetime.datetime(2020, 12, 31, 18, 0, 0))
        returned_da = subset_by_time(data_array, start_time=start_time, end_time=end_time)
        assert returned_da["time"].to_numpy()[0] == np.datetime64(start_time)
        assert returned_da["time"].to_numpy()[-1] == np.datetime64(end_time)
        assert len(returned_da) == len(returned_da["time"])

    @pytest.mark.parametrize("type_wrapper", datetime_type_wrappers)
    def test_dataset(
        self,
        type_wrapper,
    ) -> None:
        """Test dataset with "time" as variable."""
        ds = xr.Dataset(
            {
                "time": xr.DataArray(self.time, coords={"along_track": np.arange(len(self.time))}),
            },
        )
        start_time = type_wrapper(datetime.datetime(2020, 12, 31, 6, 0, 0))
        end_time = type_wrapper(datetime.datetime(2020, 12, 31, 18, 0, 0))
        returned_ds = subset_by_time(ds, start_time=start_time, end_time=end_time)
        assert returned_ds["time"].to_numpy()[0] == np.datetime64(start_time)
        assert returned_ds["time"].to_numpy()[-1] == np.datetime64(end_time)

    def test_no_dimension(self):
        da = xr.DataArray(42)  # Scalar value -> no dimension
        ds = xr.Dataset({"time": da})

        with pytest.raises(ValueError):
            subset_by_time(ds, start_time=None, end_time=None)

    def test_wrong_time_dimension(self):
        lat = np.arange(5)
        lon = np.arange(5)
        time = np.random.rand(len(lat), len(lon)) * 1e9
        time = np.array(time, dtype="datetime64[ns]")
        da = xr.DataArray(time, coords=[("lat", lat), ("lon", lon)])
        ds = xr.Dataset({"time": da})

        with pytest.raises(ValueError):
            subset_by_time(ds, start_time=None, end_time=None)

    @pytest.mark.parametrize("type_wrapper", datetime_type_wrappers)
    def test_empty_subsets(
        self,
        data_array: xr.DataArray,
        type_wrapper,
    ) -> None:
        start_time = type_wrapper(datetime.datetime(2021, 1, 1, 0, 0, 0))
        with pytest.raises(ValueError):
            subset_by_time(data_array, start_time=start_time, end_time=None)

        end_time = datetime.datetime(2020, 12, 30, 0, 0, 0)
        with pytest.raises(ValueError):
            subset_by_time(data_array, start_time=None, end_time=end_time)


def test_subset_by_time_slice():
    """Test subset_by_time_slice."""
    time = get_time_range(0, 23)
    da = xr.DataArray(np.random.rand(len(time)), coords={"time": time})
    start_time = datetime.datetime(2020, 12, 31, 6, 0, 0)
    end_time = datetime.datetime(2020, 12, 31, 18, 0, 0)
    time_slice = slice(start_time, end_time)

    returned_da = subset_by_time_slice(da, time_slice)
    assert returned_da["time"].to_numpy()[0] == np.datetime64(start_time)
    assert returned_da["time"].to_numpy()[-1] == np.datetime64(end_time)
    assert len(returned_da) == len(returned_da["time"])


def test_is_nat():
    """Test is_nat."""
    assert is_nat(np.datetime64("NaT"))
    assert not is_nat(np.datetime64("2020-01-01"))


def test_has_nat():
    """Test has_nat."""
    time = datetime.datetime(2020, 12, 31, 12, 0, 0)
    nat = np.datetime64("NaT")

    assert has_nat(np.array([time, nat]))
    assert not has_nat(np.array([time, time]))


def test_interpolate_nat():
    """Test interpolate_nat.

    Only method="linear", limit_direction=None, limit_area="inside" are used in gpm and tested here.
    """
    kwargs = {"method": "linear", "limit": 5, "limit_direction": None, "limit_area": "inside"}

    # Test with no NaNs
    time = create_fake_datetime_array_from_hours_list(np.arange(0, 10))
    returned_time = interpolate_nat(time, **kwargs)
    np.testing.assert_equal(time, returned_time)

    # Test arrays too small to interpolate
    for hour_list in ([], [N], [1, N]):
        time = create_fake_datetime_array_from_hours_list(hour_list)
        returned_time = interpolate_nat(time, **kwargs)
        np.testing.assert_equal(time, returned_time)

    # Test with outside NaNs (not extrapolated)
    time = create_fake_datetime_array_from_hours_list([N, 1, 2, 3, N])
    returned_time = interpolate_nat(time, **kwargs)
    np.testing.assert_equal(time, returned_time)

    # Test linear interpolation
    time = create_fake_datetime_array_from_hours_list([N, 1, 2, N, N, N, 6, 7, N])
    expected_time = create_fake_datetime_array_from_hours_list([N, 1, 2, 3, 4, 5, 6, 7, N])
    returned_time = interpolate_nat(time, **kwargs)
    np.testing.assert_equal(expected_time, returned_time)

    # Test with gap too large: not all values are filled
    time = create_fake_datetime_array_from_hours_list([N, 1, 2, N, N, N, N, N, N, N, 10, 11, N])
    expected_time = create_fake_datetime_array_from_hours_list(
        [N, 1, 2, 3, 4, 5, 6, 7, N, N, 10, 11, N],
    )
    returned_time = interpolate_nat(time, **kwargs)
    np.testing.assert_equal(expected_time, returned_time)


def test_infill_timesteps():
    """Test infill_timesteps."""
    # Test with no NaNs
    time = create_fake_datetime_array_from_hours_list(np.arange(0, 10))
    returned_time = infill_timesteps(time, limit=5)
    np.testing.assert_equal(time, returned_time)

    # Test arrays too small to interpolate
    for hour_list in ([], [1], [1, 2]):
        time = create_fake_datetime_array_from_hours_list(hour_list)
        returned_time = infill_timesteps(time, limit=5)
        np.testing.assert_equal(time, returned_time)

    for hour_list in ([N], [1, N]):
        time = create_fake_datetime_array_from_hours_list(hour_list)
        with pytest.raises(ValueError):
            infill_timesteps(time, limit=5)

    # Test interpolation
    time = create_fake_datetime_array_from_hours_list([1, 2, N, N, N, 6, 7])
    expected_time = create_fake_datetime_array_from_hours_list([1, 2, 3, 4, 5, 6, 7])
    returned_time = infill_timesteps(time, limit=5)
    np.testing.assert_equal(expected_time, returned_time)

    # Test with gap too large: raise error
    time = create_fake_datetime_array_from_hours_list([1, 2, N, N, N, 6, 7])
    with pytest.raises(ValueError):
        infill_timesteps(time, limit=2)

    # Test with outside NaNs: raise error
    time = create_fake_datetime_array_from_hours_list([N, 1, 2, 3, N])
    with pytest.raises(ValueError):
        infill_timesteps(time, limit=5)

    # Test all NaNs: raise error
    time = create_fake_datetime_array_from_hours_list([N, N, N, N])
    with pytest.raises(ValueError):
        infill_timesteps(time, limit=5)


class TestEnsureTimeValidity:
    """Test ensure_time_validity."""

    time = create_fake_datetime_array_from_hours_list([1, 2, N, N, N, 6, 7])
    expected_time = create_fake_datetime_array_from_hours_list([1, 2, 3, 4, 5, 6, 7])

    def test_with_time_in_dims(self) -> None:
        da = xr.DataArray(np.random.rand(len(self.time)), coords={"time": self.time})
        returned_da = ensure_time_validity(da, limit=5)
        np.testing.assert_equal(self.expected_time, returned_da["time"])

    def test_without_time_in_dims(self) -> None:
        da = xr.DataArray(self.time, dims=["lat"])
        ds = xr.Dataset({"time": da})
        returned_ds = ensure_time_validity(ds, limit=5)
        np.testing.assert_equal(self.expected_time, returned_ds["time"])


def create_test_dataset():
    """Create a mock xarray.Dataset for testing."""
    times = pd.date_range("2023-01-01", periods=10, freq="D")
    data = np.random.rand(10, 2, 2)  # Random data for the sake of example
    return xr.Dataset({"my_data": (("time", "x", "y"), data)}, coords={"time": times})


def test_get_dataset_start_end_time():
    ds = create_test_dataset()
    expected_start_time = ds["time"].to_numpy()[0]
    expected_end_time = ds["time"].to_numpy()[-1]

    start_time, end_time = get_dataset_start_end_time(ds)

    assert start_time == expected_start_time
    assert end_time == expected_end_time

    # Test raise if empty dataset
    empty_ds = xr.Dataset()
    with pytest.raises(KeyError):
        get_dataset_start_end_time(empty_ds)


def test_regularize_dataset():
    # Create a sample Dataset
    times = pd.date_range("2020-01-01", periods=4, freq="2min")
    data = np.random.rand(4)
    ds = xr.Dataset({"data": ("time", data)}, coords={"time": times})

    # Regularize the dataset
    desired_freq = "1min"
    fill_value = 0
    ds_regularized = regularize_dataset(ds, freq=desired_freq, fill_value=fill_value)

    # Check new time dimension coordinates
    expected_times = pd.date_range("2020-01-01", periods=7, freq=desired_freq)
    assert np.array_equal(ds_regularized["time"].to_numpy(), expected_times)

    # Get time index which were infilled
    new_indices = np.where(np.isin(expected_times, ds["time"].to_numpy(), invert=True))[0]
    assert np.all(ds_regularized.isel(time=new_indices)["data"].data == fill_value)
