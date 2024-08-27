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
"""This module test the visualization title functions."""

import numpy as np
import pytest
import xarray as xr

from gpm.visualization.title import (
    get_dataarray_title,
    get_dataset_title,
    get_time_str,
)


@pytest.fixture
def time_data():
    """Provide time data for tests."""
    return np.array(["2024-01-01T12:00:00", "2024-01-02T12:00:00", "2024-01-03T12:00:00"], dtype="datetime64")


class TestGetTimeStr:
    """Tests for the get_time_str function."""

    def test_single(self, time_data):
        """Test with a numpy.datetime64 and np.array of size 1 input."""
        assert get_time_str(time_data[0]) == "2024-01-01 12:00"

        assert get_time_str(time_data[[0]]) == "2024-01-01 12:00"

    def test_array_default(self, time_data):
        """Test with an array of timesteps, default middle selection."""
        assert get_time_str(time_data) == "2024-01-02 12:00"

    def test_array_specific_index(self, time_data):
        """Test with an array of timesteps, specific index."""
        assert get_time_str(time_data, time_idx=0) == "2024-01-01 12:00"


class TestGetDatasetTitle:
    """Tests for the get_dataset_title function."""

    def test_without_time(self):
        """Test dataset title without adding time."""
        time = np.array(["2024-01-01T12:00:00"], dtype="datetime64[ns]")
        ds = xr.Dataset({"time": time})
        ds.attrs["gpm_api_product"] = "test product"
        assert get_dataset_title(ds, add_timestep=False) == "Test Product"

    def test_with_time_orbit(self):
        """Test dataset title with time for orbit dataset."""
        time = np.array(["2024-01-01T12:00:00"], dtype="datetime64[ns]")
        ds = xr.Dataset({"time": time})
        ds.attrs["gpm_api_product"] = "2A-DPR"
        assert "2A-DPR (2024-01-01 12:00)" in get_dataset_title(ds, add_timestep=True)

    def test_with_time_grid(self):
        """Test dataset title with time for grid dataset."""
        time = np.array(["2024-01-01T12:00:00"], dtype="datetime64[ns]")
        ds = xr.Dataset({"time": time})
        ds.attrs["gpm_api_product"] = "IMERG-FR"
        assert "IMERG-FR (2024-01-01 12:00)" in get_dataset_title(ds, add_timestep=True)


class TestGetDataArrayTitle:
    """Tests for the get_dataarray_title function."""

    def test_without_prefix(self):
        """Test DataArray title without prefix product."""
        data = 3
        time = np.array(["2024-01-01T12:00:00"], dtype="datetime64[ns]")
        da = xr.DataArray(data, dims=["time"], coords={"time": time})
        da.name = "precipitation"
        da.attrs["gpm_api_product"] = ""
        assert get_dataarray_title(da, prefix_product=False, add_timestep=False) == "Precipitation"

    def test_with_time(self):
        """Test DataArray title with time."""
        data = 3
        time = np.array(["2024-01-01T12:00:00"], dtype="datetime64[ns]")
        da = xr.DataArray(data, dims=["time"], coords={"time": time})
        da.name = "precipitation"
        da.attrs["gpm_api_product"] = "IMERG-FR"
        assert "IMERG-FR Precipitation (2024-01-01 12:00)" in get_dataarray_title(da, add_timestep=True)
