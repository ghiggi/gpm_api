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
"""This module provide utility functions used in the unit tests."""

from typing import Union

import numpy as np
import xarray as xr

from gpm.utils import checks as gpm_checks


def create_fake_datetime_array_from_hours_list(hours: Union[list, np.ndarray]) -> np.ndarray:
    """Convert list of integers and NaNs into a np.datetime64 array."""
    start_time = np.array(["2020-12-31 00:00:00"]).astype("M8[ns]")
    hours = np.array(hours).astype("m8[h]")
    return start_time + hours


def get_time_range(start_hour: int, end_hour: int) -> np.ndarray:
    return create_fake_datetime_array_from_hours_list(np.arange(start_hour, end_hour))


def create_dataset_with_coordinate(coord_name: str, coord_values: np.ndarray) -> xr.Dataset:
    """Create a dataset with a single coordinate."""
    ds = xr.Dataset()
    ds[coord_name] = coord_values
    return ds


def create_orbit_time_array(time_template: Union[list, np.ndarray]) -> np.ndarray:
    """Create a time array with ORBIT_TIME_TOLERANCE as unit."""
    start_time = np.datetime64("2020-12-31T00:00:00", "ns")
    return np.array([start_time + gpm_checks.ORBIT_TIME_TOLERANCE * t for t in time_template])
