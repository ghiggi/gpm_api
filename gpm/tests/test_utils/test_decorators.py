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
"""This module test the function decorators."""

import numpy as np
import pytest
import xarray as xr

from gpm.utils.decorators import (
    check_has_along_track_dimension,
    check_has_cross_track_dimension,
    check_software_availability,
)


def test_check_has_cross_track_dimension() -> None:
    """Test check_has_cross_track_dimension decorator."""

    @check_has_cross_track_dimension
    def identity(xr_obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        return xr_obj

    # Test with cross_track no error is raised
    da = xr.DataArray(np.arange(10), dims=["cross_track"])
    identity(da)

    # Test without cross_track
    da = xr.DataArray(np.arange(10))
    with pytest.raises(ValueError):
        identity(da)


def test_check_has_along_track_dimension() -> None:
    """Test check_has_along_track_dimension decorator."""

    @check_has_along_track_dimension
    def identity(xr_obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        return xr_obj

    # Test with cross_track no error is raised
    da = xr.DataArray(np.arange(10), dims=["along_track"])
    identity(da)

    # Test without cross_track
    da = xr.DataArray(np.arange(10))
    with pytest.raises(ValueError):
        identity(da)


def test_check_software_availability_decorator():
    """Test check_software_availability_decorator raise ImportError."""

    @check_software_availability(software="dummy_package", conda_package="dummy_package")
    def dummy_function(a, b=1):
        return a, b

    with pytest.raises(ImportError):
        dummy_function()

    @check_software_availability(software="numpy", conda_package="numpy")
    def dummy_function(a, b=1):
        return a, b

    assert dummy_function(2, b=3) == (2, 3)
