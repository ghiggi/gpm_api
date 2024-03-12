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
import pytest
import numpy as np
import xarray as xr
from typing import Union
from gpm.utils import decorators


def test_check_has_cross_track_dimension() -> None:
    """Test check_has_cross_track_dimension decorator"""

    @decorators.check_has_cross_track_dimension
    def identity(xr_obj: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
        return xr_obj

    # Test with cross_track no error is raised
    da = xr.DataArray(np.arange(10), dims=["cross_track"])
    identity(da)

    # Test without cross_track
    da = xr.DataArray(np.arange(10))
    with pytest.raises(ValueError):
        identity(da)


def test_check_has_along_track_dimension() -> None:
    """Test check_has_along_track_dimension decorator"""

    @decorators.check_has_along_track_dimension
    def identity(xr_obj: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
        return xr_obj

    # Test with cross_track no error is raised
    da = xr.DataArray(np.arange(10), dims=["along_track"])
    identity(da)

    # Test without cross_track
    da = xr.DataArray(np.arange(10))
    with pytest.raises(ValueError):
        identity(da)
