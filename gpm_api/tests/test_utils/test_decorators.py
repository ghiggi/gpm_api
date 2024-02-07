#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:56:02 2024

@author: ghiggi
"""
import pytest
import numpy as np
import xarray as xr
from typing import Union
from gpm_api.utils import decorators


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
