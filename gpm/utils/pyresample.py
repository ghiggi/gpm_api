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
"""This module contains pyresample utility functions."""

import warnings

import numpy as np
import xarray as xr


def remap(src_ds, dst_ds, radius_of_influence=20000, fill_value=np.nan):
    """Remap data from one dataset to another one."""
    try:
        from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler
    except ImportError:
        raise ImportError(
            "The 'pyresample' package is required but not found. "
            "Please install it using the following command: "
            "conda install -c conda-forge pyresample",
        )

    # Retrieve source and destination area
    src_area = src_ds.gpm.pyresample_area
    dst_area = dst_ds.gpm.pyresample_area

    # Rename dimensions to x, y for pyresample compatibility
    if src_ds.gpm.is_orbit:
        src_ds = src_ds.swap_dims({"cross_track": "y", "along_track": "x"})
    else:
        src_ds = src_ds.swap_dims({"lat": "y", "lon": "x"})

    # Define resampler
    resampler = KDTreeNearestXarrayResampler(src_area, dst_area)
    resampler.precompute(radius_of_influence=radius_of_influence)

    # Retrieve valid variables
    variables = [var for var in src_ds.data_vars if set(src_ds[var].dims).issuperset({"x", "y"})]

    # Remap DataArrays
    with warnings.catch_warnings(record=True):
        da_dict = {var: resampler.resample(src_ds[var], fill_value=fill_value) for var in variables}

    # Create Dataset
    ds = xr.Dataset(da_dict)

    # Set correct dimensions
    if dst_ds.gpm.is_orbit:
        ds = ds.swap_dims({"y": "cross_track", "x": "along_track"})
    else:
        ds = ds.swap_dims({"y": "lat", "x": "lon"})

    # Add relevant coordinates of dst_ds
    dst_available_coords = list(dst_ds.coords)
    useful_coords = [coord for coord in dst_available_coords if np.all(np.isin(dst_ds[coord].dims, ds.dims))]
    dict_coords = {coord: dst_ds[coord] for coord in useful_coords}
    return ds.assign_coords(dict_coords)


def get_pyresample_area(xr_obj):
    """It returns the corresponding pyresample area."""
    try:
        import pyresample  # noqa
        from gpm.dataset.crs import get_pyresample_area as _get_pyresample_area
    except ImportError:
        raise ImportError(
            "The 'pyresample' package is required but not found. "
            "Please install it using the following command: "
            "conda install -c conda-forge pyresample",
        )

    # Ensure correct dimension order for Swath
    if "cross_track" in xr_obj.dims:
        xr_obj = xr_obj.transpose("cross_track", "along_track", ...)
    # Return pyresample area
    return _get_pyresample_area(xr_obj)
