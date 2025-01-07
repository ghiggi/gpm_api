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
"""This module contains functions to CF-decoding the GPM files."""
import warnings

import numpy as np
import xarray as xr
from packaging import version


def apply_cf_decoding(ds):
    """Apply CF decoding to the xarray dataset.

    For more information on CF-decoding, read:
        https://docs.xarray.dev/en/stable/generated/xarray.decode_cf.html
    """
    # Take care of numpy 2.0 FillValue CF Decoding issue
    # - https://github.com/pydata/xarray/issues/9381
    vars_and_coords = list(ds.data_vars) + list(ds.coords)
    if version.parse(np.__version__) >= version.parse("2.0.0"):
        for var in vars_and_coords:
            if "_FillValue" in ds[var].attrs:
                ds[var].attrs["_FillValue"] = ds[var].data.dtype.type(ds[var].attrs["_FillValue"])

    # At some point with xarray.2023.>8 currently require this
    # --> TODO: update code in decoding attrs ...
    for var in vars_and_coords:
        if "_FillValue" in ds[var].attrs:
            ds[var].encoding["_FillValue"] = ds[var].attrs.pop("_FillValue")

    # Decode with xr.decode_cf
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        ds = xr.decode_cf(ds, decode_timedelta=False)

    # Clean the DataArray attributes and encodings
    for var in ds:
        # When decoding with xr.decode_cf, _FillValue and the source dtype are automatically
        # added to the encoding attribute
        ds[var].attrs.pop("source_dtype", None)
        ds[var].attrs.pop("_FillValue", None)
        # Remove hdf encodings
        ds[var].encoding.pop("szip", None)
        ds[var].encoding.pop("zstd", None)
        ds[var].encoding.pop("bzip2", None)
        ds[var].encoding.pop("blosc", None)
    return ds
