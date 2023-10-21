#!/usr/bin/env python3
"""
Created on Fri Oct 20 17:21:13 2023

@author: ghiggi
"""
import warnings

import xarray as xr


def apply_cf_decoding(ds):
    """Apply CF decoding to the xarray dataset.

    For more information on CF-decoding, read:
        https://docs.xarray.dev/en/stable/generated/xarray.decode_cf.html
    """
    # Decode with xr.decode_cf
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        ds = xr.decode_cf(ds, decode_timedelta=False)

    # Clean the DataArray attributes and encodings
    for var, da in ds.items():
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
