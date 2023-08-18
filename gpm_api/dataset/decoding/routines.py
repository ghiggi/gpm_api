#!/usr/bin/env python3
"""
Created on Fri Jul 28 14:03:18 2023

@author: ghiggi
"""
import warnings

import xarray as xr

from gpm_api.dataset.decoding.attrs import clean_dataarrays_attrs
from gpm_api.dataset.decoding.coordinates import set_coordinates
from gpm_api.dataset.decoding.variables import decode_variables

# TODO REFACTORING:
# In finalize_dataset (If variable attrs not changing across granules)
#  - First clean attributes .. for _FillValue, FillValue, etc  !
#  - Then do cf decoding
#  - Then apply custom decoding
# --> Currently done per granule !

# -----------------------------------------------------------------------------.


def decode_dataset(ds):
    """CF decode the xarray dataset.

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


def apply_custom_decoding(ds, product, scan_mode):
    """Ensure correct decoding of dataset coordinates."""
    # Clean attributes
    ds = clean_dataarrays_attrs(ds, product)

    # Decode dataset
    # ds = decode_dataset(ds) # in future ... see TODO above

    # Set relevant coordinates
    ds = set_coordinates(ds, product, scan_mode)

    # Decode variables
    ds = decode_variables(ds, product)

    return ds
