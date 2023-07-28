#!/usr/bin/env python3
"""
Created on Fri Jul 28 13:50:59 2023

@author: ghiggi
"""


def _format_dataarray_attrs(da, product=None):
    attrs = da.attrs

    # Convert CodeMissingValue' to _FillValue if available
    if not attrs.get("_FillValue", False) and attrs.get("CodeMissingValue", False):
        attrs["_FillValue"] = attrs.pop("CodeMissingValue")

    # Remove 'CodeMissingValue'
    attrs.pop("CodeMissingValue", None)

    # Convert 'Units' to 'units'
    if not attrs.get("units", False) and attrs.get("Units", False):
        attrs["units"] = attrs.pop("Units")

    # Remove 'Units'
    attrs.pop("Units", None)

    # Remove 'DimensionNames'
    attrs.pop("DimensionNames", None)

    # Add source dtype from encoding
    # print(da.name)
    # print(da.encoding)
    if da.encoding.get("dtype", False):
        attrs["source_dtype"] = da.encoding["dtype"]

    # Add gpm_api product name
    if product is not None:
        attrs["gpm_api_product"] = product

    # Attach attributes
    da.attrs = attrs
    return da


def clean_dataarrays_attrs(ds, product):
    for var, da in ds.items():
        ds[var] = _format_dataarray_attrs(da, product)
    return ds
