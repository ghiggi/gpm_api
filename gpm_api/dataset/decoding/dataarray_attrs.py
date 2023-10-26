#!/usr/bin/env python3
"""
Created on Fri Jul 28 13:50:59 2023

@author: ghiggi
"""
import numpy as np


def convert_string_to_number(string):
    if string.isdigit():
        return int(string)
    else:
        return float(string)


def ensure_dtype_name(dtype):
    """Ensure the dtype is a string name.

    This function convert numpy.dtype to the string name.
    """
    if isinstance(dtype, np.dtype):
        dtype = dtype.name
    return dtype


def _check_fillvalue_format(attrs):
    # Ensure fill values are numbers
    if "CodeMissingValue" in attrs:
        if isinstance(attrs["CodeMissingValue"], str):
            attrs["CodeMissingValue"] = convert_string_to_number(attrs["CodeMissingValue"])
    if "_FillValue" in attrs:
        if isinstance(attrs["_FillValue"], str):
            attrs["_FillValue"] = convert_string_to_number(attrs["_FillValue"])

    # Check _FillValue and CodeMissingValue agrees
    # - Do not since _FillValue often badly defined !
    # - TODO: report issues to NASA team
    # if "_FillValue" in attrs  and "CodeMissingValue" in attrs:
    #     if attrs["_FillValue"] != attrs["CodeMissingValue"]:
    #         name = da.name
    #         fillvalue = attrs["_FillValue"]
    #         codevalue = attrs["CodeMissingValue"]
    #         raise ValueError(f"In {name}, _FillValue is {fillvalue} and CodeMissingValue is {codevalue}")

    # Convert CodeMissingValue' to _FillValue if available
    if "CodeMissingValue" in attrs:
        attrs["_FillValue"] = attrs["CodeMissingValue"]

    # Remove 'CodeMissingValue'
    _ = attrs.pop("CodeMissingValue", None)

    return attrs


def _format_dataarray_attrs(da, product=None):
    attrs = da.attrs

    # Ensure fill values are numbers
    # - If CodeMissingValue is present, it is used as _FillValue
    # - _FillValue are moved to encoding by xr.decode_cf !
    attrs = _check_fillvalue_format(attrs)

    # Convert 'Units' to 'units'
    if not attrs.get("units", False) and attrs.get("Units", False):
        attrs["units"] = attrs.pop("Units")

    # Remove 'Units'
    attrs.pop("Units", None)

    # Remove 'DimensionNames'
    attrs.pop("DimensionNames", None)

    # Ensure encoding and source_dtype is a dtype string name
    if "dtype" in da.encoding:
        da.encoding["dtype"] = ensure_dtype_name(da.encoding["dtype"])

    if "source_dtype" in attrs:
        attrs["source_dtype"] = ensure_dtype_name(attrs["source_dtype"])

    # Add source dtype from encoding if not present
    if "source_dtype" not in attrs and "dtype" in da.encoding:
        attrs["source_dtype"] = da.encoding["dtype"]

    # Add gpm_api product name
    if product is not None:
        attrs["gpm_api_product"] = product

    # Attach attributes
    da.attrs = attrs

    return da


def standardize_dataarrays_attrs(ds, product):
    for var, da in ds.items():
        ds[var] = _format_dataarray_attrs(da, product)
    return ds
