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
"""This module contains functions to standardize GPM-API Dataset attributes."""
import re

import numpy as np


def convert_string_to_number(string):
    if string.isdigit():
        return int(string)
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
    if "CodeMissingValue" in attrs and isinstance(attrs["CodeMissingValue"], str):
        attrs["CodeMissingValue"] = convert_string_to_number(attrs["CodeMissingValue"])
    if "_FillValue" in attrs and isinstance(attrs["_FillValue"], str):
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


def _sanitize_attributes(attrs):
    # Convert 'Units' to 'units'
    if not attrs.get("units", False) and attrs.get("Units", False):
        attrs["units"] = attrs.pop("Units")

    # Remove 'Units'
    attrs.pop("Units", None)

    # Remove 'DimensionNames'
    attrs.pop("DimensionNames", None)

    # Sanitize LongName if present
    if "LongName" in attrs:
        attrs["description"] = re.sub(
            " +",
            " ",
            attrs["LongName"].replace("\n", " ").replace("\t", " "),
        ).strip()
        attrs.pop("LongName")
    return attrs


def _format_dataarray_attrs(da, product=None):
    attrs = da.attrs

    # Ensure fill values are numbers
    # - If CodeMissingValue is present, it is used as _FillValue
    # - _FillValue are moved to encoding by xr.decode_cf !
    attrs = _check_fillvalue_format(attrs)

    # Remove Units, DimensionNames and sanitize LongName
    attrs = _sanitize_attributes(attrs)

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
    # Sanitize variable attributes
    for var, da in ds.items():
        ds[var] = _format_dataarray_attrs(da, product)

    # Drop attributes from bounds coordinates
    # - https://github.com/pydata/xarray/issues/8368
    # - Attribute is lost when writing to netcdf
    bounds_coords = ["time_bnds", "lon_bnds", "lat_bnds"]
    for bnds in bounds_coords:
        if bnds in ds:
            ds[bnds].attrs = {}

    # Sanitize coordinates attributes
    for coord in list(ds.coords):
        ds[coord].attrs = _sanitize_attributes(ds[coord].attrs)

    return ds
