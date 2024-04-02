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
"""This module contains functions to parse GPM granule attributes."""
import ast
import datetime

import numpy as np

STATIC_GLOBAL_ATTRS = (
    ## FileHeader
    "DOI",
    "DOIauthority",
    "AlgorithmID",
    "AlgorithmVersion",
    "ProductVersion",
    "SatelliteName",
    "InstrumentName",
    "ProcessingSystem",  # "PPS" or "JAXA"
    # EmptyGranule (granule discarded if empty)
    ## FileInfo,
    "DataFormatVersion",
    "MetadataVersion",
    ## JaxaInfo
    "ProcessingMode",  # "STD", "NRT"
    ## SwathHeader
    "ScanType",
    ## GSMaPInfo
    "AlgorithmName",
    ## GprofInfo
    "Satellite",
    "Sensor",
)


GRANULE_ONLY_GLOBAL_ATTRS = (
    ## FileHeader
    "FileName",
    ## Navigation Record
    "EphemerisFileName",
    "AttitudeFileName",
    ## JaxaInfo
    "TotalQualityCode",
    "DielectricFactorKa",
    "DielectricFactorKu",
)


DYNAMIC_GLOBAL_ATTRS = (
    "MissingData",  # number of missing scans
    "NumberOfRainPixelsFS",
    "NumberOfRainPixelsHS",
)


# TODO: read this dictionary from config YAML ...


def _is_str_list(s):
    """Check if the string start and end with brackets.

    Return a boolean indicating if the string can be converted to a list.

    """
    if s.startswith("[") and s.endswith("]"):
        try:
            ast.literal_eval(s)
            return True
        except ValueError:
            return False
    else:
        return False


def _isfloat(s):
    """Return a boolean indicating if the string can be converted to float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _isinteger(s):
    """Return a boolean indicating if the string can be converted to float."""
    if _isfloat(s):
        return float(s).is_integer()
    return False


def _remove_multiple_spaces(string):
    """Remove consecutive spaces in a string."""
    return " ".join(string.split())


def _parse_attr_string(s):
    """Parse attribute string value.

    This function can return a string, list, integer or float.
    """
    # If there are contiguous spaces, just keep one
    if isinstance(s, str):
        s = _remove_multiple_spaces(s)
    # If multiple stuffs between brackets [ ], convert to list
    if isinstance(s, str) and _is_str_list(s):
        s = ast.literal_eval(s)
    # If still , or \n in a string --> Convert into a list
    if isinstance(s, str) and "," in s:
        s = s.split(",")
    if isinstance(s, str) and "\n" in s:
        s = s.split("\n")
    # If the character can be a number, convert it
    if isinstance(s, str) and _isinteger(s):
        s = int(float(s))  # prior float because '0.0000' otherwise crash
    elif isinstance(s, str) and _isfloat(s):
        s = float(s)
    return s


def decode_string(string):
    r"""Decode string dictionary.

    Format: ``"<key>=<value>\\n".``.

    It removes ``;`` and ``\\t`` prior to parsing the string.
    """
    # Clean the string
    string = string.replace("\t", "").rstrip("\n")
    # Create dictionary if = is present
    if "=" in string:
        list_key_value = [key_value.split("=", 1) for key_value in string.split(";") if len(key_value) > 0]
        value = {key.replace("\n", ""): _parse_attr_string(value) for key, value in list_key_value}
    else:
        value = _parse_attr_string(string)
    return value


def decode_attrs(attrs):
    """Decode GPM nested dictionary attributes from a xarray object."""
    new_dict = {}
    for k, v in attrs.items():
        value = decode_string(v)
        if isinstance(value, dict):
            new_dict[k] = {}
            new_dict[k].update(decode_string(v))
        else:
            new_dict[k] = value
    return new_dict


def _has_nested_dictionary(attrs):
    """Check if the dictionary has nested dictionaries."""
    return np.any([isinstance(v, dict) for v in attrs.values()])


def get_granule_attrs(dt):
    """Get granule global attributes."""
    # Retrieve attributes dictionary (per group)
    nested_attrs = decode_attrs(dt.attrs)
    # Flatten attributes (without group)
    if _has_nested_dictionary(nested_attrs):
        attrs = {}
        _ = [attrs.update(group_attrs) for group, group_attrs in nested_attrs.items()]
    else:
        attrs = nested_attrs
    # Subset only required attributes
    valid_keys = GRANULE_ONLY_GLOBAL_ATTRS + DYNAMIC_GLOBAL_ATTRS + STATIC_GLOBAL_ATTRS
    return {key: attrs[key] for key in valid_keys if key in attrs}


def add_history(ds):
    """Add the history attribute to the xr.Dataset."""
    current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    history = f"Created by ghiggi/gpm_api software on {current_time}"
    ds.attrs["history"] = history
    return ds
