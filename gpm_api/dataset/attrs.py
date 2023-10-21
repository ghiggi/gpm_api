#!/usr/bin/env python3
"""
Created on Tue Jul 18 17:03:49 2023

@author: ghiggi
"""
import ast
import datetime

import numpy as np

from gpm_api.utils.utils_string import (
    str_detect,
    str_isfloat,
    str_isinteger,
    str_islist,
)

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


def parse_attr_string(s):
    """Parse attribute string value."""
    # If multiple stuffs between brackets [ ], convert to list
    if isinstance(s, str) and str_islist(s):
        s = ast.literal_eval(s)
    # If still a comma in a string --> Convert into a list
    if isinstance(s, str) and str_detect(s, ","):
        s = s.split(",")
    if isinstance(s, str) and str_detect(s, "\n"):
        s = s.split("\n")
    # If the character can be a number, convert it
    if isinstance(s, str) and str_isinteger(s):
        s = int(float(s))  # prior float because '0.0000' otherwise crash
    elif isinstance(s, str) and str_isfloat(s):
        s = float(s)
    else:
        s = s
    return s


def decode_string(string):
    """Decode dictionary string.

    Format: "<key>=<value>\n."
    It removes ; and \t prior to parsing the string.
    """
    # Clean the string
    string = string.replace("\t", "").rstrip("\n")

    # Create dictionary if = is present
    if "=" in string:
        list_key_value = [
            key_value.split("=") for key_value in string.split(";") if len(key_value) > 0
        ]
        value = {
            key_value[0].replace("\n", ""): parse_attr_string(key_value[1])
            for key_value in list_key_value
        }
    else:
        value = parse_attr_string(string)
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
    attrs = {key: attrs[key] for key in valid_keys if key in attrs}
    return attrs


def add_history(ds):
    current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    history = f"Created by ghiggi/gpm_api software on {current_time}"
    ds.attrs["history"] = history
    return ds
