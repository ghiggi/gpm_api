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
"""This module provide tools to extraction information from the granules' filenames."""

import datetime
import os
import re

import numpy as np

from gpm.io.products import get_products_pattern_dict

####---------------------------------------------------------------------------
########################
#### filename PATTERNS ####
########################
# General pattern for all GPM products
NASA_RS_filename_PATTERN = "{product_level:s}.{satellite:s}.{sensor:s}.{algorithm:s}.{start_date:%Y%m%d}-S{start_time:%H%M%S}-E{end_time:%H%M%S}.{granule_id}.{version}.{data_format}"  # noqa
NASA_NRT_filename_PATTERN = "{product_level:s}.{satellite:s}.{sensor:s}.{algorithm:s}.{start_date:%Y%m%d}-S{start_time:%H%M%S}-E{end_time:%H%M%S}.{version}.{data_format}"  # noqa

# General pattern for all JAXA products
# - Pattern for 1B-Ku and 1B-Ka
JAXA_filename_PATTERN = "{mission_id}_{sensor:s}_{start_date_time:%y%m%d%H%M}_{end_time:%H%M}_{granule_id}_{product_level:2s}{product_type}_{algorithm:s}_{version}.{data_format}"  # noqa


####---------------------------------------------------------------------------.
##########################
#### Filename parsers ####
##########################


def _parse_gpm_filename(filename):
    from trollsift import Parser

    # Retrieve information from filename
    try:
        p = Parser(NASA_RS_filename_PATTERN)
        info_dict = p.parse(filename)
        product_type = "RS"
    except ValueError:
        p = Parser(NASA_NRT_filename_PATTERN)
        info_dict = p.parse(filename)
        product_type = "NRT"

    # Retrieve correct start_time and end_time
    start_date = info_dict["start_date"]
    start_time = info_dict["start_time"]
    end_time = info_dict["end_time"]
    start_datetime = start_date.replace(
        hour=start_time.hour,
        minute=start_time.minute,
        second=start_time.second,
    )
    end_datetime = start_date.replace(
        hour=end_time.hour,
        minute=end_time.minute,
        second=end_time.second,
    )
    if end_time < start_time:
        end_datetime = end_datetime + datetime.timedelta(days=1)
    info_dict.pop("start_date")
    info_dict["start_time"] = start_datetime
    info_dict["end_time"] = end_datetime

    # Cast granule_id to integer
    if product_type == "RS":
        info_dict["granule_id"] = int(info_dict["granule_id"])
    return info_dict


def _parse_jaxa_filename(filename):
    from trollsift import Parser

    p = Parser(JAXA_filename_PATTERN)
    info_dict = p.parse(filename)
    # Retrieve correct start_time and end_time
    start_datetime = info_dict["start_date_time"]
    end_time = info_dict["end_time"]
    end_datetime = start_datetime.replace(
        hour=end_time.hour,
        minute=end_time.minute,
        second=end_time.second,
    )
    if end_datetime < start_datetime:
        end_datetime = end_datetime + datetime.timedelta(days=1)
    info_dict.pop("start_date_time")
    info_dict["start_time"] = start_datetime
    info_dict["end_time"] = end_datetime
    # Product type
    product_type = info_dict["product_type"]
    if product_type == "S":
        info_dict["product_type"] = "RS"
    elif product_type == "R":
        info_dict["product_type"] = "NRT"
    else:
        raise ValueError("Report the bug.")

    # Infer satellite
    mission_id = info_dict["mission_id"]
    if "GPM" in mission_id:
        info_dict["satellite"] = "GPM"
    if "TRMM" in mission_id:
        info_dict["satellite"] = "TRMM"

    # Cast granule_id to integer
    info_dict["granule_id"] = int(info_dict["granule_id"])

    return info_dict


def _get_info_from_filename(filename):
    """Retrieve file information dictionary from filename."""
    try:
        info_dict = _parse_gpm_filename(filename)
    except ValueError:
        try:
            info_dict = _parse_jaxa_filename(filename)
        except Exception:
            raise ValueError(f"Impossible to infer file information from '{filename}'")

    # Add product information
    # - ATTENTION: can not be inferred for products not defined in etc/products.yaml
    info_dict["product"] = get_product_from_filepath(filename)

    # Return info dictionary
    return info_dict


def get_info_from_filepath(filepath):
    """Retrieve file information dictionary from filepath."""
    if not isinstance(filepath, str):
        raise TypeError("'filepath' must be a string.")
    filename = os.path.basename(filepath)
    return _get_info_from_filename(filename)


def get_key_from_filepath(filepath, key):
    """Extract specific key information from a list of filepaths."""
    return get_info_from_filepath(filepath)[key]


def get_key_from_filepaths(filepaths, key):
    """Extract specific key information from a list of filepaths."""
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    return [get_key_from_filepath(filepath, key=key) for filepath in filepaths]


####--------------------------------------------------------------------------.
#########################################
#### Product and version information ####
#########################################


def get_product_from_filepath(filepath):
    """Infer granules ``product`` from file path."""
    patterns_dict = get_products_pattern_dict()
    for product, pattern in patterns_dict.items():
        if re.search(pattern, filepath):
            return product
    raise ValueError(f"GPM Product unknown for {filepath}.")


def get_product_from_filepaths(filepaths):
    """Infer granules ``product`` from file paths."""
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    return [get_product_from_filepath(filepath) for filepath in filepaths]


def get_version_from_filepath(filepath, integer=True):
    """Infer granule ``version`` from file path."""
    version = get_key_from_filepath(filepath, key="version")
    if integer:
        version = int(re.findall("\\d+", version)[0])
    return version


def get_version_from_filepaths(filepaths, integer=True):
    """Infer granules ``version`` from file paths."""
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    return [get_version_from_filepath(filepath, integer=integer) for filepath in filepaths]


def get_granule_from_filepaths(filepaths):
    """Infer GPM Granule IDs from file paths."""
    return get_key_from_filepaths(filepaths, key="granule_id")


def get_start_time_from_filepaths(filepaths):
    """Infer granules ``start_time`` from file paths."""
    return get_key_from_filepaths(filepaths, key="start_time")


def get_end_time_from_filepaths(filepaths):
    """Infer granules ``end_time`` from file paths."""
    return get_key_from_filepaths(filepaths, key="end_time")


def get_start_end_time_from_filepaths(filepaths):
    """Infer granules ``start_time`` and ``end_time`` from file paths."""
    list_start_time = get_key_from_filepaths(filepaths, key="start_time")
    list_end_time = get_key_from_filepaths(filepaths, key="end_time")
    return np.array(list_start_time), np.array(list_end_time)


####--------------------------------------------------------------------------.
