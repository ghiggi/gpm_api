#!/usr/bin/env python3
"""
Created on Sun Aug 14 20:49:06 2022

@author: ghiggi
"""
import datetime
import os
import re

import numpy as np

from gpm_api.io.products import get_products_pattern_dict

####---------------------------------------------------------------------------
########################
#### FNAME PATTERNS ####
########################
# General pattern for all GPM products
NASA_RS_FNAME_PATTERN = "{product_level:s}.{satellite:s}.{sensor:s}.{algorithm:s}.{start_date:%Y%m%d}-S{start_time:%H%M%S}-E{end_time:%H%M%S}.{granule_id}.{version}.{data_format}"  # noqa
NASA_NRT_FNAME_PATTERN = "{product_level:s}.{satellite:s}.{sensor:s}.{algorithm:s}.{start_date:%Y%m%d}-S{start_time:%H%M%S}-E{end_time:%H%M%S}.{version}.{data_format}"  # noqa

# General pattern for all JAXA products
# - Pattern for 1B-Ku and 1B-Ka
JAXA_FNAME_PATTERN = "{mission_id}_{sensor:s}_{start_date_time:%y%m%d%H%M}_{end_time:%H%M}_{granule_id}_{product_level:2s}{product_type}_{algorithm:s}_{version}.{data_format}"  # noqa


####---------------------------------------------------------------------------.
##########################
#### Filename parsers ####
##########################


def _parse_gpm_fname(fname):
    from trollsift import Parser

    # Retrieve information from filename
    try:
        p = Parser(NASA_RS_FNAME_PATTERN)
        info_dict = p.parse(fname)
        product_type = "RS"
    except ValueError:
        p = Parser(NASA_NRT_FNAME_PATTERN)
        info_dict = p.parse(fname)
        product_type = "NRT"

    # Retrieve correct start_time and end_time
    start_date = info_dict["start_date"]
    start_time = info_dict["start_time"]
    end_time = info_dict["end_time"]
    start_datetime = start_date.replace(
        hour=start_time.hour, minute=start_time.minute, second=start_time.second
    )
    end_datetime = start_date.replace(
        hour=end_time.hour, minute=end_time.minute, second=end_time.second
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


def _parse_jaxa_fname(fname):
    from trollsift import Parser

    p = Parser(JAXA_FNAME_PATTERN)
    info_dict = p.parse(fname)
    # Retrieve correct start_time and end_time
    start_datetime = info_dict["start_date_time"]
    end_time = info_dict["end_time"]
    end_datetime = start_datetime.replace(
        hour=end_time.hour, minute=end_time.minute, second=end_time.second
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


def _get_info_from_filename(fname):
    """Retrieve file information dictionary from filename."""
    try:
        info_dict = _parse_gpm_fname(fname)
    except ValueError:
        try:
            info_dict = _parse_jaxa_fname(fname)
        except:
            raise ValueError(f"{fname} can not be parsed. Report the issue.")

    # Add product information
    # - ATTENTION: can not be inferred for products not defined in etc/product.yml
    info_dict["product"] = get_product_from_filepath(fname)

    # Return info dictionary
    return info_dict


def get_info_from_filepath(fpath):
    """Retrieve file information dictionary from filepath."""
    if not isinstance(fpath, str):
        raise TypeError("'fpath' must be a string.")
    fname = os.path.basename(fpath)
    return _get_info_from_filename(fname)


def get_key_from_filepath(fpath, key):
    """Extract specific key information from a list of filepaths."""
    value = get_info_from_filepath(fpath)[key]
    return value


def get_key_from_filepaths(fpaths, key):
    """Extract specific key information from a list of filepaths."""
    if isinstance(fpaths, str):
        fpaths = [fpaths]
    return [get_key_from_filepath(fpath, key=key) for fpath in fpaths]


####--------------------------------------------------------------------------.
#########################################
#### Product and version information ####
#########################################
def get_product_from_filepath(filepath):
    patterns_dict = get_products_pattern_dict()
    for product, pattern in patterns_dict.items():
        if re.search(pattern, filepath):
            return product
    else:
        raise ValueError(f"GPM Product unknown for {filepath}.")


def get_product_from_filepaths(filepaths):
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    list_product = [get_product_from_filepath(fpath) for fpath in filepaths]
    return list_product


def get_version_from_filepath(filepath, integer=True):
    version = get_key_from_filepath(filepath, key="version")
    if integer:
        version = int(re.findall("\\d+", version)[0])
    return version


def get_version_from_filepaths(filepaths, integer=True):
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    list_version = [get_version_from_filepath(fpath, integer=integer) for fpath in filepaths]
    return list_version


def get_granule_from_filepaths(filepaths):
    list_id = get_key_from_filepaths(filepaths, key="granule_id")
    return list_id


def get_start_time_from_filepaths(filepaths):
    list_start_time = get_key_from_filepaths(filepaths, key="start_time")
    return list_start_time


def get_end_time_from_filepaths(filepaths):
    list_end_time = get_key_from_filepaths(filepaths, key="end_time")
    return list_end_time


def get_start_end_time_from_filepaths(filepaths):
    list_start_time = get_key_from_filepaths(filepaths, key="start_time")
    list_end_time = get_key_from_filepaths(filepaths, key="end_time")
    return np.array(list_start_time), np.array(list_end_time)


####--------------------------------------------------------------------------.
