#!/usr/bin/env python3
"""
Created on Thu Oct 13 11:30:46 2022

@author: ghiggi
"""
import datetime
import re

import numpy as np

from gpm_api.io.checks import (
    check_filepaths,
    check_product,
    check_start_end_time,
    check_version,
)
from gpm_api.io.info import (
    get_info_from_filepath,
    get_start_end_time_from_filepaths,
    get_version_from_filepaths,
)
from gpm_api.io.products import get_product_pattern


def is_granule_within_time(start_time, end_time, file_start_time, file_end_time):
    """Check if a granule is within start_time and end_time."""
    # - Case 1
    #     s               e
    #     |               |
    #   ---------> (-------->)
    is_case1 = file_start_time <= start_time and file_end_time > start_time
    # - Case 2
    #     s               e
    #     |               |
    #          --------
    is_case2 = file_start_time >= start_time and file_end_time < end_time
    # - Case 3
    #     s               e
    #     |               |
    #                ------------->
    is_case3 = file_start_time < end_time and file_end_time > end_time
    # - Check if one of the conditions occurs
    is_within = is_case1 or is_case2 or is_case3
    # - Return boolean
    return is_within


####--------------------------------------------------------------------------.
##########################
#### Filter filepaths ####
##########################


def _string_match(pattern, string):
    """Return True if a string match the pattern. Otherwise False."""
    return bool(re.search(pattern, string))


def _filter_filepath(filepath, product=None, version=None, start_time=None, end_time=None):
    """
    Check if a single filepath pass the filtering parameters.

    If do not match the filtering criteria, it returns None.

    Parameters
    ----------
    filepath : str
        Filepath string.
    product : str
        GPM product name. See: gpm_api.available_products()
        The default is None.
    start_time : datetime.datetime
        Start time
        The default is None.
    end_time : datetime.datetime
        End time.
        The default is None.
    version: int
        GPM product version.
        The default is None.

    Returns
    -------

    filepaths : list
        Returns the filepaths subset.
        If no valid filepaths, return an empty list.

    """

    try:
        info_dict = get_info_from_filepath(filepath)
    except ValueError:
        return None

    # Filter by version
    if version is not None:
        file_version = info_dict["version"]
        file_version = int(re.findall("\\d+", file_version)[0])
        if file_version != version:
            return None

    # Filter by product
    if product is not None:
        product_pattern = get_product_pattern(product)
        if not _string_match(pattern=product_pattern, string=filepath):
            return None

    # Filter by start_time and end_time
    if start_time is not None and end_time is not None:
        file_start_time = info_dict["start_time"]
        file_end_time = info_dict["end_time"]
        if not is_granule_within_time(start_time, end_time, file_start_time, file_end_time):
            return None

    return filepath


def filter_filepaths(
    filepaths,
    product=None,
    product_type=None,
    version=None,
    start_time=None,
    end_time=None,
):
    """
    Filter the GPM filepaths based on specific parameters.

    Parameters
    ----------
    filepaths : list
        List of filepaths.
    product : str
        GPM product name. See: gpm_api.available_products()
        The default is None.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    start_time : datetime.datetime
        Start time
        The default is None.
    end_time : datetime.datetime
        End time.
        The default is None.
    version: int
        GPM product version.
        The default is None.

    Returns
    -------

    filepaths : list
        Returns the filepaths subset.
        If no valid filepaths, return an empty list.

    """
    # Check filepaths
    if isinstance(filepaths, type(None)):
        return []
    filepaths = check_filepaths(filepaths)
    if len(filepaths) == 0:
        return []
    # Check product validity
    check_product(product=product, product_type=product_type)
    # Check start_time and end_time
    if start_time is not None or end_time is not None:
        if start_time is None:
            start_time = datetime.datetime(1998, 1, 1, 0, 0, 0)  # GPM start mission
        if end_time is None:
            end_time = datetime.datetime.now()  # Current time
    # Filter filepaths
    filepaths = [
        _filter_filepath(
            fpath,
            product=product,
            version=version,
            start_time=start_time,
            end_time=end_time,
        )
        for fpath in filepaths
    ]
    # Remove None from the list
    filepaths = [fpath for fpath in filepaths if fpath is not None]
    return filepaths


def filter_by_product(filepaths, product, product_type="RS"):
    """
    Filter filepaths by product.

    Parameters
    ----------
    filepaths : list
        List of filepaths.
    product : str
        GPM product name. See: gpm_api.available_products()
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).

    Returns
    ----------
    filepaths : list
        List of valid filepaths.
        If no valid filepaths, returns an empty list !

    """
    # return filter_filepaths(filepaths, product=product, product_type=product_type)
    # -------------------------------------------------------------------------.
    # Check filepaths
    if isinstance(filepaths, type(None)):
        return []
    filepaths = check_filepaths(filepaths)
    if len(filepaths) == 0:
        return []

    # -------------------------------------------------------------------------.
    # Check product validity
    check_product(product=product, product_type=product_type)

    # -------------------------------------------------------------------------.
    # Retrieve GPM filename dictionary
    product_pattern = get_product_pattern(product)

    # -------------------------------------------------------------------------.
    # Subset by specific product
    filepaths = [
        filepath
        for filepath in filepaths
        if _string_match(pattern=product_pattern, string=filepath)
    ]

    # -------------------------------------------------------------------------.
    # Return valid filepaths
    return filepaths


def filter_by_time(filepaths, start_time=None, end_time=None):
    """
    Filter filepaths by start_time and end_time.

    Parameters
    ----------
    filepaths : list
        List of filepaths.
    start_time : datetime.datetime
        Start time. Will be set to GPM start mission time (1998-01-01) if None.
    end_time : datetime.datetime
        End time. Will be set to current time (`datetime.datetime.utcnow()`) if None.

    Returns
    ----------
    filepaths : list
        List of valid filepaths.
        If no valid filepaths, returns an empty list !
    """
    # return filter_filepaths(filepaths, start_time=start_time, end_time=end_time)
    # -------------------------------------------------------------------------.
    # Check filepaths
    if isinstance(filepaths, type(None)):
        return []
    filepaths = check_filepaths(filepaths)
    if len(filepaths) == 0:
        return []

    # -------------------------------------------------------------------------.
    # Check start_time and end_time
    if start_time is None:
        start_time = datetime.datetime(1998, 1, 1, 0, 0, 0)  # GPM start mission
    if end_time is None:
        end_time = datetime.datetime.utcnow()  # Current time
    start_time, end_time = check_start_end_time(start_time, end_time)

    # -------------------------------------------------------------------------.
    # - Retrieve start_time and end_time of GPM granules
    l_start_time, l_end_time = get_start_end_time_from_filepaths(filepaths)

    # -------------------------------------------------------------------------.
    # Select granules with data within the start and end time
    # - Case 1
    #     s               e
    #     |               |
    #   ---------> (-------->)
    idx_select1 = np.logical_and(l_start_time <= start_time, l_end_time > start_time)
    # - Case 2
    #     s               e
    #     |               |
    #          --------
    idx_select2 = np.logical_and(l_start_time >= start_time, l_end_time < end_time)
    # - Case 3
    #     s               e
    #     |               |
    #                -------------
    idx_select3 = np.logical_and(l_start_time < end_time, l_end_time > end_time)
    # - Get idx where one of the cases occur
    idx_select = np.logical_or(idx_select1, idx_select2, idx_select3)
    # - Select filepaths
    filepaths = list(np.array(filepaths)[idx_select])
    # -------------------------------------------------------------------------.
    return filepaths


def filter_by_version(filepaths, version):
    """
    Filter filepaths by GPM product version.

    Parameters
    ----------
    filepaths : list
        List of filepaths or filenames.
    version: int
        GPM product version.

    Returns
    ----------
    filepaths : list
        List of valid filepaths.
        If no valid filepaths, returns an empty list !
    """
    # return filter_filepaths(filepaths, version=version)
    # -------------------------------------------------------------------------.
    # Check filepaths
    if isinstance(filepaths, type(None)):
        return []
    filepaths = check_filepaths(filepaths)
    if len(filepaths) == 0:
        return []

    # -------------------------------------------------------------------------.
    # Check version validity
    check_version(version)

    # -------------------------------------------------------------------------.
    # Retrieve GPM granules version
    l_version = get_version_from_filepaths(filepaths)

    # -------------------------------------------------------------------------.
    # Select valid filepaths
    idx_select = np.array(l_version) == version
    filepaths = list(np.array(filepaths)[idx_select])
    # -------------------------------------------------------------------------.
    return filepaths
