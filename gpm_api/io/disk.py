#!/usr/bin/env python3
"""
Created on Mon Aug 15 00:18:13 2022

@author: ghiggi
"""
import datetime
import glob
import os

import pandas as pd

from gpm_api.configs import get_gpm_base_dir
from gpm_api.io.checks import (
    check_date,
    check_product,
    check_product_type,
    check_product_version,
    check_start_end_time,
    check_valid_time_request,
    is_empty,
)
from gpm_api.io.directories import get_disk_directory, get_disk_product_directory
from gpm_api.io.filter import filter_filepaths

####--------------------------------------------------------------------------.
######################
#### Find utility ####
######################


def _get_disk_daily_filepaths(product, product_type, date, version, verbose=True):
    """
    Retrieve GPM data filepaths on the local disk directory of a specific day and product.

    Parameters
    ----------
    product : str
        GPM product acronym. See gpm_api.available_products()
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    date : datetime
        Single date for which to retrieve the data.
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
    verbose : bool, optional
        Whether to print processing details. The default is True.
    """
    # Retrieve the directory on disk where the data are stored
    dir_path = get_disk_directory(
        product=product,
        product_type=product_type,
        date=date,
        version=version,
    )

    # Check if the folder exists
    if not os.path.exists(dir_path):
        return []

    # Retrieve the file names in the directory
    filenames = sorted(os.listdir(dir_path))  # returns [] if empty

    # Retrieve the filepaths
    filepaths = [os.path.join(dir_path, filename) for filename in filenames]

    return filepaths


def get_disk_filepaths(product, product_type, version, base_dir=None):
    """
    Retrieve all GPM filepaths on the local disk directory for a specific product.

    Parameters
    ----------
    product : str
        GPM product acronym. See gpm_api.available_products()
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
    verbose : bool, optional
        Whether to print processing details. The default is True.
    base_dir : str
        The base directory where to store GPM data.
    """
    # Retrieve the directory on disk where the data are stored
    base_dir = get_gpm_base_dir(base_dir)
    product_dir = get_disk_product_directory(
        base_dir=base_dir,
        product=product,
        product_type=product_type,
        version=version,
    )

    # Check if the folder exists
    if not os.path.exists(product_dir):
        return []

    # Retrieve the filepaths
    glob_pattern = os.path.join(product_dir, "*", "*", "*", "*")

    filepaths = sorted(glob.glob(glob_pattern))
    return filepaths


def define_disk_filepath(product, product_type, date, version, filename):
    """Define disk file path."""
    # Define disk directory path
    dir_tree = get_disk_directory(
        product=product,
        product_type=product_type,
        date=date,
        version=version,
    )
    # Define disk file path
    fpath = os.path.join(dir_tree, filename)
    return fpath


def _find_daily_filepaths(
    date,
    product,
    product_type,
    version,
    start_time=None,
    end_time=None,
    verbose=True,
):
    """
    Retrieve GPM data filepaths on a local disk directory for a specific day..

    Parameters
    ----------
    base_dir : str
        The base directory where to store GPM data.
    product : str
        GPM product acronym. See gpm_api.available_products()
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    date : datetime.date
        Single date for which to retrieve the data.
    start_time : datetime.datetime
        Filtering start time.
    end_time : datetime.datetime
        Filtering end time.
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
    verbose : bool, optional
        Whether to print processing details. The default is True.

    Returns
    -------
    filepaths : list
        List of GPM filepaths.

    """
    ##------------------------------------------------------------------------.
    # Check date
    date = check_date(date)

    ##------------------------------------------------------------------------.
    # Retrieve filepaths
    filepaths = _get_disk_daily_filepaths(
        product=product,
        product_type=product_type,
        date=date,
        version=version,
        verbose=verbose,
    )
    if is_empty(filepaths):
        if verbose:
            version_str = str(int(version))
            print(
                f"The GPM product {product} (V0{version_str}) on date {date} has not been downloaded !"
            )
        return []
    ##------------------------------------------------------------------------.
    # Filter the GPM daily file list (for product, version, start_time & end_time)
    filepaths = filter_filepaths(
        filepaths,
        product=product,
        product_type=product_type,
        # version=version, # DO NOT FILTER BY VERSION BECAUSE SOME PMW V7 HAVE VERSION V05D
        start_time=start_time,
        end_time=end_time,
    )

    ##-------------------------------------------------------------------------.
    # Print an optional message if daily data are not available
    # - This message should be removed in future because can pop up when granule is in previous day
    if is_empty(filepaths):
        # if verbose:
        #     version_str = str(int(version))
        #     print(
        #         f"No GPM {product} (V0{version_str}) product has been found on disk on date {date} !"
        #     )
        return []

    ##------------------------------------------------------------------------.
    return filepaths


def find_filepaths(
    product,
    start_time,
    end_time,
    product_type="RS",
    version=None,
    verbose=True,
):
    """
    Retrieve GPM data filepaths on local disk for a specific time period and product.

    Parameters
    ----------
    product : str
        GPM product acronym. See gpm_api.available_products()
    start_time : datetime.datetime
        Start time.
    end_time : datetime.datetime
        End time.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
    verbose : bool, optional
        Whether to print processing details. The default is True.

    Returns
    -------
    filepaths : list
        List of GPM filepaths.

    """
    # -------------------------------------------------------------------------.
    ## Checks input arguments
    version = check_product_version(version, product)

    check_product_type(product_type=product_type)
    check_product(product=product, product_type=product_type)
    start_time, end_time = check_start_end_time(start_time, end_time)
    check_valid_time_request(start_time, end_time, product)
    # Retrieve sequence of dates
    # - Specify start_date - 1 day to include data potentially on previous day directory
    # --> Example granules starting at 23:XX:XX in the day before and extending to 01:XX:XX
    start_date = datetime.datetime(start_time.year, start_time.month, start_time.day)
    start_date = start_date - datetime.timedelta(days=1)
    end_date = datetime.datetime(end_time.year, end_time.month, end_time.day)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    dates = list(date_range.to_pydatetime())

    # -------------------------------------------------------------------------.
    # Loop over dates and retrieve available filepaths
    # TODO:
    # - start_time and end_time filtering could be done only on first and last iteration
    # - misleading error message can occur on first and last iteration if end_time is close to 00:00:00
    #   and the searched granule is in previous day directory
    # - this can be done in parallel !!!
    list_filepaths = []
    verbose_arg = verbose
    for i, date in enumerate(dates):
        verbose = False if i == 0 else verbose_arg
        filepaths = _find_daily_filepaths(
            version=version,
            product=product,
            product_type=product_type,
            date=date,
            start_time=start_time,
            end_time=end_time,
            verbose=verbose,
        )
        list_filepaths += filepaths
    # -------------------------------------------------------------------------.
    # Check unique version
    # - TODO, warning if same integer but different letter
    # - TODO: error if different integer

    # Return filepaths
    filepaths = sorted(list_filepaths)
    return filepaths
