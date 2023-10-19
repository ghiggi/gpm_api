#!/usr/bin/env python3
"""
Created on Thu Oct 13 17:45:37 2022

@author: ghiggi
"""
import datetime
import os
import subprocess
import warnings

import numpy as np
import pandas as pd

from gpm_api.configs import get_gpm_password, get_gpm_username
from gpm_api.io.checks import (
    check_date,
    check_product,
    check_product_type,
    check_product_version,
    check_start_end_time,
    is_empty,
)
from gpm_api.io.directories import get_pps_directory
from gpm_api.io.filter import filter_filepaths
from gpm_api.io.info import get_version_from_filepaths
from gpm_api.io.products import available_products
from gpm_api.utils.warnings import GPMDownloadWarning

VERSION_WARNING = True


def flatten_list(nested_list):
    """Flatten a nested list into a single-level list."""
    return (
        [item for sublist in nested_list for item in sublist]
        if isinstance(nested_list, list)
        else [nested_list]
    )


def ensure_valid_start_date(start_date, product):
    if product == "2A-SAPHIR-MT1-CLIM":
        min_start_date = "2011-10-13 00:00:00"
    elif product in available_products(product_category="PMW"):
        min_start_date = "1987-07-09 00:00:00"
    elif product in available_products(product_category="RADAR") or product in available_products(
        product_category="CMB"
    ):
        min_start_date = "1997-12-07 00:00:00"
    elif "IMERG" in product:
        min_start_date = "2000-06-01 00:00:00"
    else:
        min_start_date = "1987-07-09 00:00:00"
    min_start_date = datetime.datetime.fromisoformat(min_start_date)
    start_date = max(start_date, min_start_date)
    return start_date


def _check_correct_version(filepaths, product, version):
    """Check the file version is correct.

    If 'version' is the last version, we retrieve data from 'gpmalldata' directory.
    But many products are not available to the last version.
    So to archive data correctly on the user side, we check that the file version
    actually match the asked version, and otherwise we download the last available version.
    """
    global VERSION_WARNING  # To just warn once. Maybe to be defined at each download call

    if len(filepaths) == 0:
        return filepaths, version

    files_versions = np.unique(get_version_from_filepaths(filepaths, integer=True)).tolist()
    if len(files_versions) > 1:
        raise ValueError(
            f"Multiple file versions found: {files_versions}. Please report their occurrence !"
        )
    files_version = files_versions[0]
    if files_version != version:
        if VERSION_WARNING:
            VERSION_WARNING = False
            msg = f"The last available version for {product} product is version {files_version}! "
            msg += f"Starting the download of version {files_version}."
            warnings.warn(msg, GPMDownloadWarning)
    return filepaths, files_version


def _get_pps_file_list(username, password, url_file_list, product, date, version, verbose=True):
    """
    Retrieve the filepaths of the files available on the NASA PPS server for a specific day and product.

    The query is done using https !
    The function does not return the full PPS server url, but the filepath
    from the server root: i.e: '/gpmdata/2020/07/05/radar/<...>.HDF5'

    Parameters
    ----------
    username: str
        Email address with which you registered on the NASA PPS.
    password: str
        Password to access the NASA PPS server.
    product : str
        GPM product acronym. See gpm_api.available_products() .
    date : datetime
        Single date for which to retrieve the data.
    verbose : bool, optional
        Default is False. Whether to specify when data are not available for a specific date.
    """
    # Ensure url_file_list ends with "/"
    if url_file_list[-1] != "/":
        url_file_list = url_file_list + "/"

    # Define curl command
    # -k is required with curl > 7.71 otherwise results in "unauthorized access".
    cmd = "curl -k --user " + username + ":" + password + " " + url_file_list

    # Run command
    args = cmd.split()
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = process.communicate()[0].decode()

    # Check if server is available
    if stdout == "":
        version_str = str(int(version))
        print("The PPS server is currently unavailable.")
        print(
            f"This occurred when searching for product {product} (V0{version_str}) at date {date}."
        )
        raise ValueError("Sorry for the incovenience.")

    # Check if data are available
    if stdout[0] == "<":
        if verbose:
            version_str = str(int(version))
            print(f"No data found on PPS on date {date} for product {product} (V0{version_str})")
        return []
    else:
        # Retrieve filepaths
        filepaths = stdout.split()

    # Return file paths
    return filepaths


def _get_pps_daily_filepaths(
    username, password, product, product_type, date, version, verbose=True
):
    """
    Retrieve the complete url to the files available on the NASA PPS server for a specific day and product.

    Parameters
    ----------

    username: str
        Email address with which you registered on the NASA PPS.
    password: str
        Password to access the NASA PPS server.
    product : str
        GPM product acronym. See gpm_api.available_products() .
    date : datetime
        Single date for which to retrieve the data.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
    verbose : bool, optional
        Whether to specify when data are not available for a specific date.
        The default is True.
    """
    # Retrieve server urls of NASA PPS
    (url_data_server, url_file_list) = get_pps_directory(
        product=product, product_type=product_type, date=date, version=version
    )
    # Retrieve filepaths
    # - If empty: return []
    filepaths = _get_pps_file_list(
        username=username,
        password=password,
        url_file_list=url_file_list,
        product=product,
        date=date,
        version=version,
        verbose=verbose,
    )

    # Define the complete url of pps filepaths
    # - Need to remove the starting "/" to each filepath
    pps_fpaths = [os.path.join(url_data_server, filepath[1:]) for filepath in filepaths]

    # Return the pps data server filepaths
    return pps_fpaths


##-----------------------------------------------------------------------------.


def _find_pps_daily_filepaths(
    username,
    password,
    date,
    product,
    product_type,
    version,
    start_time=None,
    end_time=None,
    verbose=False,
):
    """
    Retrieve GPM data filepaths for NASA PPS server for a specific day and product.

    Parameters
    ----------
    username: str
        Email address with which you registered on the NASA PPS.
    password: str
        Password to access the NASA PPS server.
    date : datetime.date
        Single date for which to retrieve the data.
    product : str
        GPM product acronym. See gpm_api.available_products()
    start_time : datetime.datetime
        Filtering start time.
    end_time : datetime.datetime
        Filtering end time.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
    verbose : bool, optional
        Default is False.

    Returns
    -------
    pps_fpaths: list
        List of file paths on the NASA PPS server.
    disk_fpaths: list
        List of file paths on the local disk.

    """
    ##------------------------------------------------------------------------.
    # Check date
    date = check_date(date)

    ##------------------------------------------------------------------------.
    # Retrieve list of available files on pps
    filepaths = _get_pps_daily_filepaths(
        username=username,
        password=password,
        product=product,
        product_type=product_type,
        date=date,
        version=version,
        verbose=verbose,
    )
    if is_empty(filepaths):
        return [], []
    ##------------------------------------------------------------------------.
    # Filter the GPM daily file list (for product, start_time & end time)
    filepaths = filter_filepaths(
        filepaths,
        product=product,
        product_type=product_type,
        version=None,  # important to not filter !
        start_time=start_time,
        end_time=end_time,
    )
    if is_empty(filepaths):
        return [], []

    ## -----------------------------------------------------------------------.
    ## Check correct version and return the available version
    filepaths, available_version = _check_correct_version(
        filepaths=filepaths, product=product, version=version
    )
    return filepaths, [available_version]


##-----------------------------------------------------------------------------.


def find_pps_filepaths(
    product,
    start_time,
    end_time,
    product_type="RS",
    version=None,
    verbose=True,
    parallel=True,
    username=None,
    password=None,
):
    """
    Retrieve GPM data filepaths on NASA PPS server a specific time period and product.

    Parameters
    ----------
    product : str
        GPM product acronym.
    start_time : datetime.datetime
        Start time.
    end_time : datetime.datetime
        End time.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
    parallel : bool, optional
        Whether to loop over dates in parallel.
        The default is True.
    base_dir : str, optional
        The path to the GPM base directory. If None, it use the one specified
        in the GPM-API config file.
        The default is None.
    username: str, optional
        Email address with which you registered on the NASA PPS.
        If None, it uses the one specified in the GPM-API config file.
        The default is None.
    password: str, optional
        Password to access the NASA PPS server.
        If None, it uses the one specified in the GPM-API config file.
        The default is None.

    Returns
    -------
    filepaths : list
        List of GPM filepaths.

    """
    import dask

    # -------------------------------------------------------------------------.
    # Retrieve GPM-API configs
    username = get_gpm_username(username)
    password = get_gpm_password(password)

    # -------------------------------------------------------------------------.
    ## Checks input arguments
    version = check_product_version(version, product)
    check_product_type(product_type=product_type)
    check_product(product=product, product_type=product_type)
    start_time, end_time = check_start_end_time(start_time, end_time)

    # Retrieve sequence of dates
    # - Specify start_date - 1 day to include data potentially on previous day directory
    # --> Example granules starting at 23:XX:XX in the day before and extending to 01:XX:XX
    start_date = datetime.datetime(start_time.year, start_time.month, start_time.day)
    start_date = start_date - datetime.timedelta(days=1)
    start_date = ensure_valid_start_date(start_date, product)
    end_date = datetime.datetime(end_time.year, end_time.month, end_time.day)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    dates = list(date_range.to_pydatetime())

    # If NRT, all data lies in a single directory
    if product_type == "NRT":
        dates = [dates[0]]
        parallel = False
    # -------------------------------------------------------------------------.
    # Loop over dates and retrieve available filepaths
    if parallel:
        list_delayed = []
        for date in dates:
            del_op = dask.delayed(_find_pps_daily_filepaths)(
                username=username,
                password=password,
                version=version,
                product=product,
                product_type=product_type,
                date=date,
                start_time=start_time,
                end_time=end_time,
                verbose=verbose,
            )
            list_delayed.append(del_op)
        # Get filepaths list for each date
        list_filepaths = dask.compute(*list_delayed)
        list_filepaths = [tpl[0] for tpl in list_filepaths]
        filepaths = flatten_list(list_filepaths)
    else:
        # TODO list
        # - start_time and end_time filtering could be done only on first and last iteration
        # - misleading error message can occur on last iteration if end_time is close to 00:00:00
        #   and the searched granule is in previous day directory
        list_filepaths = []
        for date in dates:
            filepaths, _ = _find_pps_daily_filepaths(
                username=username,
                password=password,
                version=version,
                product=product,
                product_type=product_type,
                date=date,
                start_time=start_time,
                end_time=end_time,
                verbose=verbose,
            )
            # Concatenate filepaths
            list_filepaths += filepaths
        filepaths = list_filepaths

    # -------------------------------------------------------------------------.
    # Return filepaths
    filepaths = sorted(filepaths)
    return filepaths
