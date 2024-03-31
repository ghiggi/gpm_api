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
"""This module contains functions to find data on local and NASA servers."""
import datetime
import os
import warnings

import dask
import numpy as np
import pandas as pd

from gpm._config import config
from gpm.io.checks import (
    check_date,
    check_product,
    check_product_type,
    check_product_version,
    check_start_end_time,
    check_storage,
    check_valid_time_request,
)
from gpm.io.filter import filter_filepaths
from gpm.io.ges_disc import get_ges_disc_daily_filepaths
from gpm.io.info import get_version_from_filepaths
from gpm.io.local import get_local_daily_filepaths
from gpm.io.pps import get_pps_daily_filepaths
from gpm.io.products import available_products
from gpm.utils.list import flatten_list
from gpm.utils.warnings import GPMDownloadWarning

VERSION_WARNING = config.get("warn_multiple_product_versions")


def _get_all_daily_filepaths(storage, date, product, product_type, version, verbose):
    """Return the ``find_daily_filepaths_func``.

    This functions returns a tuple ``([filepaths],[available_version])``.
    """
    if storage == "LOCAL":
        filepaths = get_local_daily_filepaths(
            product=product,
            product_type=product_type,
            date=date,
            version=version,
        )
    elif storage == "PPS":
        filepaths = get_pps_daily_filepaths(
            product=product,
            product_type=product_type,
            date=date,
            version=version,
            verbose=verbose,
        )
    elif storage == "GES_DISC":
        filepaths = get_ges_disc_daily_filepaths(
            product=product,
            product_type=product_type,
            date=date,
            version=version,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Invalid storage {storage}.")
    return filepaths


def _check_correct_version(filepaths, product, version):
    """Check the file version is correct.

    Several products are not available to the last version.
    So to archive data correctly on the user side, we check that the file version
    actually match the asked version, and otherwise we download the last available version.
    """
    # FIXME: Maybe to be defined/reset at each download call
    global VERSION_WARNING  # To just warn once

    if len(filepaths) == 0:
        return filepaths, version

    files_versions = np.unique(get_version_from_filepaths(filepaths, integer=True)).tolist()
    if len(files_versions) > 1:
        raise ValueError(
            f"Multiple file versions found: {files_versions}. Please report their occurrence !",
        )
    files_version = files_versions[0]
    if files_version != version and VERSION_WARNING:
        VERSION_WARNING = False
        msg = f"The last available version for {product} product is version {files_version}! "
        msg += f"Starting the download of version {files_version}."
        warnings.warn(msg, GPMDownloadWarning, stacklevel=2)
    return filepaths, files_version


def _ensure_valid_start_date(start_date, product):
    """Ensure that the product directory exists for start_date."""
    if product == "2A-SAPHIR-MT1-CLIM":
        min_start_date = "2011-10-13 00:00:00"
    elif "1A-" in product or "1B-" in product:
        min_start_date = "1997-12-07 00:00:00"
    elif product in available_products(product_categories="PMW"):
        min_start_date = "1987-07-09 00:00:00"
    elif product in available_products(product_categories="RADAR") or product in available_products(
        product_categories="CMB",
    ):
        min_start_date = "1997-12-07 00:00:00"
    elif "IMERG" in product:
        min_start_date = "2000-06-01 00:00:00"
    else:
        min_start_date = "1987-07-09 00:00:00"
    min_start_date = datetime.datetime.fromisoformat(min_start_date)
    return max(start_date, min_start_date)


def find_daily_filepaths(
    storage,
    date,
    product,
    product_type,
    version,
    start_time=None,
    end_time=None,
    verbose=False,
):
    """Retrieve GPM data filepaths for a specific day and product.

    Parameters
    ----------
    date : datetime.date
        Single date for which to retrieve the data.
    product : str
        GPM product acronym. See ``gpm.available_products()``.
    start_time : datetime.datetime
        Filtering start time.
    end_time : datetime.datetime
        Filtering end time.
    product_type : str, optional
        GPM product type. Either ``RS`` (Research) or ``NRT`` (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if ``product_type = "RS"``.
    verbose : bool, optional
        Default is ``False``.

    Returns
    -------
    pps_filepaths: list
        List of file paths on the NASA PPS server.
    available_version: list
        List of available versions.

    """
    ##------------------------------------------------------------------------.
    # Check date
    date = check_date(date)

    ##------------------------------------------------------------------------.
    # Retrieve list of available files on pps
    filepaths = _get_all_daily_filepaths(
        storage=storage,
        product=product,
        product_type=product_type,
        date=date,
        version=version,
        verbose=verbose,
    )
    if len(filepaths) == 0:
        if storage == "LOCAL" and verbose:
            version_str = str(int(version))
            print(
                f"The GPM product {product} (V0{version_str}) on date {date} has not been downloaded !",
            )
        return [], []

    ##------------------------------------------------------------------------.
    # Filter the GPM daily file list (for product, start_time & end time)
    # - The version mismatch is raised later eventually !
    filepaths = filter_filepaths(
        filepaths,
        product=product,
        product_type=product_type,
        version=None,  # important to not filter !
        start_time=start_time,
        end_time=end_time,
    )
    if len(filepaths) == 0:
        return [], []

    ## -----------------------------------------------------------------------.
    ## Check correct version and return the available version
    filepaths, available_version = _check_correct_version(
        filepaths=filepaths,
        product=product,
        version=version,
    )
    return filepaths, [available_version]


def find_filepaths(
    storage,
    product,
    start_time,
    end_time,
    product_type="RS",
    version=None,
    verbose=True,
    parallel=True,
):
    """Retrieve GPM data filepaths on local disk for a specific time period and product.

    Parameters
    ----------
    product : str
        GPM product acronym. See ``gpm.available_products()``.
    start_time : datetime.datetime
        Start time.
    end_time : datetime.datetime
        End time.
    product_type : str, optional
        GPM product type. Either ``RS`` (Research) or ``NRT`` (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if ``product_type = "RS"``.
    verbose : bool, optional
        Whether to print processing details. The default is ``True``.
    parallel : bool, optional
        Whether to loop over dates in parallel.
        The default is ``True``.

    Returns
    -------
    filepaths : list
        List of GPM filepaths.

    """
    # -------------------------------------------------------------------------.
    ## Checks input arguments
    storage = check_storage(storage)
    version = check_product_version(version, product)
    product_type = check_product_type(product_type=product_type)
    product = check_product(product=product, product_type=product_type)
    start_time, end_time = check_start_end_time(start_time, end_time)
    start_time, end_time = check_valid_time_request(start_time, end_time, product)

    # Retrieve sequence of dates
    # - Specify start_date - 1 day to include data potentially on previous day directory
    # --> Example granules starting at 23:XX:XX in the day before and extending to 01:XX:XX
    start_date = datetime.datetime(start_time.year, start_time.month, start_time.day)
    start_date = start_date - datetime.timedelta(days=1)
    start_date = _ensure_valid_start_date(start_date=start_date, product=product)
    end_date = datetime.datetime(end_time.year, end_time.month, end_time.day)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    dates = list(date_range.to_pydatetime())

    # -------------------------------------------------------------------------.
    # If NRT, all data lies in a single directory at PPS
    if storage == "PPS" and product_type == "NRT":
        dates = [dates[0]]
        parallel = False

    # -------------------------------------------------------------------------.
    # Loop over dates and retrieve available filepaths
    if parallel:
        list_delayed = []
        verbose_arg = verbose
        for i, date in enumerate(dates):
            verbose = False if i == 0 else verbose_arg
            del_op = dask.delayed(find_daily_filepaths)(
                storage=storage,
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
        list_filepaths = [tpl[0] for tpl in list_filepaths]  # tpl[1] is the available version

    else:
        # TODO list
        # - start_time and end_time filtering could be done only on first and last iteration
        # - misleading error message can occur on last iteration if end_time is close to 00:00:00
        #   and the searched granule is in previous day directory
        list_filepaths = []
        verbose_arg = verbose
        for i, date in enumerate(dates):
            verbose = False if i == 0 else verbose_arg
            filepaths, _ = find_daily_filepaths(
                storage=storage,
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

    filepaths = flatten_list(list_filepaths)

    # -------------------------------------------------------------------------.
    # Check unique version
    # - TODO, warning if same integer but different letter
    # - TODO: error if different integer

    # -------------------------------------------------------------------------.
    # Return sorted filepaths
    return sorted(filepaths)


def find_associated_filepath(filepath, product, storage="LOCAL", product_type="RS", version=7):
    """Return filepath of another product associated to the input filepath.

    Please specify a product derived from the same sensor !
    """
    from gpm.io.info import get_key_from_filepath

    filename = os.path.basename(filepath)
    start_time = get_key_from_filepath(filename, key="start_time")
    end_time = get_key_from_filepath(filename, key="end_time")
    date = start_time.date()
    candidate_filepaths, _ = find_daily_filepaths(
        storage=storage,
        date=date,
        product=product,
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
        verbose=False,
    )
    if len(candidate_filepaths) == 0:
        raise ValueError(f"The {product} product associated to {filename} is not available.")
    if len(candidate_filepaths) >= 2:
        raise ValueError(
            f"The are multiple {product} files associated to the {filename} file:  {candidate_filepaths}.",
        )
    return candidate_filepaths[0]
