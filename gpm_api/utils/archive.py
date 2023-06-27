#!/usr/bin/env python3
"""
Created on Tue Nov  1 11:24:29 2022

@author: ghiggi
"""
import datetime
import os
import time
import warnings

import dask
import h5py
import numpy as np
from dateutil.relativedelta import relativedelta

from gpm_api.configs import get_gpm_base_dir, get_gpm_password, get_gpm_username
from gpm_api.io import GPM_VERSION  # CURRENT GPM VERSION
from gpm_api.io.checks import (
    check_base_dir,
    check_product,
    check_start_end_time,
)
from gpm_api.io.disk import find_filepaths
from gpm_api.io.info import (
    get_end_time_from_filepaths,
    get_granule_from_filepaths,
    get_start_time_from_filepaths,
)
from gpm_api.io.pps import find_pps_filepaths
from gpm_api.utils.warnings import GPM_Warning


####--------------------------------------------------------------------------.
###########################
#### Archiving utility ####
###########################
def print_elapsed_time(fn):
    def decorator(*args, **kwargs):
        start_time = time.perf_counter()
        results = fn(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        timedelta_str = str(datetime.timedelta(seconds=execution_time))
        print(f"Elapsed time: {timedelta_str} .", end="\n")
        return results

    return decorator


####--------------------------------------------------------------------------.
#########################
#### Data corruption ####
#########################
def get_corrupted_filepaths(filepaths):
    l_corrupted = []
    for filepath in filepaths:
        # Load hdf granule file
        try:
            hdf = h5py.File(filepath, "r")  # h5py._hl.files.File
            hdf.close()
        except OSError:
            l_corrupted.append(filepath)
    return l_corrupted


def remove_corrupted_filepaths(filepaths, verbose=True):
    for filepath in filepaths:
        if verbose:
            print(f"{filepath} is corrupted and is being removed.")
        os.remove(filepath)


def check_file_integrity(
    product,
    start_time,
    end_time,
    version=GPM_VERSION,
    product_type="RS",
    remove_corrupted=True,
    verbose=True,
    base_dir=None,
):
    """
    Check GPM granule file integrity over a given period.

    If remove_corrupted=True, it removes the corrupted files.

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
        GPM data readers currently support version 4, 5, 6 and 7.
    remove_corrupted : bool, optional
        Whether to remove the corrupted files.
        The default is True.
    base_dir : str, optional
        The path to the GPM base directory. If None, it use the one specified
        in the GPM-API config file.
        The default is None.

    Returns
    -------
    filepaths, list
        List of file paths which are corrupted.

    """
    # Retrieve GPM-API configs
    base_dir = get_gpm_base_dir(base_dir)

    ##--------------------------------------------------------------------.
    # Check base_dir
    base_dir = check_base_dir(base_dir)
    ## Check valid product and variables
    check_product(product, product_type=product_type)
    # Check valid start/end time
    start_time, end_time = check_start_end_time(start_time, end_time)

    ##--------------------------------------------------------------------.
    # Find filepaths
    filepaths = find_filepaths(
        base_dir=base_dir,
        version=version,
        product=product,
        product_type=product_type,
        start_time=start_time,
        end_time=end_time,
        verbose=False,
    )
    ##---------------------------------------------------------------------.
    # Check that files have been downloaded  on disk
    if len(filepaths) == 0:
        raise ValueError("No files found on disk. Please download them before.")

    ##---------------------------------------------------------------------.
    # Loop over files and list file that can't be opened
    l_corrupted = get_corrupted_filepaths(filepaths)

    ##---------------------------------------------------------------------.
    # Report corrupted and remove if asked
    if remove_corrupted:
        remove_corrupted_filepaths(filepaths=l_corrupted, verbose=verbose)
    else:
        for filepath in l_corrupted:
            print(f"{filepath} is corrupted.")

    ##---------------------------------------------------------------------.
    return l_corrupted


####--------------------------------------------------------------------------.
#######################
#### Data coverage ####
#######################
def get_product_temporal_coverage(
    product,
    username,
    version,
    start_year=1997,
    end_year=datetime.datetime.utcnow().year,
    step_months=6,
    verbose=False,
):
    """
    Get product temporal coverage information querying the PPS (text) server.

    Parameters
    ----------
    username: str
        Email address with which you registered on on NASA PPS.
    product : str
        GPM product acronym. See gpm_api.available_products()
    version : int
        GPM version of the product.
    start_year: int
        Year from which to start for data.
        If looking for old SSMI PMW products, you might want to lower it to 1987.
    end_year: int
        Year where to stop to search for data
    step_months: int
        Number of monthly steps by which to search in the first loop.
    verbose : bool, optional
        The default is False.

    Returns
    -------
    info_dict: dict
        It has the following keys:
        - "first_start_time": Start time of the first granule
        - "last_end_time": End time of the last granule
        - "first_granule": PPS file path of the first granule
        - "last_granule": PPS file path of the last granule

    """
    # Routine
    # - Start to loop over years, with a month interval of step_months
    #   --> With step_months=6 --> 25 years records, 2 requests per year = 50 calls
    # - Then search for start time/end_time by
    #   - looping a single day of each month
    #      - all dates within a month where data start to appear
    # Timing notes
    # - Usually it takes about 3 seconds per call
    # - With 8 cores, 50 calls takes about 40 secs

    # --------------------------------------------------------------------------.
    product_type = "RS"
    list_delayed = []
    for year in np.arange(start_year, end_year + 1):
        for month in np.arange(1, 13, step=step_months):
            start_time = datetime.datetime(year, month, 1, 0, 0, 0)
            end_time = start_time + datetime.timedelta(days=1)
            del_op = dask.delayed(find_pps_filepaths)(
                username=username,
                product=product,
                product_type=product_type,
                version=version,
                start_time=start_time,
                end_time=end_time,
                verbose=verbose,
                parallel=False,
            )
            list_delayed.append(del_op)

    # Search filepaths
    list_filepaths = dask.compute(*list_delayed)
    # Flat the list
    tmp_filepaths = [item for sublist in list_filepaths for item in sublist]
    # Retrieve start_time
    l_start_time = get_start_time_from_filepaths(tmp_filepaths)

    if len(l_start_time) > 0:
        raise ValueError(f"No data found for {product}. Reduce step_month parameter.")

    ####----------------------------------
    #### Find accurate start_time
    print("- Searching for first granule")
    first_start_time = min(l_start_time)
    # Loop every month
    start_times = [
        first_start_time - datetime.timedelta(days=31 * m) for m in range(1, step_months + 1)
    ]
    start_times = sorted(start_times)
    for start_time in start_times:
        end_time = start_time + datetime.timedelta(days=1)
        filepaths = find_pps_filepaths(
            username=username,
            product=product,
            product_type=product_type,
            version=version,
            start_time=start_time,
            end_time=end_time,
            verbose=verbose,
            parallel=True,
        )
        # When filepaths starts to occurs, loops within the month
        if len(filepaths) > 0:
            end_time = start_time
            start_time = start_time - datetime.timedelta(days=31)
            filepaths = find_pps_filepaths(
                username=username,
                product=product,
                product_type=product_type,
                version=version,
                start_time=start_time,
                end_time=end_time,
                verbose=verbose,
                parallel=True,
            )
            # Get list of start_time
            granule_start_time = get_start_time_from_filepaths(filepaths)
            # Get idx of first start_time
            idx_min = np.argmin(granule_start_time)
            product_start_time = granule_start_time[idx_min]
            first_pps_filepath = filepaths[idx_min]
            break

    ####----------------------------------
    #### Find accurate end_time
    print("- Searching for last granule")
    last_start_time = max(l_start_time)
    # Loop every month
    start_times = [
        last_start_time + datetime.timedelta(days=31 * m) for m in range(1, step_months + 1)
    ]
    start_times = sorted(start_times)[::-1]
    for start_time in start_times:
        print(start_time)
        end_time = start_time + datetime.timedelta(days=1)
        filepaths = find_pps_filepaths(
            username=username,
            product=product,
            product_type=product_type,
            version=version,
            start_time=start_time,
            end_time=end_time,
            verbose=verbose,
            parallel=True,
        )
        # When filepaths starts to occurs, loops within the month
        if len(filepaths) > 0:
            end_time = start_time + datetime.timedelta(days=31)
            if end_time > datetime.datetime.utcnow():
                end_time = datetime.datetime.utcnow()
            filepaths = find_pps_filepaths(
                username=username,
                product=product,
                product_type=product_type,
                version=version,
                start_time=start_time,
                end_time=end_time,
                verbose=verbose,
                parallel=True,
            )
            # Get list of start_time
            granules_end_time = get_end_time_from_filepaths(filepaths)
            # Get idx of first start_time
            idx_max = np.argmax(granules_end_time)
            product_end_time = granules_end_time[idx_max]
            last_pps_filepath = filepaths[idx_max]
            break
    # Create info dictionary
    info_dict = {}
    info_dict["first_start_time"] = product_start_time
    info_dict["last_end_time"] = product_end_time
    info_dict["first_granule"] = first_pps_filepath
    info_dict["last_granule"] = last_pps_filepath
    return info_dict


####--------------------------------------------------------------------------.
#######################
#### Data download ####
#######################


@print_elapsed_time
def download_daily_data(
    product,
    year,
    month,
    day,
    product_type="RS",
    version=GPM_VERSION,
    n_threads=10,
    transfer_tool="curl",
    progress_bar=False,
    force_download=False,
    check_integrity=True,
    remove_corrupted=True,
    verbose=True,
    retry=1,
    base_dir=None,
    username=None,
    password=None,
):
    from gpm_api.io.download import download_data

    start_time = datetime.date(year, month, day)
    end_time = start_time + relativedelta(days=1)

    l_corrupted = download_data(
        product=product,
        start_time=start_time,
        end_time=end_time,
        product_type=product_type,
        version=version,
        n_threads=n_threads,
        transfer_tool=transfer_tool,
        progress_bar=progress_bar,
        force_download=force_download,
        check_integrity=check_integrity,
        remove_corrupted=remove_corrupted,
        verbose=verbose,
        retry=retry,
        base_dir=base_dir,
        username=username,
        password=password,
    )
    return l_corrupted


@print_elapsed_time
def download_monthly_data(
    product,
    year,
    month,
    product_type="RS",
    version=GPM_VERSION,
    n_threads=10,
    transfer_tool="curl",
    progress_bar=False,
    force_download=False,
    check_integrity=True,
    remove_corrupted=True,
    verbose=True,
    retry=1,
    base_dir=None,
    username=None,
    password=None,
):
    from gpm_api.io.download import download_data

    start_time = datetime.date(year, month, 1)
    end_time = start_time + relativedelta(months=1)

    l_corrupted = download_data(
        product=product,
        start_time=start_time,
        end_time=end_time,
        product_type=product_type,
        version=version,
        n_threads=n_threads,
        transfer_tool=transfer_tool,
        progress_bar=progress_bar,
        force_download=force_download,
        check_integrity=check_integrity,
        remove_corrupted=remove_corrupted,
        verbose=verbose,
        retry=retry,
        base_dir=base_dir,
        username=username,
        password=password,
    )
    return l_corrupted


####--------------------------------------------------------------------------.
##########################
#### Data completness ####
##########################


def check_no_duplicated_files(
    base_dir,
    product,
    start_time,
    end_time,
    version=GPM_VERSION,
    product_type="RS",
    verbose=True,
):
    """Check that there are not duplicated files based on granule number."""
    ##--------------------------------------------------------------------.
    # Find filepaths
    filepaths = find_filepaths(
        base_dir=base_dir,
        version=version,
        product=product,
        product_type=product_type,
        start_time=start_time,
        end_time=end_time,
        verbose=verbose,
    )
    ##---------------------------------------------------------------------.
    # Check that files have been downloaded  on disk
    if len(filepaths) == 0:
        raise ValueError("No files found on disk. Please download them before.")
    ##---------------------------------------------------------------------.
    # Retrieve granule id from filename
    filepaths = np.array(filepaths)
    granule_ids = get_granule_from_filepaths(filepaths)

    # Count granule ids occurence
    ids, counts = np.unique(granule_ids, return_counts=True)

    # Get duplicated indices
    idx_ids_duplicated = np.where(counts > 1)[0].flatten()
    n_duplicated = len(idx_ids_duplicated)
    if n_duplicated > 0:
        duplicated_ids = ids[idx_ids_duplicated]
        for granule_id in duplicated_ids:
            idx_paths_duplicated = np.where(granule_id == granule_ids)[0].flatten()
            tmp_paths_duplicated = filepaths[idx_paths_duplicated].tolist()
            print(f"Granule {granule_id} has duplicated filepaths:")
            for path in tmp_paths_duplicated:
                print(f"- {path}")
        raise ValueError("There are {n_duplicated} duplicated granules.")


def check_time_period_coverage(filepaths, start_time, end_time, raise_error=False):
    """Check time period start_time, end_time is covered.

    If raise_error=True, raise error if time period is not covered.
    If raise_error=False, it raise a GPM warning.

    """
    from gpm_api.io.info import (
        get_end_time_from_filepaths,
        get_start_time_from_filepaths,
    )

    # Check valid start/end time
    start_time, end_time = check_start_end_time(start_time, end_time)

    # Get first and last timestep from filepaths
    filepaths = sorted(filepaths)
    first_start = get_start_time_from_filepaths(filepaths[0])[0]
    last_end = get_end_time_from_filepaths(filepaths[-1])[0]
    # Check time period is covered
    msg = ""
    if first_start > start_time:
        msg = f"The first file start_time ({first_start}) occurs after the specified start_time ({start_time})"

    if last_end < end_time:
        msg1 = (
            f"The last file end_time ({last_end}) occurs before the specified end_time ({end_time})"
        )
        msg = msg + "; and t" + msg1[1:] if msg != "" else msg1
    if msg != "":
        if raise_error:
            raise ValueError(msg)
        else:
            warnings.warn(msg, GPM_Warning)


def get_time_period_with_missing_files(filepaths):
    """
    It returns the time period where the are missing granules.

    It assumes the input filepaths are for a single GPM product.

    Parameters
    ----------
    filepaths : list
        List of GPM file paths.

    Returns
    -------
    list_missing : list
        List of tuple (start_time, end_time).

    """
    from gpm_api.io.info import (
        get_end_time_from_filepaths,
        get_granule_from_filepaths,
        get_start_time_from_filepaths,
    )
    from gpm_api.utils.checks import _is_contiguous_granule
    from gpm_api.utils.slices import get_list_slices_from_bool_arr

    # Retrieve granule id from filename
    granule_ids = get_granule_from_filepaths(filepaths)

    # Sort filepaths by granule number
    indices = np.argsort(granule_ids)
    filepaths = np.array(filepaths)[indices]
    granule_ids = np.array(granule_ids)[indices]

    # Check if next file granule number is +1
    is_not_missing = _is_contiguous_granule(granule_ids)

    # If there are missing files
    list_missing = []
    if np.any(~is_not_missing):
        # Retrieve slices with unmissing granules
        # - Do not skip consecutive False
        # --> is_not_missing=np.array([False, False, True, True, False, False])
        # --> list_slices = [slice(0, 1, None), slice(1, 2, None), slice(2, 5, None), slice(5, 6, None)]
        list_slices = get_list_slices_from_bool_arr(
            is_not_missing, include_false=True, skip_consecutive_false=False
        )
        # Retrieve start and end_time where there are missing files
        for slc in list_slices[0:-1]:
            missing_start = get_end_time_from_filepaths(filepaths[slc.stop - 1])[0]
            missing_end = get_start_time_from_filepaths(filepaths[slc.stop])[0]
            list_missing.append((missing_start, missing_end))
    return list_missing


def check_archive_completness(
    product,
    start_time,
    end_time,
    version=GPM_VERSION,
    product_type="RS",
    download=True,
    transfer_tool="wget",
    n_threads=4,
    verbose=True,
    base_dir=None,
    username=None,
    password=None,
):
    """
    Check that the GPM product archive is not missing granules over a given period.

    This function does not require connection to the PPS to search for the missing files.
    However, the start and end period are based on the first and last file found on disk !

    If download=True, it attempt to download the missing granules.

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
        GPM data readers currently support version 4, 5, 6 and 7.
    download : bool, optional
        Whether to download the missing files.
        The default is True.
    n_threads : int, optional
        Number of parallel downloads. The default is set to 10.
    transfer_tool : str, optional
        Wheter to use curl or wget for data download. The default is "wget".
    verbose : bool, optional
        Whether to print processing details. The default is False.
    base_dir : str, optional
        The path to the GPM base directory. If None, it use the one specified
        in the GPM-API config file.
        The default is None.
    username: str, optional
        Email address with which you registered on the NASA PPS.
        If None, it uses the one specified in the GPM-API config file.
        The default is None.
    password: str, optional
        Email address with which you registered on the NASA PPS.
        If None, it uses the one specified in the GPM-API config file.
        The default is None.
    """
    ##--------------------------------------------------------------------.
    from gpm_api.io.download import download_data

    # -------------------------------------------------------------------------.
    # Retrieve GPM-API configs
    base_dir = get_gpm_base_dir(base_dir)
    username = get_gpm_username(username)
    password = get_gpm_password(password)

    # Check valid start/end time
    start_time, end_time = check_start_end_time(start_time, end_time)

    ##--------------------------------------------------------------------.
    # Find filepaths
    filepaths = find_filepaths(
        base_dir=base_dir,
        version=version,
        product=product,
        product_type=product_type,
        start_time=start_time,
        end_time=end_time,
        verbose=verbose,
    )
    ##---------------------------------------------------------------------.
    # Check that files have been downloaded on disk
    if len(filepaths) == 0:
        raise ValueError("No files found on disk. Please download them before.")

    ##---------------------------------------------------------------------.
    # Check that the specified time period is covered
    check_time_period_coverage(filepaths, start_time, end_time, raise_error=False)

    ##---------------------------------------------------------------------.
    # Loop over files and retrieve time period with missing granules
    list_missing_periods = get_time_period_with_missing_files(filepaths)

    # If there are missing data,
    if len(list_missing_periods) > 0:
        if download:  # and download=True
            # Attempt to download the missing data
            for s_time, e_time in list_missing_periods:
                download_data(
                    base_dir=base_dir,
                    username=username,
                    version=version,
                    product=product,
                    product_type=product_type,
                    start_time=s_time,
                    end_time=e_time,
                    n_threads=n_threads,
                    transfer_tool=transfer_tool,
                    check_integrity=True,
                    remove_corrupted=True,
                    retry=2,
                    verbose=verbose,
                )
        else:
            # Otherwise print time periods with missing data and raise error
            for s_time, e_time in list_missing_periods:
                print(f"- Missing data between {s_time} and {e_time}")
            raise ValueError(
                "The GPM {product} archive is not complete between {start_time} and {end_time}."
            )


####--------------------------------------------------------------------------.
