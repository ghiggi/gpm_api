#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:24:29 2022

@author: ghiggi
"""
import os
import h5py
import gpm_api
import time
import dask 
import datetime
import numpy as np
from gpm_api.io.pps import find_pps_filepaths
from gpm_api.io.info import get_start_time_from_filepaths, get_end_time_from_filepaths
from gpm_api.io.disk import find_filepaths
from gpm_api.io.checks import (
    check_product,
    check_base_dir,
    check_start_end_time,
)


###########################
#### Archiving utility ####
###########################


def check_file_integrity(
    base_dir,
    product,
    start_time,
    end_time,
    version=7,
    product_type="RS",
    remove_corrupted=True,
    verbose=True,
):
    """
    Check GPM granule file integrity over a given period.

    Parameters
    ----------
    base_dir : str
       The base directory where GPM data are stored.
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

    Returns
    -------
    filepaths, list
        List of file paths which are corrupted.

    """
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
        raise ValueError(
            "Requested files are not found on disk. Please download them before."
        )

    ##---------------------------------------------------------------------.
    # Loop over files and list file that can't be opened
    l_corrupted = []
    for filepath in filepaths:
        # Load hdf granule file
        try:
            hdf = h5py.File(filepath, "r")  # h5py._hl.files.File
            hdf.close()
        except OSError:
            l_corrupted.append(filepath)
            
    ##---------------------------------------------------------------------.            
    # Report corrupted and remove if asked
    for filepath in l_corrupted:
        if verbose and remove_corrupted:
            print(f"{filepath} is corrupted and is being removed.")
        else: 
            print(f"{filepath} is corrupted.")
        if remove_corrupted:
            os.remove(filepath)
    ##---------------------------------------------------------------------.  
    return l_corrupted


def get_product_temporal_coverage(product, 
                                  username, 
                                  version, 
                                  start_year=1997, 
                                  end_year=datetime.datetime.utcnow().year, 
                                  step_months=6,
                                  verbose=False): 
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
    
    #--------------------------------------------------------------------------.
    product_type = "RS"
    list_delayed = []
    for year in np.arange(start_year, end_year+1): 
        for month in np.arange(1, 13, step=step_months):  
            start_time = datetime.datetime(year, month, 1, 0, 0, 0)
            end_time = start_time + datetime.timedelta(days=1)
            del_op = dask.delayed(find_pps_filepaths)(username=username,
                                                      product=product,  
                                                      product_type=product_type,
                                                      version=version,
                                                      start_time=start_time, 
                                                      end_time=end_time,
                                                      verbose=verbose,
                                                      parallel=False)
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
    start_times = [first_start_time - datetime.timedelta(days=31*m) for m in range(1,step_months+1)]
    start_times = sorted(start_times)
    for start_time in start_times: 
        end_time = start_time + datetime.timedelta(days=1)
        filepaths = find_pps_filepaths(username=username,
                                       product=product,  
                                       product_type=product_type,
                                       version=version,
                                       start_time=start_time, 
                                       end_time=end_time,
                                       verbose=verbose,
                                       parallel=True)
        # When filepaths starts to occurs, loops within the month 
        if len(filepaths) > 0: 
            end_time = start_time 
            start_time = start_time - datetime.timedelta(days=31)
            filepaths = find_pps_filepaths(username=username,
                                           product=product,  
                                           product_type=product_type,
                                           version=version,
                                           start_time=start_time, 
                                           end_time=end_time,
                                           verbose=verbose,
                                           parallel=True)
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
    start_times = [last_start_time + datetime.timedelta(days=31*m) for m in range(1,step_months+1)]
    start_times = sorted(start_times)[::-1]
    for start_time in start_times:
        print(start_time) 
        end_time = start_time + datetime.timedelta(days=1)
        filepaths = find_pps_filepaths(username=username,
                                       product=product,  
                                       product_type=product_type,
                                       version=version,
                                       start_time=start_time, 
                                       end_time=end_time,
                                       verbose=verbose,
                                       parallel=True)
        # When filepaths starts to occurs, loops within the month 
        if len(filepaths) > 0: 
            end_time = start_time + datetime.timedelta(days=31)
            if end_time > datetime.datetime.utcnow():
                end_time = datetime.datetime.utcnow()
            filepaths = find_pps_filepaths(username=username,
                                           product=product,  
                                           product_type=product_type,
                                           version=version,
                                           start_time=start_time, 
                                           end_time=end_time,
                                           verbose=verbose,
                                           parallel=True)
            # Get list of start_time 
            granules_end_time = get_end_time_from_filepaths(filepaths)
            # Get idx of first start_time 
            idx_max = np.argmax(granules_end_time)
            product_end_time = granules_end_time[idx_max]
            last_pps_filepath = filepaths[idx_max] 
            break 
    # Create info dictionary 
    info_dict = {}
    info_dict['first_start_time'] = product_start_time
    info_dict['last_end_time'] = product_end_time
    info_dict['first_granule'] = first_pps_filepath
    info_dict['last_granule'] = last_pps_filepath
    return info_dict


def download_monthly_data(
    base_dir,
    username,
    product,
    year, 
    month,
    product_type="RS",
    version=7,
    n_threads=10,
    transfer_tool="curl",
    progress_bar=False,
    force_download=False,
    check_integrity=True,
    remove_corrupted=True,
    verbose=True,
):
    import datetime
    start_time = datetime.datetime(year, month, 0, 0, 0, 0)
    end_time = datetime.datetime(year, month, 0, 0, 0, 0)
    l_corrupted = gpm_api.download(base_dir=base_dir, 
                                   username=username, 
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
                                   verbose=verbose)
    return l_corrupted


# TODO: from fpath, extract product, start_time, end_time, version, and redownload
