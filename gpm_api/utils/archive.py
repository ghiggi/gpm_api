#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:24:29 2022

@author: ghiggi
"""
import os
import h5py
import gpm_api
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
