#!/usr/bin/env python3
"""
Created on Tue Jul 25 19:30:55 2023

@author: ghiggi
"""
import os

import h5py

from gpm_api.io.checks import (
    check_product,
    check_start_end_time,
    check_valid_time_request,
)
from gpm_api.io.find import find_filepaths


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


def check_filepaths_integrity(filepaths, remove_corrupted=True, verbose=True):
    """
    Check the integrity of GPM files.

    Parameters
    ----------
    filepaths : list
        List of file paths.
    remove_corrupted : bool, optional
       Whether to remove the corrupted files.
       The default is True.
    verbose : bool, optional
        Whether to verbose the corrupted files. The default is True.

    Returns
    -------
    l_corrupted : list
        List of corrupted file paths.
    """
    # Loop over files and list file that can't be opened
    l_corrupted = get_corrupted_filepaths(filepaths)

    # Report corrupted and remove if asked
    if remove_corrupted:
        remove_corrupted_filepaths(filepaths=l_corrupted, verbose=verbose)
    else:
        for filepath in l_corrupted:
            print(f"{filepath} is corrupted.")

    return l_corrupted


def check_archive_integrity(
    product,
    start_time,
    end_time,
    version=None,
    product_type="RS",
    remove_corrupted=True,
    verbose=True,
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

    Returns
    -------
    filepaths, list
        List of file paths which are corrupted.

    """
    # Check valid product and variables
    check_product(product, product_type=product_type)
    # Check valid start/end time
    start_time, end_time = check_start_end_time(start_time, end_time)
    check_valid_time_request(start_time, end_time, product)
    # Find filepaths
    filepaths = find_filepaths(
        storage="local",
        version=version,
        product=product,
        product_type=product_type,
        start_time=start_time,
        end_time=end_time,
        verbose=False,
    )

    # Check that files have been downloaded on disk
    if len(filepaths) == 0:
        raise ValueError("No files found on disk. Please download them before.")

    # Check the file integrity
    l_corrupted = check_filepaths_integrity(
        filepaths=filepaths, remove_corrupted=remove_corrupted, verbose=verbose
    )
    return l_corrupted
