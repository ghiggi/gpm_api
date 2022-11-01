#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 11:24:29 2022

@author: ghiggi
"""
import os
import h5py
import numpy as np
import pandas as pd 
import xarray as xr
from gpm_api.io.disk import find_filepaths
from gpm_api.io.checks import (
    check_product, 
    check_base_dir,
    check_start_end_time,
)
 
 
###########################
#### Archiving utility ####
###########################


def check_file_integrity(base_dir, 
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
        verbose=verbose, 
    )
    ##---------------------------------------------------------------------.
    # Check that files have been downloaded  on disk
    if len(filepaths) == 0:
        raise ValueError("Requested files are not found on disk. Please download them before.")
    
    ##---------------------------------------------------------------------.
    # Loop over files and list file that can't be opened
    l_corrupted = []
    for filepath in filepaths:
        # Load hdf granule file
        try:
            hdf = h5py.File(filepath, "r")  # h5py._hl.files.File
            hdf.close()
        except OSError:
            if not os.path.exists(filepath):
                raise ValueError("This is a gpm_api bug. `find_GPM_files` should not have returned this filepath.")
            else:
                l_corrupted.append(filepath)
                if remove_corrupted and verbose: 
                    print(f"The following file is corrupted and is being removed: {filepath}.")
                    os.remove(filepath)
    return l_corrupted 


# TODO: from fpath, extract product, start_time, end_time, version, and redownload
