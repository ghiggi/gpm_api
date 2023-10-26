#!/usr/bin/env python3
"""
Created on Mon Aug 15 00:18:13 2022

@author: ghiggi
"""
import glob
import os

from gpm_api.configs import get_gpm_base_dir
from gpm_api.io.directories import get_disk_directory, get_disk_product_directory

####--------------------------------------------------------------------------.


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
    """Define local file path."""
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
