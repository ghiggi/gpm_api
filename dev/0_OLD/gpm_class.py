#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 19:59:58 2022

@author: ghiggi
"""
from gpm_api.dataset import (
    GPM_Dataset,
    GPM_variables,
)
from gpm_api.checks import (
    check_version,
    check_product,
    check_scan_mode,
)

# For create_GPM_Class
from gpm_api.DPR.DPR import create_DPR
from gpm_api.DPR.DPR_ENV import create_DPR_ENV
from gpm_api.PMW.GMI import create_GMI
from gpm_api.IMERG.IMERG import create_IMERG

################
### Classes ####
################
def create_GPM_class(base_DIR, product, bbox=None, start_time=None, end_time=None):
    # TODO add for ENV and SLH
    if product in ["1B-Ka", "1B-Ku", "2A-Ku", "2A-Ka", "2A-DPR", "2A-SLH"]:
        x = create_DPR(
            base_DIR=base_DIR,
            product=product,
            bbox=bbox,
            start_time=start_time,
            end_time=end_time,
        )
    elif product in GPM_IMERG_products():
        x = create_IMERG(
            base_DIR=base_DIR,
            product=product,
            bbox=bbox,
            start_time=start_time,
            end_time=end_time,
        )
    elif product in GPM_PMW_2A_GPROF_RS_products():
        x = create_GMI(
            base_DIR=base_DIR,
            product=product,
            bbox=bbox,
            start_time=start_time,
            end_time=end_time,
        )
    elif product in GPM_PMW_2A_PRPS_RS_products():
        x = create_GMI(
            base_DIR=base_DIR,
            product=product,
            bbox=bbox,
            start_time=start_time,
            end_time=end_time,
        )
    elif product in GPM_DPR_2A_ENV_RS_products():
        x = create_DPR_ENV(
            base_DIR=base_DIR,
            product=product,
            bbox=bbox,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        raise ValueError("Class method for such product not yet implemented")
    return x


def read_GPM(
    base_DIR,
    product,
    start_time,
    end_time,
    variables=None,
    scan_modes=None,
    GPM_version=6,
    product_type="RS",
    bbox=None,
    enable_dask=True,
    chunks="auto",
):
    """
    Construct a GPM object (DPR,GMI, IMERG) depending on the product specified.
    Map HDF5 data on disk lazily into dask arrays with relevant data and attributes.

    Parameters
    ----------
    base_DIR : str
       The base directory where GPM data are stored.
    product : str
        GPM product acronym.
    variables : str, list, optional
         Datasets names to extract from the HDF5 file.
         By default all variables available.
         Hint: GPM_variables(product) to see available variables.
    start_time : datetime
        Start time.
    end_time : datetime
        End time.
    scan_modes : str,list optional
        None --> Default to all the available for the specific GPM product
        'NS' = Normal Scan --> For Ku band and DPR
        'MS' = Matched Scans --> For Ka band and DPR
        'HS' = High-sensitivity Scans --> For Ka band and DPR
        For products '1B-Ku', '2A-Ku' and '2A-ENV-Ku', specify 'NS'.
        For products '1B-Ka', '2A-Ka' and '2A-ENV-Ka', specify 'MS' or 'HS'.
        For product '2A-DPR', specify 'NS', 'MS' or 'HS'.
        For product '2A-ENV-DPR', specify either 'NS' or 'HS'.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    GPM_version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
        GPM data readers are currently implemented only for GPM V06.
    bbox : list, optional
         Spatial bounding box. Format: [lon_0, lon_1, lat_0, lat_1]
    dask : bool, optional
         Wheter to lazy load data (in parallel) with dask. The default is True.
         Hint: xarray’s lazy loading of remote or on-disk datasets is often but not always desirable.
         Before performing computationally intense operations, load the Dataset
         entirely into memory by invoking the Dataset.load()
    chunks : str, list, optional
        Chunck size for dask. The default is 'auto'.
        Alternatively provide a list (with length equal to 'variables') specifying
        the chunk size option for each variable.

    Returns
    -------
    xarray.Dataset

    """
    from gpm_api.checks import initialize_scan_modes, check_version, check_product

    ##------------------------------------------------------------------------.
    ## Check GPM version
    check_version(GPM_version)
    ## Check product is valid
    check_product(product)
    # Initialize variables if not provided
    if variables is None:
        variables = GPM_variables(product=product, GPM_version=GPM_version)
    ## Initialize or check the scan_modes
    if scan_modes is not None:
        if isinstance(scan_modes, str):
            scan_modes = [scan_modes]
        scan_modes = [check_scan_mode(scan_mode, product) for scan_mode in scan_modes]
    else:
        scan_modes = initialize_scan_modes(product)
    ##------------------------------------------------------------------------.
    # Initialize GPM class
    x = create_GPM_class(
        base_DIR=base_DIR,
        product=product,
        bbox=bbox,
        start_time=start_time,
        end_time=end_time,
    )
    # Add the requested scan_mode to the GPM DPR class object
    for scan_mode in scan_modes:
        print("Retrieving", product, scan_mode, "data")
        x.__dict__[scan_mode] = GPM_Dataset(
            base_DIR=base_DIR,
            GPM_version=GPM_version,
            product=product,
            product_type=product_type,
            variables=variables,
            scan_mode=scan_mode,
            start_time=start_time,
            end_time=end_time,
            bbox=bbox,
            enable_dask=enable_dask,
            chunks=chunks,
        )
    ##------------------------------------------------------------------------.
    # Return the GPM class object with the requested  GPM data
    return x
