#!/usr/bin/env python3
"""
Created on Tue Jul 18 17:30:43 2023

@author: ghiggi
"""
import os
import warnings

import xarray as xr

from gpm_api.configs import get_gpm_base_dir
from gpm_api.dataset.attrs import decode_string
from gpm_api.dataset.conventions import finalize_dataset
from gpm_api.dataset.granule import _open_granule
from gpm_api.io import GPM_VERSION  # CURRENT GPM VERSION
from gpm_api.io.checks import (
    check_base_dir,
    check_product,
    check_scan_mode,
    check_start_end_time,
    check_variables,
)
from gpm_api.io.disk import find_filepaths
from gpm_api.utils.checks import has_missing_granules
from gpm_api.utils.warnings import GPM_Warning


def _is_valid_granule(filepath):
    """Check the GPM HDF file is readable, not corrupted and not EMPTY."""
    # Try loading the HDF granule file
    try:
        with xr.open_dataset(filepath, engine="netcdf4", group="") as ds:
            attrs = ds.attrs
            attrs = decode_string(attrs["FileHeader"])
            is_empty_granule = attrs["EmptyGranule"] == "NOT_EMPTY"

    # netCDF4._netCDF4._ensure_nc_success

    # OSError: [Errno -51] NetCDF: Unknown file format
    # --> TODO: Try to download a file and interrupt download for test case
    # --> If raise an OSError, warn and remove the file

    # FileNotFoundError: [Errno 2] No such file or directory:
    except OSError as e:
        error_str = str(e)
        if not os.path.exists(filepath):
            raise ValueError(
                "This is a gpm_api bug. `find_filepaths` should not have returned this filepath."
            )
        elif "lock" in error_str:
            msg = "Unfortunately, HDF locking is occurring."
            msg += "Export the environment variable HDF5_USE_FILE_LOCKING = 'FALSE' into your environment (i.e. in the .bashrc).\n"  # noqa
            msg += f"The error is: '{error_str}'."
            raise ValueError(msg)
        else:
            msg = f"The following file is corrupted and is being removed: {filepath}. Redownload the file."
            warnings.warn(msg, GPM_Warning)
            # os.remove(filepath) # TODO !!!
            return False

    # If the GPM granule is empty, return False, otherwise True
    return is_empty_granule


def _open_valid_granules(
    filepaths,
    scan_mode,
    variables,
    groups,
    decode_cf,
    prefix_group,
    chunks,
):
    """
    Open a list of HDF granules.

    Corrupted granules are not returned !

    Returns
    -------
    l_Datasets : list
        List of xr.Datasets.

    """
    l_datasets = []
    for filepath in filepaths:
        # Retrieve data if granule is not empty
        if _is_valid_granule(filepath):
            ds = _open_granule(
                filepath,
                scan_mode=scan_mode,
                variables=variables,
                groups=groups,
                decode_cf=False,
                prefix_group=prefix_group,
                chunks=chunks,
            )
            if ds is not None:
                l_datasets.append(ds)
    if len(l_datasets) == 0:
        raise ValueError(
            "No valid GPM granule available for current request. All granules are EMPTY."
        )
    return l_datasets


def _concat_datasets(l_datasets):
    """Concatenate datasets together."""
    dims = list(l_datasets[0].dims)
    is_grid = "time" in dims
    concat_dim = "time" if is_grid else "along_track"

    # Concatenate the datasets
    ds = xr.concat(
        l_datasets,
        dim=concat_dim,
        coords="minimal",  # "all"
        compat="override",
        combine_attrs="override",
    )
    return ds


def open_dataset(
    product,
    start_time,
    end_time,
    variables=None,
    groups=None,  # TODO implement
    scan_mode=None,
    version=GPM_VERSION,
    product_type="RS",
    chunks={},
    decode_cf=True,
    prefix_group=False,
    verbose=False,
    base_dir=None,
):
    """
    Lazily map HDF5 data into xarray.Dataset with relevant GPM data and attributes.

    Note:
    - gpm_api.open_dataset does not load GPM granules with the FileHeader flag 'EmptyGranule' != 'NOT_EMPTY'
    - The group "ScanStatus" provides relevant data flags for Swath products.
    - The variable "dataQuality" provides an overall quality flag status.
      If dataQuality = 0, no issues have been detected.
    - The variable "SCorientation" provides the orientation of the sensor
      from the forward track of the satellite.


    Parameters
    ----------
    product : str
        GPM product acronym.
    variables : list, str
         Datasets names to extract from the HDF5 file.
    groups : list, str
         Groups from which to extract all variables.
         The default is None.
    start_time : datetime.datetime
        Start time.
    end_time : datetime.datetime
        End time.
    scan_mode : str, optional
        'NS' = Normal Scan --> For Ku band and DPR
        'MS' = Matched Scans --> For Ka band and DPR
        'HS' = High-sensitivity Scans --> For Ka band and DPR
        For products '1B-Ku', '2A-Ku' and '2A-ENV-Ku', specify 'NS'.
        For products '1B-Ka', '2A-Ka' and '2A-ENV-Ka', specify either 'MS' or 'HS'.
        For product '2A-DPR', specify either 'NS', 'MS' or 'HS'.
        For product '2A-ENV-DPR', specify either 'NS' or 'HS'.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
        GPM data readers currently support version 4, 5, 6 and 7.
    chunks : str, list, optional
        Chunk size for dask. The default is {}.
        Alternatively provide a list (with length equal to 'variables') specifying
        the chunk size option for each variable.
    decode_cf: bool, optional
        Whether to decode the dataset. The default is False.
    prefix_group: bool, optional
        Whether to add the group as a prefix to the variable names.
        If you aim to save the Dataset to disk as netCDF or Zarr, you need to set prefix_group=False
        or later remove the prefix before writing the dataset.
        The default is False.
    base_dir : str, optional
        The path to the GPM base directory. If None, it use the one specified
        in the GPM-API config file.
        The default is None.

    Returns
    -------
    xarray.Dataset

    """
    # -------------------------------------------------------------------------.
    # Retrieve GPM-API configs
    base_dir = get_gpm_base_dir(base_dir)

    ##------------------------------------------------------------------------.
    # Check base_dir
    base_dir = check_base_dir(base_dir)
    ## Check scan_mode
    scan_mode = check_scan_mode(scan_mode, product, version=version)
    ## Check valid product and variables
    check_product(product, product_type=product_type)
    variables = check_variables(variables)
    # Check valid start/end time
    start_time, end_time = check_start_end_time(start_time, end_time)

    ##------------------------------------------------------------------------.
    ## TODO: Check for chunks
    # - check works in open_dataset
    # - smart_autochunk per variable (based on dim...)
    # chunks = check_chunks(chunks)

    ##------------------------------------------------------------------------.
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

    ##------------------------------------------------------------------------.
    # Check that files have been downloaded on disk
    if len(filepaths) == 0:
        raise ValueError("No files found on disk. Please download them before.")

    ##------------------------------------------------------------------------.
    # Initialize list (to store Dataset of each granule )
    l_datasets = _open_valid_granules(
        filepaths=filepaths,
        scan_mode=scan_mode,
        variables=variables,
        groups=groups,
        decode_cf=False,
        prefix_group=prefix_group,
        chunks=chunks,
    )

    ##-------------------------------------------------------------------------.
    # TODO - Extract attributes and add as coordinate ?
    # - From each granule, select relevant (discard/sum values/copy)
    # - Sum of MissingData, NumberOfRainPixels
    # - MissingData in FileHeaderGroup: The number of missing scans.

    ##-------------------------------------------------------------------------.
    # Concat all datasets
    ds = _concat_datasets(l_datasets)

    ##-------------------------------------------------------------------------.
    # Finalize dataset
    ds = finalize_dataset(
        ds=ds, product=product, decode_cf=decode_cf, start_time=start_time, end_time=end_time
    )

    ##------------------------------------------------------------------------.
    # Warns about missing granules
    if has_missing_granules(ds):
        msg = "The GPM Dataset has missing granules !"
        warnings.warn(msg, GPM_Warning)

    ##------------------------------------------------------------------------.
    # Return Dataset
    return ds


####--------------------------------------------------------------------------.
