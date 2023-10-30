#!/usr/bin/env python3
"""
Created on Tue Jul 18 17:30:43 2023

@author: ghiggi
"""
import warnings
from functools import partial

import xarray as xr

from gpm_api.dataset.conventions import finalize_dataset
from gpm_api.dataset.granule import _open_granule
from gpm_api.io.checks import (
    check_groups,
    check_product,
    check_scan_mode,
    check_start_end_time,
    check_valid_time_request,
    check_variables,
)
from gpm_api.io.find import find_filepaths
from gpm_api.utils.checks import has_missing_granules
from gpm_api.utils.warnings import GPM_Warning


def _get_scheduler(get=None, collection=None):
    """Determine the dask scheduler that is being used.

    None is returned if no dask scheduler is active.

    See Also
    --------
    dask.base.get_scheduler
    """
    try:
        import dask
        from dask.base import get_scheduler  # noqa: F401

        actual_get = get_scheduler(get, collection)
    except ImportError:
        return None

    try:
        from dask.distributed import Client

        if isinstance(actual_get.__self__, Client):
            return "distributed"
    except (ImportError, AttributeError):
        pass

    try:
        if actual_get is dask.multiprocessing.get:
            return "multiprocessing"
    except AttributeError:
        pass

    return "threaded"


def _try_open_granule(filepath, scan_mode, variables, groups, prefix_group, decode_cf, chunks):
    """Try open a granule."""
    try:
        ds = _open_granule(
            filepath,
            scan_mode=scan_mode,
            variables=variables,
            groups=groups,
            decode_cf=decode_cf,
            prefix_group=prefix_group,
            chunks=chunks,
        )
    except Exception as e:
        msg = f"The following error occurred while opening the {filepath} granule: {e}"
        warnings.warn(msg, GPM_Warning)
        ds = None
    return ds


def _get_datasets_and_closers(filepaths, parallel, **open_kwargs):
    """Open the granule in parallel with dask delayed."""
    if parallel:
        import dask

        # wrap the _try_open_granule and getattr with delayed
        open_ = dask.delayed(_try_open_granule)
        getattr_ = dask.delayed(getattr)
    else:
        open_ = _try_open_granule
        getattr_ = getattr

    list_ds = [open_(p, **open_kwargs) for p in filepaths]
    list_closers = [getattr_(ds, "_close", None) for ds in list_ds]

    # If parallel=True, compute the delayed datasets lists here
    # - The underlying data are stored as dask arrays (and are not computed !)
    if parallel:
        list_ds, list_closers = dask.compute(list_ds, list_closers)

    # Remove None elements from the list
    list_ds = [ds for ds in list_ds if ds is not None]
    list_closers = [closer for closer in list_closers if closer is not None]
    return list_ds, list_closers


def _multi_file_closer(closers):
    """Close connection of multiple files."""
    for closer in closers:
        closer()


def _open_valid_granules(
    filepaths,
    scan_mode,
    variables,
    groups,
    decode_cf,
    prefix_group,
    chunks,
    parallel=False,
):
    """
    Open a list of HDF granules.

    Corrupted granules are not returned !

    Returns
    -------
    list_datasets : list
        List of xr.Datasets.
    list_closers : list
         List of xr.Datasets closers.
    """
    if parallel and chunks is None:
        return ValueError("If parallel=True, 'chunks' can not be None.")
    list_ds, list_closers = _get_datasets_and_closers(
        filepaths,
        scan_mode=scan_mode,
        variables=variables,
        groups=groups,
        decode_cf=False,
        prefix_group=prefix_group,
        chunks=chunks,
        parallel=parallel,
    )

    if len(list_ds) == 0:
        raise ValueError("No valid GPM granule available for current request.")
    return list_ds, list_closers


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
    version=None,
    product_type="RS",
    chunks={},
    decode_cf=True,
    parallel=False,
    prefix_group=False,
    verbose=False,
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
    parallel : bool, default: False
        If True, the dataset opening is performed in parallel using ``dask.delayed``.
        If parallel=True, 'chunks' can not be None. The underlying data must be dask.Array.
        The default is False.
    prefix_group: bool, optional
        Whether to add the group as a prefix to the variable names.
        If you aim to save the Dataset to disk as netCDF or Zarr, you need to set prefix_group=False
        or later remove the prefix before writing the dataset.
        The default is False.

    Returns
    -------
    xarray.Dataset

    """
    ## Check scan_mode
    scan_mode = check_scan_mode(scan_mode, product, version=version)
    ## Check valid product and variables
    check_product(product, product_type=product_type)
    variables = check_variables(variables)
    groups = check_groups(groups)
    # Check valid start/end time
    start_time, end_time = check_start_end_time(start_time, end_time)
    check_valid_time_request(start_time, end_time, product)

    ##------------------------------------------------------------------------.
    # Find filepaths
    filepaths = find_filepaths(
        storage="local",
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
    list_ds, list_closers = _open_valid_granules(
        filepaths=filepaths,
        scan_mode=scan_mode,
        variables=variables,
        groups=groups,
        decode_cf=False,
        prefix_group=prefix_group,
        parallel=parallel,
        chunks=chunks,
    )

    ##-------------------------------------------------------------------------.
    # TODO - Extract attributes and add as coordinate ?
    # - From each granule, select relevant (discard/sum values/copy)
    # - Sum of MissingData, NumberOfRainPixels
    # - MissingData in FileHeaderGroup: The number of missing scans.

    ##-------------------------------------------------------------------------.
    # Concat all datasets
    # - If concatenation fails, close connection to disk !
    try:
        ds = _concat_datasets(list_ds)
    except ValueError:
        for ds in list_ds:
            ds.close()
        raise

    ##-------------------------------------------------------------------------.
    # Set dataset closers to execute when ds is closed
    ds.set_close(partial(_multi_file_closer, list_closers))

    ##-------------------------------------------------------------------------.
    # Finalize dataset
    ds = finalize_dataset(
        ds=ds,
        product=product,
        scan_mode=scan_mode,
        decode_cf=decode_cf,
        start_time=start_time,
        end_time=end_time,
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
