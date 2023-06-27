#!/usr/bin/env python3
"""
Created on Mon Aug 15 14:35:55 2022

@author: ghiggi
"""
import os
import warnings

import h5py
import numpy as np
import pyproj
import xarray as xr

from gpm_api.configs import get_gpm_base_dir
from gpm_api.dataset.attrs import add_history, get_granule_attrs
from gpm_api.dataset.coords import get_coords, set_coords_attrs
from gpm_api.dataset.crs import set_dataset_crs
from gpm_api.dataset.decoding import apply_custom_decoding, decode_dataset
from gpm_api.dataset.group import _get_hdf_groups, _open_hdf_group
from gpm_api.io import GPM_VERSION  # CURRENT GPM VERSION
from gpm_api.io.checks import (
    check_base_dir,
    check_groups,
    check_product,
    check_scan_mode,
    check_start_end_time,
    check_variables,
)
from gpm_api.io.disk import find_filepaths
from gpm_api.io.info import get_product_from_filepath, get_version_from_filepath
from gpm_api.utils.checks import has_missing_granules, is_regular
from gpm_api.utils.time import (
    ensure_time_validity,
    subset_by_time,
)
from gpm_api.utils.utils_HDF5 import hdf5_file_attrs
from gpm_api.utils.warnings import GPM_Warning

EPOCH = "seconds since 1970-01-01 00:00:00"

SWMR = False  # HDF options


####--------------------------------------------------------------------------.
#################################
#### GPM Dataset Conventions ####
#################################
def is_orbit(xr_obj):
    """Check whether the GPM xarray object is an orbit."""
    return "along_track" in list(xr_obj.dims)


def is_grid(xr_obj):
    """Check whether the GPM xarray object is a grid."""
    return bool("longitude" in list(xr_obj.dims) or "lon" in list(xr_obj.dims))


def reshape_dataset(ds):
    """Define the dataset dimension order.

    It ensure that the output dimension order is  (y, x)
    This shape is expected by i.e. pyresample and matplotlib
    For GPM GRID objects:  (..., time, lat, lon)
    For GPM ORBIT objects: (cross_track, along_track, ...)
    """
    if is_grid(ds):
        ds = ds.transpose(..., "lat", "lon")
    else:
        if "cross_track" in ds.dims:
            ds = ds.transpose("cross_track", "along_track", ...)
        else:
            ds = ds.transpose("along_track", ...)
    return ds


def finalize_dataset(ds, product, decode_cf, start_time=None, end_time=None):
    """Finalize GPM dataset."""
    ##------------------------------------------------------------------------.
    # Tranpose to have (y, x) dimension order
    ds = reshape_dataset(ds)

    ##------------------------------------------------------------------------.
    # Decode dataset
    if decode_cf:
        ds = decode_dataset(ds)

    ##------------------------------------------------------------------------.
    # Add coordinates attributes
    ds = set_coords_attrs(ds)

    ##------------------------------------------------------------------------.
    # Add CRS information
    # - See Geolocation toolkit ATBD at
    #   https://gpm.nasa.gov/sites/default/files/document_files/GPMGeolocationToolkitATBDv2.1-2012-07-31.pdf
    # TODO: set_dataset_crs should be migrated to cf_xarray ideally
    crs = pyproj.CRS(proj="longlat", ellps="WGS84")
    ds = set_dataset_crs(ds, crs=crs, grid_mapping_name="crsWGS84", inplace=False)

    ##------------------------------------------------------------------------.
    # Add time encoding
    encoding = {}
    encoding["units"] = EPOCH
    encoding["calendar"] = "proleptic_gregorian"
    ds["time"].encoding = encoding

    ##------------------------------------------------------------------------.
    # Add GPM-API global attributes
    ds = add_history(ds)
    ds.attrs["gpm_api_product"] = product

    ##------------------------------------------------------------------------.
    # Subset dataset for start_time and end_time
    # - Raise warning if the time period is not fully covered
    # - The warning can raise if some data are not downloaded or some granule
    #   at the start/end of the period are empty
    ds = subset_by_time(ds, start_time=start_time, end_time=end_time)
    _check_time_period_coverage(ds, start_time=start_time, end_time=end_time, raise_error=False)

    return ds


####--------------------------------------------------------------------------.
##############################
#### gpm_api.open_granule ####
##############################


def get_variables_dims(ds):
    """Retrieve the dimensions used by the xr.Dataset variables."""
    dims = np.unique(np.concatenate([list(ds[var].dims) for var in ds.data_vars]))
    return dims


def unused_var_dims(ds):
    """Retrieve the dimensions not used by the the xr.Dataset variables."""
    var_dims = set(get_variables_dims(ds))
    ds_dims = set(list(ds.dims))
    unused_dims = ds_dims.difference(var_dims)
    return list(unused_dims)


def remove_unused_var_dims(ds):
    """Remove coordinates and dimensions not used by the xr.Dataset variables."""
    unused_dims = unused_var_dims(ds)
    return ds.drop_dims(unused_dims)


def _get_granule_info(filepath, scan_mode, variables, groups):
    """Retrieve coordinates, attributes and valid variables and groups from the HDF file."""
    # Open HDF5 file
    with h5py.File(filepath, "r", locking=False, swmr=SWMR) as hdf:

        # Get coordinates
        coords = get_coords(hdf, scan_mode)

        # Get global attributes from the HDF file
        attrs = get_granule_attrs(hdf)
        attrs["ScanMode"] = scan_mode

        # Get groups to process (filtering out groups without any `variables`)
        groups, variables = _get_hdf_groups(
            hdf, scan_mode=scan_mode, variables=variables, groups=groups
        )

    return (coords, attrs, groups, variables)


def _get_granule_dataset(
    filepath, scan_mode, variables, groups, prefix_group, chunks, decode_cf=False
):
    """Open a grouped HDF file into a xr.Dataset."""
    # Retrieve coords, attrs and valid group and variables from the HDF file
    coords, attrs, groups, variables = _get_granule_info(
        filepath=filepath, scan_mode=scan_mode, variables=variables, groups=groups
    )
    # Iterate over groups and create xr.Datasets
    list_ds = [
        _open_hdf_group(
            filepath,
            scan_mode=scan_mode,
            group=group,
            variables=variables,
            prefix_group=prefix_group,
            decode_cf=False,
            chunks=chunks,
        )
        for group in groups
    ]
    # Create single xr.Dataset
    ds = xr.merge(list_ds)
    # Assign coords
    ds = ds.assign_coords(coords)
    # Assign global attributes
    ds.attrs = attrs
    # Remove list_ds
    del list_ds
    # Return dataset
    return ds


def _open_granule(
    filepath,
    scan_mode=None,
    groups=None,
    variables=None,
    decode_cf=False,
    chunks="auto",
    prefix_group=True,
):
    """Open granule file into xarray Dataset."""
    # Get product
    product = get_product_from_filepath(filepath)
    version = get_version_from_filepath(filepath)

    # Check variables and groups
    variables = check_variables(variables)
    groups = check_groups(groups)

    # Check scan_mode
    scan_mode = check_scan_mode(scan_mode, product, version)

    ###-----------------------------------------------------------------------.
    # Retrieve the granule dataset (without cf decoding)
    ds = _get_granule_dataset(
        filepath=filepath,
        scan_mode=scan_mode,
        groups=groups,
        variables=variables,
        prefix_group=prefix_group,
        chunks=chunks,
        decode_cf=False,
    )

    ###-----------------------------------------------------------------------.
    ### Clean attributes, decode variables
    # Apply custom processing
    ds = apply_custom_decoding(ds, product)

    # Apply CF decoding
    if decode_cf:
        ds = decode_dataset(ds)

    # Remove coords and dimensions not exploited by data variables
    ds = remove_unused_var_dims(ds)

    ###-----------------------------------------------------------------------.
    ## Check swath time coordinate
    # Ensure validity of the time dimension
    # - Infill up to 10 consecutive NaT
    # - Do not check for regular time dimension !
    # --> TODO: this can be moved into get_orbit_coords !
    ds = ensure_time_validity(ds, limit=10)

    # Try to warn if non-contiguous scans are present in a GPM Orbit
    # - If any of the  GPM Orbit specified variables has the cross-track dimension, the check raise an error
    # - If ds is a GPM Grid Granule, is always a single timestep so always True
    try:
        if not is_regular(ds):
            msg = f"The GPM granule {filepath} has non-contiguous scans !"
            warnings.warn(msg, GPM_Warning)
    except Exception:
        pass

    ###-----------------------------------------------------------------------.
    ## Check geolocation latitude/longitude coordinates
    # TODO: check_valid_geolocation
    # TODO: ensure_valid_geolocation (1 spurious pixel)
    # TODO: ds_gpm.gpm_api.valid_geolocation

    ###-----------------------------------------------------------------------.
    # Return xr.Dataset
    return ds


def open_granule(
    filepath,
    scan_mode=None,
    groups=None,
    variables=None,
    decode_cf=False,
    chunks="auto",
    prefix_group=True,
):
    """
    Create a lazy xarray.Dataset with relevant GPM data and attributes
    for a specific granule.

    Parameters
    ----------
    fpath : str
        Filepath of GPM granule dataset
    scan_mode : str
        The radar products have the following scan modes
        - 'FS' = Full Scan --> For Ku, Ka and DPR      (since version 7 products)
        - 'NS' = Normal Scan --> For Ku band and DPR   (till version 6  products)
        - 'MS' = Matched Scans --> For Ka band and DPR  (till version 6 for L2 products)
        - 'HS' = High-sensitivity Scans --> For Ka band and DPR
        For version 7:
        - For products '1B-Ku', '2A-Ku' and '2A-ENV-Ku', specify 'FS'
        - For products '1B-Ka' specify either 'MS' or 'HS'.
        - For products '2A-Ka' and '2A-ENV-Ka' specify 'FS' or 'HS'.
        - For products '2A-DPR' and '2A-ENV-DPR' specify either 'FS' or 'HS'
        For version < 7:
        - NS must be used instead of FS in Ku product.
        - MS is available in DPR L2 products till version 6.

        For product '2A-SLH', specify scan_mode = 'Swath'
        For product '2A-<PMW>', specify scan_mode = 'S1'
        For product '2B-GPM-CSH', specify scan_mode = 'Swath'.
        For product '2B-GPM-CORRA', specify either 'KuKaGMI' or 'KuGMI'.
        For product 'IMERG-ER','IMERG-LR' and 'IMERG-FR', specify scan_mode = 'Grid'.

        The above guidelines related to product version 7.

    variables : list, str
         Datasets names to extract from the HDF5 file.
         Hint: utils_HDF5.hdf5_datasets_names() to see available datasets.
    groups
        TODO
    chunks : str, list, optional
        Chunk size for dask array. The default is 'auto'.
        If you want to load data in memory directly, specify chunks=None.

        Hint: xarray’s lazy loading of remote or on-disk datasets is often but not always desirable.
        Before performing computationally intense operations, load the Dataset
        entirely into memory by invoking ds.compute().

        Custom chunks can be specified by: TODO
        - Provide a list (with length equal to 'variables') specifying
          the chunk size option for each variable.
    decode_cf: bool, optional
        Whether to decode the dataset. The default is False.
    prefix_group: bool, optional
        Whether to add the group as a prefix to the variable names.
        THe default is True.

    Returns
    -------

    ds:  xarray.Dataset

    """
    # Open granule
    ds = _open_granule(
        filepath=filepath,
        scan_mode=scan_mode,
        groups=groups,
        variables=variables,
        decode_cf=decode_cf,
        chunks=chunks,
        prefix_group=prefix_group,
    )

    # Finalize granule
    product = get_product_from_filepath(filepath)
    ds = finalize_dataset(
        ds=ds, product=product, decode_cf=decode_cf, start_time=None, end_time=None
    )
    return ds


####---------------------------------------------------------------------------.
##############################
#### gpm_api.open_dataset ####
##############################


def _check_time_period_coverage(ds, start_time=None, end_time=None, raise_error=False):
    """Check time period start_time, end_time is covered.

    If raise_error=True, raise error if time period is not covered.
    If raise_error=False, it raise a GPM warning.

    """
    # Get first and last timestep from xr.Dataset
    first_start = ds["time"].data[0].astype("M8[s]").tolist()
    last_end = ds["time"].data[-1].astype("M8[s]").tolist()
    # Check time period is covered
    msg = ""
    if start_time and first_start > start_time:
        msg = f"The dataset start at {first_start}, although the specified start_time is {start_time}."
    if end_time and last_end < end_time:
        msg1 = f"The dataset end_time {last_end} occurs before the specified end_time {end_time}."
        msg = msg[:-1] + "; and t" + msg1[1:] if msg != "" else msg1
    if msg != "":
        if raise_error:
            raise ValueError(msg)
        else:
            warnings.warn(msg, GPM_Warning)


def _is_valid_granule(filepath):
    """Chech the GPM HDF file is readable, not corrupted and not EMPTY."""
    # Try loading the HDF granule file
    try:
        with h5py.File(filepath, "r", locking=False, swmr=SWMR) as hdf:
            hdf_attr = hdf5_file_attrs(hdf)
    # If raise an OSError, warn and remove the file
    # - OSError: Unable to open file (file locking flag values don't match)
    # - OSError: Unable to open file (file signature not found)
    # - OSError: Unable to open file (truncated file: eof = ...)
    except OSError as e:
        error_str = str(e)
        if not os.path.exists(filepath):
            raise ValueError(
                "This is a gpm_api bug. `find_GPM_files` should not have returned this filepath."
            )
        elif "lock" in error_str:
            msg = "Unfortunately, HDF locking is occuring."
            msg += "Export the environment variable HDF5_USE_FILE_LOCKING = 'FALSE' into your environment (i.e. in the .bashrc).\n"  # noqa
            msg += f"The error is: '{error_str}'."
            raise ValueError(msg)
        else:
            msg = f"The following file is corrupted and is being removed: {filepath}. Redownload the file."
            warnings.warn(msg, GPM_Warning)
            # os.remove(filepath) # TODO !!!
            return False
    # If the GPM granule is empty, return False, otherwise True
    return hdf_attr["FileHeader"]["EmptyGranule"] == "NOT_EMPTY"


def _open_valid_granules(
    filepaths,
    scan_mode,
    variables,
    # groups,
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
                # groups=groups, # TODO
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
    chunks="auto",
    decode_cf=True,
    prefix_group=True,
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
        Chunck size for dask. The default is 'auto'.
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

    # Check same version
    # - Filter by version make no sense because gpm_api version != filename version
    # --> TODO: PUT IN DISK FIND_FILES

    ##------------------------------------------------------------------------.
    # Initialize list (to store Dataset of each granule )
    l_datasets = _open_valid_granules(
        filepaths=filepaths,
        scan_mode=scan_mode,
        variables=variables,
        # groups=groups, # TODO
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
