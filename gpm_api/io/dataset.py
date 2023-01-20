#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 14:35:55 2022

@author: ghiggi
"""
# ------------------------------------------------.
import os
import re
import h5py
import warnings
import numpy as np
import pandas as pd
import xarray as xr

from gpm_api.io.disk import find_filepaths
from gpm_api.io.checks import (
    check_variables,
    check_groups,
    check_scan_mode,
    check_product,
    check_base_dir,
    check_start_end_time,
)
from gpm_api.io.info import get_version_from_filepath, get_product_from_filepath
from gpm_api.io.decoding import apply_custom_decoding, decode_dataset
from gpm_api.utils.utils_HDF5 import hdf5_datasets, hdf5_groups, hdf5_file_attrs
from gpm_api.utils.time import (
    subset_by_time,
    ensure_time_validity,
)
from gpm_api.utils.checks import has_regular_time, is_regular, has_missing_granules
from gpm_api.utils.warnings import GPM_Warning
from gpm_api.io import VERSION # CURRENT GPM VERSION

####--------------------------------------------------------------------------.
### Define GPM_API Dataset Dimensions
DIM_DICT = {
    "nscan": "along_track",
    "nray": "cross_track",
    "npixel": "cross_track",
    "nrayMS": "cross_track",
    "nrayHS": "cross_track",
    "nrayNS": "cross_track",
    "nrayFS": "cross_track",
    "nbin": "range",
    "nbinMS": "range",
    "nbinHS": "range",
    "nbinFS": "range",
    "nfreq": "frequency",
    # PMW 1B-TMI
    "npixelev1": "cross_track",
    "npixelev2": "cross_track", 
    # PMW 1B-GMI (V7)
    "npix1": "cross_track",  # PMW (i.e. GMI)
    "npix2": "cross_track",  # PMW (i.e. GMI)
    "nfreq1": "frequency",
    "nfreq2": "frequency",
    "nchan1": "channel",
    "nchan2": "channel",
    # PMW 1C-GMI (V7)
    "npixel1": "cross_track",
    "npixel2": "cross_track",
    # PMW 1A-GMI and 1C-GMI (V7)
    "nscan1": "along_track",
    "nscan2": "along_track",
    "nchannel1": "channel",
    "nchannel2": "channel",
    # PMW 1A-GMI (V7)
    "npixelev": "cross_track",
    "npixelht": "cross_track",
    "npixelcs": "cross_track",
    "npixelfr": "cross_track",  # S4 mode
    # 2A-DPR, 2A-Ku, 2A-Ka
    "nDSD": "DSD_params",
    # nBnPSD --> "range" in CORRA --> 88, 250 M interval
    # "nfreqHI":
    # nlayer --> in CSH, SLH --> converted to height in decoding
}


def _decode_dimensions(ds):
    dataset_dims = list(ds.dims)
    dataset_vars = list(ds.data_vars)
    rename_dim_dict = {}
    for var in dataset_vars:
        da = ds[var]
        dim_names_str = da.attrs.get("DimensionNames", None)
        if dim_names_str is not None:
            dim_names = dim_names_str.split(",")
            for dim, new_dim in zip(list(da.dims), dim_names):
                if dim not in rename_dim_dict:
                    rename_dim_dict[dim] = new_dim
                else:
                    if rename_dim_dict[dim] == new_dim:
                        pass
                    else:  # when more variable share same dimension length (and same phony_dim_<n>)
                        ds[var] = ds[var].rename({dim: new_dim})
    if len(rename_dim_dict) > 0:
        ds = ds.rename_dims(rename_dim_dict)
    return ds


def assign_dataset_dimensions(ds):
    dataset_dims = list(ds.dims)

    # Do not assign dimension name if already exists (i.e. IMERG)
    if not re.match("phony_dim", dataset_dims[0]):
        return ds

    # Get dimension name from DimensionNames attribute
    ds = _decode_dimensions(ds)

    # Switch dimensions to gpm_api standard dimension names
    ds_dims = list(ds.dims)
    rename_dim_dict = {}
    for dim in ds_dims:
        new_dim = DIM_DICT.get(dim, None)
        if new_dim:
            rename_dim_dict[dim] = new_dim
    ds = ds.rename_dims(rename_dim_dict)
    return ds


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
    return ds.drop_dims(unused_var_dims(ds))


####--------------------------------------------------------------------------.
#####################
#### Coordinates ####
#####################
def _parse_hdf_gpm_scantime(h):
    df = pd.DataFrame(
        {
            "year": h["Year"][:],
            "month": h["Month"][:],
            "day": h["DayOfMonth"][:],
            "hour": h["Hour"][:],
            "minute": h["Minute"][:],
            "second": h["Second"][:],
        }
    )
    return pd.to_datetime(df).to_numpy()


def get_orbit_coords(hdf, scan_mode):
    hdf_attr = hdf5_file_attrs(hdf)
    granule_id = hdf_attr["FileHeader"]["GranuleNumber"]
    lon = hdf[scan_mode]["Longitude"][:]
    lat = hdf[scan_mode]["Latitude"][:]
    time = _parse_hdf_gpm_scantime(hdf[scan_mode]["ScanTime"])
    n_along_track, n_cross_track = lon.shape
    granule_id = np.repeat(granule_id, n_along_track)
    along_track_id = np.arange(n_along_track)
    cross_track_id = np.arange(n_cross_track)
    gpm_id = [str(g) + "-" + str(z) for g, z in zip(granule_id, along_track_id)]
    coords = {
        "lon": (["along_track", "cross_track"], lon),
        "lat": (["along_track", "cross_track"], lat),
        "time": (["along_track"], time),
        "gpm_id": (["along_track"], gpm_id),
        "gpm_granule_id": (["along_track"], granule_id),
        "gpm_cross_track_id": (["cross_track"], cross_track_id),
        "gpm_along_track_id": (["along_track"], along_track_id),
    }
    return coords


def get_grid_coords(hdf, scan_mode):
    hdf_attr = hdf5_file_attrs(hdf)
    lon = hdf[scan_mode]["lon"][:]
    lat = hdf[scan_mode]["lat"][:]
    time = hdf_attr["FileHeader"]["StartGranuleDateTime"][:-1]
    time = np.array(
        np.datetime64(time) + np.timedelta64(30, "m"), ndmin=1
    )  # TODO: document why + 30 min
    coords = {"time": time, "lon": lon, "lat": lat}
    return coords


def get_coords(hdf, scan_mode):
    if scan_mode == "Grid":
        coords = get_grid_coords(hdf, scan_mode)
    else:
        coords = get_orbit_coords(hdf, scan_mode)
    return coords


####--------------------------------------------------------------------------.
#####################
#### Attributes  ####
#####################
def get_attrs(hdf):
    attrs = {}
    hdf_attr = hdf5_file_attrs(hdf)
    # FileHeader attributes
    fileheader_keys = [
        "ProcessingSystem",
        "ProductVersion",
        "EmptyGranule",
        "DOI",
        "MissingData",
        "SatelliteName",
        "InstrumentName",
        "AlgorithmID",
    ]
    #
    fileheader_attrs = hdf_attr.get("FileHeader", None)
    if fileheader_attrs:
        attrs.update(
            {
                k: fileheader_attrs[k]
                for k in fileheader_attrs.keys() & set(fileheader_keys)
            }
        )

    # JAXAInfo attributes
    # - In DPR products
    jaxa_keys = ["TotalQualityCode"]
    jaxa_attrs = hdf_attr.get("JAXA_Info", None)
    if jaxa_attrs:
        attrs.update({k: jaxa_attrs[k] for k in jaxa_attrs.keys() & set(jaxa_keys)})
    return attrs


####--------------------------------------------------------------------------.
###################
#### HDF TOOLS ####
###################
def _get_hdf_groups(hdf, scan_mode, variables=None, groups=None):
    # Select only groups containing the specified variables
    if variables is not None:
        # Get dataset variables
        # - scan_mode
        dataset_dict = hdf5_datasets(hdf)
        var_group_dict = {}
        for k in dataset_dict.keys():
            if k.find("/") != -1:
                list_split = k.split("/")
                if len(list_split) == 3:  # scan_mode/group/var
                    var_group_dict[list_split[2]] = list_split[1]
                if len(list_split) == 2:  # scan_mode/var
                    var_group_dict[
                        list_split[1]
                    ] = ""  # to get variables from scan_mode group

        dataset_variables = list(var_group_dict)
        # Include 'height' variable if available (for radar)
        if "height" in dataset_variables:
            variables = np.unique(np.append(variables, ["height"]))
        # Check variables validity
        idx_subset = np.where(np.isin(variables, dataset_variables, invert=True))[0]
        if len(idx_subset) > 0:
            wrong_variables = variables[idx_subset]
            raise ValueError(
                f"The following variables are not available: {wrong_variables}."
            )
        # Get groups subset
        groups = np.unique([var_group_dict[var] for var in variables])
    elif groups is not None:
        # Check group validity
        dataset_groups = np.unique(list(hdf5_groups(hdf[scan_mode])) + [""])
        idx_subset = np.where(np.isin(groups, dataset_groups, invert=True))[0]
        if len(idx_subset) > 0:
            wrong_groups = groups[idx_subset]
            raise ValueError(f"The following groups are not available: {wrong_groups}.")

    # Select all groups
    else:
        groups = np.unique(list(hdf5_groups(hdf[scan_mode])) + [""])
    # ----------------------------------------------------.
    # Remove "ScanTime" from groups
    groups = np.setdiff1d(groups, ["ScanTime"])
    # ----------------------------------------------------.
    # Return groups
    return groups, variables


def _open_hdf_group(
    filepath,
    scan_mode,
    group,
    variables=None,
    prefix_group=True,
    decode_cf=False,
    chunks=None,
    engine="netcdf4",
):
    # Define hdf group
    # - group == '' to retrieve variables attached to scan_mode group
    hdf_group = scan_mode + "/" + group
    ds = xr.open_dataset(
        filepath, engine=engine, mode="r",
        group=hdf_group, decode_cf=decode_cf, chunks=chunks,
        # autoclose=True,
    )
    # Assign dimensions
    ds = assign_dataset_dimensions(ds)

    # Subset variables
    if variables is not None:
        variables_subset = variables[np.isin(variables, list(ds.data_vars))]
        ds = ds[variables_subset]
    
    # Remove unuseful variables
    dummy_variables = [
        "Latitude",
        "Longitude",
        # IMERG bnds
        "time_bnds",
        "lat_bnds",
        "lon_bnds",
    ]
    dummy_variables = np.array(dummy_variables)
    variables_to_drop = dummy_variables[np.isin(dummy_variables, list(ds.data_vars))]
    ds = ds.drop_vars(variables_to_drop)
    # Prefix variables with group name
    if prefix_group:
        if len(group) > 0:
            rename_var_dict = {var: group + "/" + var for var in ds.data_vars}
            ds = ds.rename_vars(rename_var_dict)
    return ds


def _get_granule_info(filepath, 
                      scan_mode,
                      variables,
                      groups):
    """Retrieve coordinates, attributes and valid variables and groups from the HDF file."""
    # Open HDF5 file
    hdf = h5py.File(filepath, "r")
    
    # Get coordinates
    coords = get_coords(hdf, scan_mode)
    
    # Get global attributes from the HDF file
    # TODO Add FileHeader Group attributes (see metadata doc)
    # TODO Select all possible attributes?
    
    # Global attributes:
    # - ProcessingSystem, DOI, InstrumentName,
    # - SatelliteName, AlgorithmID, ProductVersion
    attrs = get_attrs(hdf)
    attrs["ScanMode"] = scan_mode
    
    # Get groups to process (filtering out groups without any `variables`)
    groups, variables = _get_hdf_groups(
        hdf, scan_mode=scan_mode, variables=variables, groups=groups
    )
    # Close HDF5 file 
    hdf.close()
    
    return (coords, attrs, groups, variables)


def _get_granule_dataset(filepath,
                         scan_mode,
                         variables,
                         groups,
                         prefix_group,
                         chunks,
                         decode_cf=False):
    """Open a grouped HDF file into a xr.Dataset."""
    # Retrieve coords, attrs and valid group and variables from the HDF file
    coords, attrs, groups, variables = _get_granule_info(filepath=filepath,
                                                         scan_mode=scan_mode,
                                                         variables=variables, 
                                                         groups=groups)
    
    # Iterate over groups and create xr.Datasets
    list_ds = []
    for group in groups:
        ds = _open_hdf_group(
            filepath,
            scan_mode=scan_mode,
            group=group,
            variables=variables,
            prefix_group=prefix_group,
            decode_cf=False,
            chunks=chunks,
        )
        list_ds.append(ds)

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


####--------------------------------------------------------------------------.
##############################
#### gpm_api.open_granule ####
##############################
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
    ds = _get_granule_dataset(filepath=filepath,
                              scan_mode=scan_mode,
                              groups=groups,
                              variables=variables,
                              prefix_group=prefix_group,
                              chunks=chunks,
                              decode_cf=False)
                          
    ###-----------------------------------------------------------------------.
    ### Clean attritubes, decode variables
    # Apply custom processing
    ds = apply_custom_decoding(ds, product)

    # Apply CF decoding
    if decode_cf:
        ds = decode_dataset(ds)
    
    # Remove coords and dimensions not exploited by data variables
    ds = remove_unused_var_dims(ds)

    ###-----------------------------------------------------------------------.
    #### Check swath time coordinate
    # Ensure validity of the time dimension
    # - Infill up to 10 consecutive NaT
    # - Do not check for regular time dimension !
    # --> TODO: this can be moved into get_orbit_coords !
    ds = ensure_time_validity(ds, limit=10)
            
    # Warn  if non-contiguous scans are present in a GPM Orbit
    # - If ds is a GPM Grid Granule, is always a single timestep so always True
    if not is_regular(ds):
        msg = f"The GPM granule {filepath} has non-contiguous scans !"
        warnings.warn(msg, GPM_Warning)

    ###-----------------------------------------------------------------------.
    #### Check geolocation latitude/longitude coordinates
    # TODO: check_valid_geolocation
    # TODO: ensure_valid_geolocation (1 spurious pixel)
    # TODO: ds_gpm.gpm_api.valid_geolocation

    # Add QUALITY_FLAGS ATTRS  

    # Add global attributes
    # TODO: i.e. gpm_api_product for gpm_api.title accessor


    # TODO: tranpose data already here !!! 
    # cross-track - along_track
    # lat, lon 
    
    ###-----------------------------------------------------------------------.
    # Return xr.Dataset
    return ds


####---------------------------------------------------------------------------.
##############################
#### gpm_api.open_dataset ####
##############################


def _check_time_period_coverage(ds, start_time, end_time, raise_error=False): 
    """Check time period start_time, end_time is covered.
    
    If raise_error=True, raise error if time period is not covered.
    If raise_error=False, it raise a GPM warning.
    
    """
    # Get first and last timestep from xr.Dataset 
    first_start = ds['time'].data[0].astype("M8[s]").tolist()
    last_end = ds['time'].data[-1].astype("M8[s]").tolist()
    # Check time period is covered
    msg = ""
    if first_start > start_time:
        msg = f"The dataset start at {first_start}, although the specified start_time is {start_time}."
        
    if last_end < end_time:
        msg1 = f"The dataset end_time {last_end} occurs before the specified end_time {end_time}."
        if msg != "":
           msg = msg[:-1] + "; and t" + msg1[1:]
        else: 
            msg = msg1
    if msg != "": 
        if raise_error:
            raise ValueError(msg)
        else: 
            warnings.warn(msg, GPM_Warning)


def _is_valid_granule(filepath):
    """Chech the GPM HDF file is readable, not corrupted and not EMPTY."""
    # Try loading the HDF granule file
    try:
        hdf = h5py.File(filepath, "r")  # h5py._hl.files.File
        hdf_attr = hdf5_file_attrs(hdf)
        hdf.close()
    # If raise an OSError, warn and remove the file
    except OSError:
        if not os.path.exists(filepath):
            raise ValueError(
                "This is a gpm_api bug. `find_GPM_files` should not have returned this filepath."
            )
        else:
            msg = f"The following file is corrupted and is being removed: {filepath}. Redownload the file."
            warnings.warn(msg, GPM_Warning)
            os.remove(filepath)
            return False
    # If the GPM granule is empty, return False, otherwise True
    if hdf_attr["FileHeader"]["EmptyGranule"] == "NOT_EMPTY":
        return True
    else:
        return False


def _open_valid_granules(filepaths,
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
            ds = open_granule(
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
    return l_datasets          


def _concat_datasets(l_datasets):
    """Concatenate datasets together."""
    if len(l_datasets) >= 1:
        dims = list(l_datasets[0].dims)
        is_grid = "time" in dims
        if is_grid:
            concat_dim = "time"
        else: 
            concat_dim = "along_track"
        
        # Concatenate the datasets 
        ds = xr.concat(l_datasets, dim=concat_dim,
                       coords="minimal", # "all"
                       compat="override",
                       combine_attrs='override')
        
        # Tranpose to have (y,x) dimension order 
        # - This shape is expected by i.e. pyresample and matplotlib
        # - GRID:  (..., time, lat, lon)
        # - ORBIT: (cross_track, along_track, ...)
        if is_grid:          
            ds = ds.transpose(..., "lat", "lon")
        else:
            ds = ds.transpose("cross_track", "along_track", ...)
    else:
        raise ValueError("No valid GPM granule available for current request.")
    return ds 


def open_dataset(
    base_dir,
    product,
    start_time,
    end_time,
    variables=None,
    groups=None,  # TODO implement
    scan_mode=None,
    version=VERSION,
    product_type="RS",
    chunks="auto",
    decode_cf=True,
    prefix_group=True,
    verbose=False,
):
    """
    Lazily map HDF5 data into xarray.Dataset with relevant GPM data and attributes.

    Note:
    It does not load GPM granules with flag 'EmptyGranule' != "NOT_EMPTY"

    Parameters
    ----------
    base_dir : str
       The base directory where GPM data are stored.
    product : str
        GPM product acronym.
    variables : list, str
         Datasets names to extract from the HDF5 file.
         Hint: GPM_variables(product) to see available variables.
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
         THe default is True.
    Returns
    -------
    xarray.Dataset

    """
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
        raise ValueError(
            "Requested files are not found on disk. Please download them before."
        )

    # Check same version
    # - Filter by version make no sense because gpm_api version != filename version 
    # --> TODO: PUT IN DISK FIND_FILES 

    ##------------------------------------------------------------------------.
    # Initialize list (to store Dataset of each granule )
    l_datasets = _open_valid_granules(filepaths=filepaths,
                                      scan_mode=scan_mode,
                                      variables=variables,
                                      # groups=groups, # TODO
                                      decode_cf=False,
                                      prefix_group=prefix_group,
                                      chunks=chunks,
                                      )
    
    ##-------------------------------------------------------------------------.
    # TODO - Extract attributes and add as coordinate ?
    # - MissingData in FileHeaderGroup: The number of missing scans.
    # - TotalQualityCode in JAXAInfo Group
    # - NumberOfRainPixels (FS/HS) in JAXAInfo Group
    # - ProcessingSubSystem in JAXAInfo Group
    # - ProcessingMode in JAXAInfo Group
    # - DielectricFactorKa in JAXAInfo Group
    # - DielectricFactorKu in JAXAInfo Group
    # - ScanType: SwathHeader Group (”CROSSTRACK”, ”CONICAL”)
    
    ##-------------------------------------------------------------------------.
    # Concat all datasets
    ds = _concat_datasets(l_datasets)
    
    ##-------------------------------------------------------------------------.
    # Decode dataset
    if decode_cf:
        ds = decode_dataset(ds)

    # Add global attributes
    ds.attrs["gpm_api_product"] = product
    
    # Warns about missing granules 
    if has_missing_granules(ds):
        msg = "The GPM Dataset has missing granules !"
        warnings.warn(msg, GPM_Warning)
        
    ##------------------------------------------------------------------------.
    # Subset dataset for start_time and end_time
    # - Raise warning if the time period is not fully covered
    ds = subset_by_time(ds, start_time=start_time, end_time=end_time)
    _check_time_period_coverage(ds, start_time, end_time, raise_error=False)
    
    ##------------------------------------------------------------------------.
    # Return Dataset
    return ds


####--------------------------------------------------------------------------.
