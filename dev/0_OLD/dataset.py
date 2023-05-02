#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:50:39 2020

@author: ghiggi
"""
import os

import dask.array
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from gpm_api.dataset.decoding import apply_custom_decoding
from gpm_api.io.checks import (
    check_bbox,
    check_product,
    check_scan_mode,
    check_version,
    is_empty,
    is_not_empty,
)
from gpm_api.io.find import find_GPM_files
from gpm_api.io.products import (
    GPM_DPR_2A_ENV_RS_products,
    GPM_DPR_RS_products,
    GPM_IMERG_products,
    GPM_PMW_2A_GPROF_RS_products,
    GPM_PMW_2A_PRPS_RS_products,
    GPM_products,
)
from gpm_api.utils.utils_HDF5 import hdf5_file_attrs


##----------------------------------------------------------------------------.
def subset_dict(x, keys):
    return dict((k, x[k]) for k in keys)


def remove_dict_keys(x, keys):
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        x.pop(key, None)
    return


def flip_boolean(x):
    # return list(~numpy.array(x))
    # return [not i for i in x]
    # return list(np.invert(x))
    return list(np.logical_not(x))


# ----------------------------------------------------------------------------.
#### Variables
# --> TO BE DEPRECATED
def GPM_variables_dict(product, scan_mode, version=6):
    """
    Return a dictionary with variables information for a specific GPM product.
    ----------
    product : str
        GPM product acronym.
    scan_mode : str
        Radar products have the following scan modes
        - 'FS' = Full Scan --> For Ku, Ka and DPR      (since version 7 products)
        - 'NS' = Normal Scan --> For Ku band and DPR   (till version 6  products)
        - 'MS' = Matched Scans --> For Ka band and DPR  (till version 6 for L2 products)
        - 'HS' = High-sensitivity Scans --> For Ka band and DPR
        - For products '1B-Ku', '2A-Ku' and '2A-ENV-Ku', specify 'FS'
        - For products '1B-Ka' specify either 'MS' or 'HS'.
        - For products '2A-Ka' and '2A-ENV-Ka' specify 'FS' or 'HS'.
        - For products '2A-DPR' and '2A-ENV-DPR' specify either 'FS' or 'HS'

        For product '2A-SLH', specify scan_mode = 'Swath'
        For product '2A-<PMW>', specify scan_mode = 'S1'
        For product '2B-GPM-CSH', specify scan_mode = 'Swath'.
        For product '2B-GPM-CORRA', specify either 'KuKaGMI' or 'KuGMI'.
        For product 'IMERG-ER','IMERG-LR' and 'IMERG-FR', specify scan_mode = 'Grid'.

        The above guidelines related to product version 7. For lower product version:
        - NS must be used instead of FS in Ku product.
        - MS is available in DPR L2 products till version 6.

    version : int, optional
        GPM version of the data to retrieve. Only GPM V06 currently implemented.

    Returns
    -------
    dict

    """
    if product not in GPM_products():
        raise ValueError("Retrievals not yet implemented for product", product)
    if version != 6:
        raise ValueError("Retrievals currently implemented only for GPM V06.")
    dict_path = os.path.dirname(os.path.abspath(__file__)) + "/CONFIG/"  # './CONFIG/'
    filename = "GPM_V" + str(version) + "_" + product + "_" + scan_mode
    filepath = dict_path + filename + ".yaml"
    with open(filepath) as file:
        d = yaml.safe_load(file)
    return d


def GPM_variables(product, scan_modes=None, version=6):
    """
    Return a list of variables available for a specific GPM product.

    Parameters
    ----------
    product : str
        GPM product acronym.
    version : int, optional
        GPM version of the data to retrieve. Only GPM V06 currently implemented.
    Returns
    -------
    list

    """
    from gpm_api.checks import initialize_scan_modes

    if scan_modes is None:
        # Retrieve scan modes
        scan_modes = initialize_scan_modes(product, version=version)
    if isinstance(scan_modes, str):
        scan_modes = [scan_modes]
    # If more than one, retrieve the union of the variables
    if len(scan_modes) > 1:
        l_vars = []
        for scan_mode in scan_modes:
            GPM_vars = list(
                GPM_variables_dict(product=product, scan_mode=scan_mode, version=version)
            )
            l_vars = l_vars + GPM_vars
        GPM_vars = list(np.unique(np.array(l_vars)))
    else:
        GPM_var_dict = GPM_variables_dict(product=product, scan_mode=scan_modes[0], version=version)
        GPM_vars = list(GPM_var_dict.keys())
    return GPM_vars


##----------------------------------------------------------------------------.
def check_variables(variables, product, scan_mode, version=6):
    """Checks the validity of variables."""
    # Make sure variable is a list (if str --> convert to list)
    if isinstance(variables, str):
        variables = [variables]
    # Check variables are valid
    valid_variables = GPM_variables(product=product, scan_modes=scan_mode, version=version)
    idx_valid = [var in valid_variables for var in variables]
    if not all(idx_valid):
        idx_not_valid = np.logical_not(idx_valid)
        if all(idx_not_valid):
            raise ValueError("All variables specified are not valid")
        else:
            variables = list(np.array(variables)[idx_valid])
            # variables_not_valid = list(np.array(variables)[idx_not_valid])
            # raise ValueError('The following variable are not valid:', variables_not_valid)
    ##------------------------------------------------------------------------.
    # Treat special cases for variables not available for specific products
    # This is now done using the YAML file !
    # 1B products
    # 2A products
    # if ('flagAnvil' in variables):
    #     if ((product == '2A-Ka') or (product == '2A-DPR' and scan_mode in ['MS','HS'])):
    #         # print('flagAnvil available only for Ku-band and DPR NS.\n Silent removal from the request done.')
    #         variables = str_remove(variables, 'flagAnvil')
    # if ('binDFRmMLBottom' in variables):
    #     if ((product in ['2A-Ka','2A-Ku']) or (product == '2A-DPR' and scan_mode == 'NS')):
    #         # print('binDFRmMLBottom available only for 2A-DPR.\n Silent removal from the request done.')
    #         variables = str_remove(variables, 'binDFRmMLBottom')
    # if ('binDFRmMLTop' in variables):
    #     if ((product in ['2A-Ka','2A-Ku']) or (product == '2A-DPR' and scan_mode == 'NS')):
    #         # print('binDFRmMLTop available only for 2A-DPR.\n Silent removal from the request done.')
    #         variables = str_remove(variables, 'binDFRmMLTop')
    # if (product == '2A-DPR' and scan_mode == 'MS'):
    #     variables = str_remove(variables, 'precipRate')
    #     variables = str_remove(variables, 'paramDSD')
    #     variables = str_remove(variables, 'phase')
    ##------------------------------------------------------------------------.
    # Check that there are still some variables to retrieve
    if len(variables) == 0:
        raise ValueError("No valid variables to retrieve")
    return variables


# -----------------------------------------------------------------------------.
#### HDF parsers
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


####--------------------------------------------------------------------------.
#####################
#### Coordinates ####
#####################
def get_orbit_coords(hdf, scan_mode):
    # Get Granule Number
    hdf_attr = hdf5_file_attrs(hdf)
    granule_id = hdf_attr["FileHeader"]["GranuleNumber"]

    lon = hdf[scan_mode]["Longitude"][:]
    lat = hdf[scan_mode]["Latitude"][:]
    time = _parse_hdf_gpm_scantime(hdf[scan_mode]["ScanTime"])
    n_along_track, n_cross_track = lon.shape
    granule_id = np.repeat(granule_id, n_along_track)
    scan_id = np.arange(n_along_track)
    gpm_id = [str(g) + "-" + str(z) for g, z in zip(granule_id, scan_id)]
    coords = {
        "lon": (["along_track", "cross_track"], lon),
        "lat": (["along_track", "cross_track"], lat),
        "time": (["along_track"], time),
        "granule_id": (["along_track"], granule_id),
        "scan_id": (["along_track"], scan_id),
        "gpm_id": (["along_track"], gpm_id),
    }
    return coords


def get_grid_coords(hdf, scan_mode):
    hdf_attr = hdf5_file_attrs(hdf)
    lon = hdf[scan_mode]["lon"][:]
    lat = hdf[scan_mode]["lat"][:]
    time = hdf_attr["FileHeader"]["StartGranuleDateTime"][:-1]
    time = np.array(np.datetime64(time) + np.timedelta64(30, "m"), ndmin=1)  # TODO: why plus 30
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
            {k: fileheader_attrs[k] for k in fileheader_attrs.keys() & set(fileheader_keys)}
        )

    # JAXAInfo attributes
    # - In DPR products
    jaxa_keys = ["TotalQualityCode"]
    jaxa_attrs = hdf_attr.get("JAXA_Info", None)
    if jaxa_attrs:
        attrs.update({k: jaxa_attrs[k] for k in jaxa_attrs.keys() & set(jaxa_keys)})
    return attrs


#######################
##### OLD VERSION #####
#######################
# # ----------------------------------------------------------------------------.
# def GPM_granule_Dataset(
#     hdf,
#     product,
#     variables,
#     scan_mode=None,
#     variables_dict=None,
#     version=6,
#     bbox=None,
#     enable_dask=True,
#     chunks="auto",
# ):
#     """
#     Create a lazy xarray.Dataset with relevant GPM data and attributes
#     for a specific granule.

#     Parameters
#     ----------
#     hdf : h5py.File
#         HFD5 object read with h5py.
#     scan_mode : str
#         'NS' = Normal Scan --> For Ku band and DPR
#         'MS' = Matched Scans --> For Ka band and DPR
#         'HS' = High-sensitivity Scans --> For Ka band and DPR
#         For products '1B-Ku', '2A-Ku' and '2A-ENV-Ku', specify 'NS'.
#         For products '1B-Ka', '2A-Ka' and '2A-ENV-Ka', specify either 'MS' or 'HS'.
#         For product '2A-DPR', specify either 'NS', 'MS' or 'HS'.
#         For product '2A-ENV-DPR', specify either 'NS' or 'HS'.
#         For product '2A-SLH', specify scan_mode = 'Swath'.
#         For product 'IMERG-ER','IMERG-LR' and 'IMERG-FR', specify scan_mode = 'Grid'.
#     product : str
#         GPM product acronym.
#     variables : list, str
#          Datasets names to extract from the HDF5 file.
#          Hint: utils_HDF5.hdf5_datasets_names() to see available datasets.
#     variables_dict : dict, optional
#          Expect dictionary from GPM_variables_dict(product, scan_mode)
#          Provided to avoid recomputing it at every call.
#          If variables_dict is None --> Perform also checks on the arguments.
#     version : int, optional
#         GPM version of the data to retrieve. Only GPM V06 currently implemented.
#     bbox : list, optional
#          Spatial bounding box. Format: [lon_0, lon_1, lat_0, lat_1]
#          For radar products it subset only along_track !
#     dask : bool, optional
#          Wheter to lazy load data (in parallel) with dask. The default is True.
#          Hint: xarray’s lazy loading of remote or on-disk datasets is often but not always desirable.
#          Before performing computationally intense operations, load the Dataset
#          entirely into memory by invoking the Dataset.load()
#     chunks : str, list, optional
#         Chunck size for dask. The default is 'auto'.
#         Alternatively provide a list (with length equal to 'variables') specifying
#         the chunk size option for each variable.

#     Returns
#     -------
#     xarray.Dataset

#     """
#     ##------------------------------------------------------------------------.
#     ## Arguments checks are usually done in GPM _Dataset()
#     if variables_dict is None:
#         ## Check valid product
#         check_product(product, product_type=None)
#         ## Check scan_mode
#         scan_mode = check_scan_mode(scan_mode=scan_mode, product=product, version=version)
#         ## Check variables
#         variables = check_variables(
#             variables=variables,
#             product=product,
#             scan_mode=scan_mode,
#             version=version,
#         )
#         ## Check bbox
#         bbox = check_bbox(bbox)
#         ## Retrieve variables dictionary
#         variables_dict = GPM_variables_dict(
#             product=product, scan_mode=scan_mode, version=version
#         )

#     ##------------------------------------------------------------------------.
#     ## Retrieve basic coordinates
#     # - For GPM radar orbit products array are stored ('along_track','cross_track')
#     if product not in GPM_IMERG_products():
#         coords = get_orbit_coords(hdf, scan_mode)
#     # - For IMERG products array are stored (lat, lon)
#     else:
#         coords = get_grid_coords(hdf, scan_mode)

#     ##------------------------------------------------------------------------.
#     ## Check if there is some data in the bounding box
#     if bbox is not None:
#         # - For GPM PMW and radar products (1A, 1B, 1C, 2A, 2B)
#         if product not in GPM_IMERG_products():
#             idx_row, idx_col = np.where(
#                 (lon >= bbox[0])
#                 & (lon <= bbox[1])
#                 & (lat >= bbox[2])
#                 & (lat <= bbox[3])
#             )
#         # - For IMERG products
#         else:
#             idx_row = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]
#             idx_col = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]
#         # If no data in the bounding box in current granule, return empty list
#         if idx_row.size == 0 or idx_col.size == 0:
#             return None

#     ##------------------------------------------------------------------------.
#     # Retrieve each variable
#     flag_first = True  # Required to decide if to create/append to Dataset
#     for var in variables:
#         # print(var)
#         ##--------------------------------------------------------------------.
#         # Prepare attributes for the DataArray
#         dict_attr = subset_dict(
#             variables_dict[var], ["description", "units", "standard_name"]
#         )
#         dict_attr["product"] = product
#         dict_attr["scan_mode"] = scan_mode
#         ##--------------------------------------------------------------------.
#         # Choose if using dask
#         if enable_dask is True:
#             hdf_obj = dask.array.from_array(
#                 hdf[scan_mode][variables_dict[var]["path"]], chunks=chunks
#             )
#         else:
#             hdf_obj = hdf[scan_mode][variables_dict[var]["path"]]
#         ##--------------------------------------------------------------------.
#         # Create the DataArray
#         da = xr.DataArray(
#             hdf_obj, dims=variables_dict[var]["dims"], coords=coords, attrs=dict_attr
#         )
#         da.name = var

#         ## -------------------------------------------------------------------.
#         #### Decode variables
#         ## Convert to float explicitly (needed?)
#         # hdf_obj.dtype  ## int16
#         # da = da.astype(np.float)
#         ## -------------------------------------------------------------------.
#         ## Parse missing values and errors
#         da = xr.where(da.isin(variables_dict[var]["_FillValue"]), np.nan, da)
#         # for value in dict_attr['_FillValue']:
#         #     da = xr.where(da == value, np.nan, da)

#         ## -------------------------------------------------------------------.
#         ## Add scale and offset
#         if len(variables_dict[var]["offset_scale"]) == 2:
#             da = (da / variables_dict[var]["offset_scale"][1] - variables_dict[var]["offset_scale"][0])

#         ## --------------------------------------------------------------------.
#         ## Create/Add to Dataset
#         if flag_first is True:
#             ds = da.to_dataset()
#             flag_first = False
#         else:
#             ds[var] = da
#         ##--------------------------------------------------------------------.

#     ##------------------------------------------------------------------------.
#     ## Subsetting based on bbox (lazy with dask)
#     # --> TODO: outside the loop
#     if bbox is not None:
#         # - For GPM radar products
#         # --> Subset only along_track to allow concat on cross_track
#         if product not in GPM_IMERG_products():
#             ds = ds.isel(along_track=slice((min(idx_row)), (max(idx_row) + 1)))
#         # - For IMERG products
#         else:
#             ds = ds.isel(lon=idx_row, lat=idx_col)

#     ##------------------------------------------------------------------------.
#     ## Special processing for specific fields
#     ds = apply_custom_decoding(ds, product)

#     ##------------------------------------------------------------------------.
#     # Add other stuffs to dataset
#     return ds


# def GPM_Dataset(
#     base_DIR,
#     product,
#     variables,
#     start_time,
#     end_time,
#     scan_mode=None,
#     version=6,
#     product_type="RS",
#     bbox=None,
#     enable_dask=False,
#     chunks="auto",
# ):
#     """
#     Lazily map HDF5 data into xarray.Dataset with relevant GPM data and attributes.

#     Parameters
#     ----------
#     base_DIR : str
#        The base directory where GPM data are stored.
#     product : str
#         GPM product acronym.
#     variables : list, str
#          Datasets names to extract from the HDF5 file.
#          Hint: GPM_variables(product) to see available variables.
#     start_time : datetime
#         Start time.
#     end_time : datetime
#         End time.
#     scan_mode : str, optional
#         'NS' = Normal Scan --> For Ku band and DPR
#         'MS' = Matched Scans --> For Ka band and DPR
#         'HS' = High-sensitivity Scans --> For Ka band and DPR
#         For products '1B-Ku', '2A-Ku' and '2A-ENV-Ku', specify 'NS'.
#         For products '1B-Ka', '2A-Ka' and '2A-ENV-Ka', specify either 'MS' or 'HS'.
#         For product '2A-DPR', specify either 'NS', 'MS' or 'HS'.
#         For product '2A-ENV-DPR', specify either 'NS' or 'HS'.
#     product_type : str, optional
#         GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
#     version : int, optional
#         GPM version of the data to retrieve if product_type = 'RS'.
#         GPM data readers are currently implemented only for GPM V06.
#     bbox : list, optional
#          Spatial bounding box. Format: [lon_0, lon_1, lat_0, lat_1]
#     dask : bool, optional
#          Wheter to lazy load data (in parallel) with dask. The default is True.
#          Hint: xarray’s lazy loading of remote or on-disk datasets is often but not always desirable.
#          Before performing computationally intense operations, load the Dataset
#          entirely into memory by invoking the Dataset.load()
#     chunks : str, list, optional
#         Chunck size for dask. The default is 'auto'.
#         Alternatively provide a list (with length equal to 'variables') specifying
#         the chunk size option for each variable.

#     Returns
#     -------
#     xarray.Dataset

#     """
#     ##------------------------------------------------------------------------.
#     ## Check valid product
#     check_product(product, product_type=product_type)
#     ## Check scan_mode
#     scan_mode = check_scan_mode(scan_mode, product, version=version)
#     ## Check variables
#     variables = check_variables(
#         variables=variables,
#         product=product,
#         scan_mode=scan_mode,
#         version=version,
#     )
#     ## Check bbox
#     bbox = check_bbox(bbox)
#     ##------------------------------------------------------------------------.
#     ## Check for chunks
#     # TODO smart_autochunk per variable (based on dim...)
#     # chunks = check_chunks(chunks)
#     ##------------------------------------------------------------------------.
#     # Find filepaths
#     filepaths = find_GPM_files(
#         base_dir=base_DIR,
#         version=version,
#         product=product,
#         product_type=product_type,
#         start_time=start_time,
#         end_time=end_time,
#     )
#     ##------------------------------------------------------------------------.
#     # Check that files have been downloaded  on disk
#     if is_empty(filepaths):
#         raise ValueError(
#             "Requested files are not found on disk. Please download them before"
#         )
#     ##------------------------------------------------------------------------.
#     # Initialize list (to store Dataset of each granule )
#     l_Datasets = []
#     # Retrieve variables dictionary
#     variables_dict = GPM_variables_dict(
#         product=product, scan_mode=scan_mode, version=version
#     )
#     for filepath in filepaths:
#         # Load hdf granule file
#         try:
#             hdf = h5py.File(filepath, "r")  # h5py._hl.files.File
#         except OSError:
#             if not os.path.exists(filepath):
#                 raise ValueError("This is a gpm_api bug. `find_GPM_files` should not have returned this filepath.")
#             else:
#                 print(f"The following file is corrupted and is being removed: {filepath}")
#                 print("Redownload the file !!!")
#                 os.remove(filepath)
#                 continue
#         hdf_attr = hdf5_file_attrs(hdf)
#         # --------------------------------------------------------------------.
#         ## Decide if retrieve data based on JAXA quality flags
#         # Do not retrieve data if TotalQualityCode not ...
#         if product in GPM_DPR_RS_products():
#             DataQualityFiltering = {
#                 "TotalQualityCode": ["Good"]
#             }  # TODO future fun args
#             if hdf_attr["JAXAInfo"]["TotalQualityCode"] not in DataQualityFiltering["TotalQualityCode"]:
#                 continue
#         # ---------------------------------------------------------------------.
#         # Retrieve data if granule is not empty
#         if hdf_attr["FileHeader"]["EmptyGranule"] == "NOT_EMPTY":
#             ds = GPM_granule_Dataset(
#                 hdf=hdf,
#                 version=version,
#                 product=product,
#                 scan_mode=scan_mode,
#                 variables=variables,
#                 variables_dict=variables_dict,
#                 bbox=bbox,
#                 enable_dask=enable_dask,
#                 chunks="auto",
#             )
#             if ds is not None:
#                 l_Datasets.append(ds)

#     ##-------------------------------------------------------------------------.
#     # Concat all Datasets
#     if len(l_Datasets) >= 1:
#         if product in GPM_IMERG_products():
#             ds = xr.concat(l_Datasets, dim="time")
#         else:
#             ds = xr.concat(l_Datasets, dim="along_track")
#         print(f"GPM {product} has been loaded successfully !")
#     else:
#         print("No data available for current request. Try for example to modify the bbox.")
#         return
#     ##------------------------------------------------------------------------.
#     # Return Dataset
#     return ds


# ##----------------------------------------------------------------------------.
