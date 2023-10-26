#!/usr/bin/env python3
"""
Created on Thu Oct 13 17:45:37 2022

@author: ghiggi
"""
import datetime
import os
import subprocess

from dateutil.relativedelta import relativedelta

from gpm_api.configs import get_gpm_password, get_gpm_username
from gpm_api.io.checks import (
    check_product_type,
    check_product_validity,
    check_product_version,
)
from gpm_api.io.products import available_products, get_product_info

####--------------------------------------------------------------------------.
#####################
#### Directories ####
#####################


def _get_pps_servers(product_type):
    """Return the url to the PPS servers."""
    if product_type == "NRT":
        url_text_server = "https://jsimpsonhttps.pps.eosdis.nasa.gov/text"
        url_data_server = "ftps://jsimpsonftps.pps.eosdis.nasa.gov/data"
    else:
        url_text_server = "https://arthurhouhttps.pps.eosdis.nasa.gov/text"
        url_data_server = "ftps://arthurhouftps.pps.eosdis.nasa.gov"
    return (url_text_server, url_data_server)


def _get_pps_nrt_product_folder_name(product):
    """ "Retrieve NASA PPS server folder name for NRT product_type."""
    folder_name = get_product_info(product).get("pps_nrt_dir", None)
    if folder_name is None:
        raise ValueError(
            f"The pps_nrt_dir key of the {product} product is not specified in the config files."
        )
    return folder_name


def _get_pps_rs_product_folder_name(product):
    """ "Retrieve NASA PPS server folder name for RS product_type."""
    folder_name = get_product_info(product).get("pps_rs_dir", None)
    if folder_name is None:
        raise ValueError(
            f"The pps_rs_dir key of the {product} product is not specified in the config files."
        )
    return folder_name


def _get_pps_nrt_product_dir(product, date):
    """
    Retrieve the NASA PPS server directory structure where NRT data are stored.

    Parameters
    ----------
    product : str
        GPM product name. See: gpm_api.available_products() .
    date : datetime.date
        Single date for which to retrieve the data.
        Note: this is currently only needed when retrieving IMERG data.
    """
    folder_name = _get_pps_nrt_product_folder_name(product)
    # Specify the directory tree
    if product in available_products(product_type="NRT", product_category="IMERG"):
        directory_tree = os.path.join(folder_name, datetime.datetime.strftime(date, "%Y%m"))
    else:
        directory_tree = folder_name
    return directory_tree


def _get_pps_rs_product_dir(product, date, version):
    """
    Retrieve the NASA PPS server directory structure where RS data are stored.

    Parameters
    ----------
    product : str
        GPM product name. See: gpm_api.available_products() .

    date : datetime.date
        Single date for which to retrieve the data.
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
    """
    version = check_product_version(version, product)
    check_product_validity(product, product_type="RS")

    # Retrieve NASA server folder name for RS
    folder_name = _get_pps_rs_product_folder_name(product)

    # Specify the directory tree for current RS version
    if version == 7:
        directory_tree = os.path.join(
            "gpmdata",
            datetime.datetime.strftime(date, "%Y/%m/%d"),
            folder_name,
        )

    # Specify the directory tree for old RS version
    else:  #  version in [4, 5, 6]:
        version_str = "V0" + str(int(version))
        directory_tree = os.path.join(
            "gpmallversions",
            version_str,
            datetime.datetime.strftime(date, "%Y/%m/%d"),
            folder_name,
        )

    # Return the directory tree
    return directory_tree


def _get_pps_directory_tree(product, product_type, date, version):
    """
    Retrieve the NASA PPS server directory tree where the GPM data are stored.

    Parameters
    ----------
    product : str
        GPM product name. See: gpm_api.available_products() .
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    date : datetime.date
        Single date for which to retrieve the data.
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.

    Returns
    -------
    directory_tree : str
        DIrectory tree on the NASA PPS server where the data are stored.
    """
    check_product_type(product_type)
    if product_type == "NRT":
        return _get_pps_nrt_product_dir(product, date)
    else:  # product_type == "RS"
        return _get_pps_rs_product_dir(product, date, version)


def _get_pps_directory(product, product_type, date, version):
    """
    Retrieve the NASA PPS server directory paths where the GPM data for
    a specific date are listed and stored.

    The data list is retrieved using https.
    The data stored are retrieved using ftps.

    Parameters
    ----------
    product : str
        GPM product name. See: gpm_api.available_products() .
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    date : datetime.date
        Single date for which to retrieve the data.
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.

    Returns
    -------
    url_data_server : str
        url of the NASA PPS server where the data are stored.
    url_data_list: list
        url of the NASA PPS server where the data are listed.

    """
    # Retrieve servers URLs
    url_text_server, url_data_server = _get_pps_servers(product_type)

    # Retrieve directory tree structure
    dir_structure = _get_pps_directory_tree(
        product=product, product_type=product_type, date=date, version=version
    )

    # Define url where data are listed
    url_data_list = os.path.join(url_text_server, dir_structure)

    # Return tuple
    return (url_data_server, url_data_list)


####--------------------------------------------------------------------------.
############################
#### Filepath retrieval ####
############################


def ensure_valid_start_date(start_date, product):
    """Ensure that the product directory exists on the PPS server."""
    # TODO: where it is used ?
    if product == "2A-SAPHIR-MT1-CLIM":
        min_start_date = "2011-10-13 00:00:00"
    elif "1A-" in product or "1B-" in product:
        min_start_date = "1997-12-07 00:00:00"
    elif product in available_products(product_category="PMW"):
        min_start_date = "1987-07-09 00:00:00"
    elif product in available_products(product_category="RADAR") or product in available_products(
        product_category="CMB"
    ):
        min_start_date = "1997-12-07 00:00:00"
    elif "IMERG" in product:
        min_start_date = "2000-06-01 00:00:00"
    else:
        min_start_date = "1987-07-09 00:00:00"
    min_start_date = datetime.datetime.fromisoformat(min_start_date)
    start_date = max(start_date, min_start_date)
    return start_date


def _get_pps_file_list(url_file_list, product, date, version, verbose=True):
    """
    Retrieve the filepaths of the files available on the NASA PPS server for a specific day and product.

    The query is done using https !
    The function does not return the full PPS server url, but the filepath
    from the server root: i.e: '/gpmdata/2020/07/05/radar/<...>.HDF5'

    Parameters
    ----------
    product : str
        GPM product acronym. See gpm_api.available_products() .
    date : datetime
        Single date for which to retrieve the data.
    verbose : bool, optional
        Default is False. Whether to specify when data are not available for a specific date.
    """
    # TODO: maybe remove arguments product, date, version, verbose
    # --> Define try catch outside !

    # Retrieve GPM-API configs
    username = get_gpm_username(None)
    password = get_gpm_password(None)

    # Ensure url_file_list ends with "/"
    if url_file_list[-1] != "/":
        url_file_list = url_file_list + "/"

    # Define curl command
    # -k is required with curl > 7.71 otherwise results in "unauthorized access".
    cmd = "curl -k --user " + username + ":" + password + " " + url_file_list

    # Run command
    args = cmd.split()
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = process.communicate()[0].decode()

    # Check if server is available
    if stdout == "":
        version_str = str(int(version))
        print("The PPS server is currently unavailable.")
        print(
            f"This occurred when searching for product {product} (V0{version_str}) at date {date}."
        )
        raise ValueError("Sorry for the inconvenience.")

    # Check if data are available
    if stdout[0] == "<":
        if verbose:
            version_str = str(int(version))
            print(f"No data found on PPS on date {date} for product {product} (V0{version_str})")
        return []
    else:
        # Retrieve filepaths
        filepaths = stdout.split()

    # Return file paths
    return filepaths


def get_pps_daily_filepaths(product, product_type, date, version, verbose=True):
    """
    Retrieve the complete url to the files available on the NASA PPS server for a specific day and product.

    Parameters
    ----------
    product : str
        GPM product acronym. See gpm_api.available_products() .
    date : datetime
        Single date for which to retrieve the data.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
    verbose : bool, optional
        Whether to specify when data are not available for a specific date.
        The default is True.
    """
    # Retrieve server urls of NASA PPS
    (url_data_server, url_file_list) = _get_pps_directory(
        product=product, product_type=product_type, date=date, version=version
    )
    # Retrieve filepaths
    # - If empty: return []
    filepaths = _get_pps_file_list(
        url_file_list=url_file_list,
        product=product,
        date=date,
        version=version,
        verbose=verbose,
    )

    # Define the complete url of pps filepaths
    # - Need to remove the starting "/" to each filepath
    pps_fpaths = [os.path.join(url_data_server, filepath[1:]) for filepath in filepaths]

    # Return the pps data server filepaths
    return pps_fpaths


def define_pps_filepath(product, product_type, date, version, filename):
    """Define PPS filepath from filename."""
    # Retrieve PPS directory tree
    dir_tree = _get_pps_directory_tree(
        product=product, product_type=product_type, date=date, version=version
    )
    # Retrieve PPS servers URLs
    url_text_server, url_data_server = _get_pps_servers(product_type)
    # Define PPS filepath
    fpath = os.path.join(url_data_server, dir_tree, filename)
    return fpath


####--------------------------------------------------------------------------.
#################
#### Utility ####
#################


def find_first_pps_granule_filepath(product: str, product_type: str, version: int) -> str:
    """Return the PPS filepath of the first available granule."""
    from gpm_api.io.find import find_filepaths

    # Retrieve product start_time from product.yaml file.
    start_time = get_product_info(product).get("start_time", None)
    if start_time is None:
        raise ValueError(f"{product} product start_time is not provided in the product.yaml file.")

    # Find filepath
    end_time = start_time + relativedelta(days=1)
    pps_filepaths = find_filepaths(
        protocol="pps",
        product=product,
        start_time=start_time,
        end_time=end_time,
        version=version,
        product_type=product_type,
    )
    if len(pps_filepaths) == 0:
        raise ValueError(f"No PPS files found for {product} product around {start_time}.")
    return pps_filepaths[0]
