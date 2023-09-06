#!/usr/bin/env python3
"""
Created on Thu Oct 13 16:48:22 2022

@author: ghiggi
"""
import datetime
import os

from gpm_api.io.checks import (
    check_base_dir,
    check_product_type,
    check_product_validity,
    check_version,
)
from gpm_api.io.products import available_products, get_info_dict

####--------------------------------------------------------------------------.
####################
#### LOCAL DISK ####
####################


def get_product_category(product):
    """Get the product_category of a GPM product.

    The product_category is used to organize file on disk.
    """
    product_category = get_info_dict()[product].get("product_category", None)
    if product_category is None:
        raise ValueError(
            f"The product_category for {product} product is not specified in the config files"
        )
    return product_category


def get_disk_dir_pattern(product, product_type, version):
    """
    Defines the local (disk) repository base pattern where data are stored and searched.

    Parameters
    ----------
    product : str
        GPM product name. See: gpm_api.available_products()
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.

    Returns
    -------

    pattern : str
        Directory base pattern.
        If product_type == "RS": GPM/RS/V<version>/<product_category>/<product>
        If product_type == "NRT": GPM/NRT/<product_category>/<product>
        Product category are: RADAR, PMW, CMB, IMERG

    """
    # Define pattern
    product_category = get_product_category(product)
    if product_type == "NRT":
        dir_structure = os.path.join("GPM", product_type, product_category, product)
    else:  # product_type == "RS"
        version_str = "V0" + str(int(version))
        dir_structure = os.path.join("GPM", product_type, version_str, product_category, product)
    return dir_structure


def get_time_tree(date):
    """Get time tree path."""
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    time_tree = os.path.join(year, month, day)
    return time_tree


def get_disk_product_directory(base_dir, product, product_type, version):
    """
    Provide the disk product directory path where the requested GPM data are stored/need to be saved.

    Parameters
    ----------
    base_dir : str
        The base directory where to store GPM data.
    product : str
        GPM product name. See: gpm_api.available_products()
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.

    Returns
    -------

    product_dir : str
        Product directory path where data are located.
    """
    base_dir = check_base_dir(base_dir)
    product_dir_pattern = get_disk_dir_pattern(product, product_type, version)
    product_dir = os.path.join(base_dir, product_dir_pattern)
    return product_dir


def get_disk_directory(base_dir, product, product_type, version, date):
    """
    Provide the disk repository path where the requested daily GPM data are stored/need to be saved.

    Parameters
    ----------
    base_dir : str
        The base directory where to store GPM data.
    product : str
        GPM product name. See: gpm_api.available_products()
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
    date : datetime.date
        Single date for which to retrieve the data.

    Returns
    -------

    dir_path : str
        Directory path where daily GPM data are located.
        If product_type == "RS": <base_dir>/GPM/RS/V0<version>/<product_category>/<product>/<YYYY>/<MM>/<DD>
        If product_type == "NRT": <base_dir>/GPM/NRT/<product_category>/<product>/<YYYY>/<MM>/<DD>
        <product_category> are: RADAR, PMW, CMB, IMERG.

    """
    product_dir = get_disk_product_directory(
        base_dir=base_dir, product=product, product_type=product_type, version=version
    )
    time_tree = get_time_tree(date)
    dir_path = os.path.join(product_dir, time_tree)
    return dir_path


####--------------------------------------------------------------------------.
####################
#### PPS SERVER ####
####################


def _get_pps_nrt_product_folder_name(product):
    """ "Retrieve NASA PPS server folder name for NRT product_type."""
    folder_name = get_info_dict()[product].get("pps_nrt_dir", None)
    if folder_name is None:
        raise ValueError(
            f"The pps_nrt_dir key of the {product} product is not specified in the config files."
        )
    return folder_name


def _get_pps_rs_product_folder_name(product):
    """ "Retrieve NASA PPS server folder name for RS product_type."""
    folder_name = get_info_dict()[product].get("pps_rs_dir", None)
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
    check_version(version)
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


def get_pps_directory_tree(product, product_type, date, version):
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


def get_pps_servers(product_type):
    """Return the url to the PPS servers."""
    if product_type == "NRT":
        url_text_server = "https://jsimpsonhttps.pps.eosdis.nasa.gov/text"
        url_data_server = "ftps://jsimpsonftps.pps.eosdis.nasa.gov/data"
    else:
        url_text_server = "https://arthurhouhttps.pps.eosdis.nasa.gov/text"
        url_data_server = "ftps://arthurhouftps.pps.eosdis.nasa.gov"
    return (url_text_server, url_data_server)


def get_pps_directory(product, product_type, date, version):
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
    check_product_type(product_type)

    # Retrieve servers URLs
    url_text_server, url_data_server = get_pps_servers(product_type)

    # Retrieve directory tree structure
    dir_structure = get_pps_directory_tree(
        product=product, product_type=product_type, date=date, version=version
    )

    # Define url where data are listed
    url_data_list = os.path.join(url_text_server, dir_structure)

    # Return tuple
    return (url_data_server, url_data_list)
