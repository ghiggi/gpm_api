#!/usr/bin/env python3
"""
Created on Mon Oct  9 12:44:42 2023

@author: ghiggi
"""
import os
import re
import subprocess

from gpm_api.io.products import get_info_dict, is_trmm_product

###---------------------------------------------------------------------------.


def get_ges_disc_dir_key(product):
    info_dict = get_info_dict()[product]
    dir_pattern = info_dict["ges_disc_dir"]
    return dir_pattern


def _get_gesc_disc_product_level_dirname(product):
    dir_pattern = get_ges_disc_dir_key(product)
    if isinstance(dir_pattern, str):
        return dir_pattern.split("/")[0]
    else:
        return None


def _get_gesc_disc_product_name(product):
    dir_pattern = get_ges_disc_dir_key(product)
    if isinstance(dir_pattern, str):
        return dir_pattern.split("/")[1]
    else:
        return None


def _get_ges_disc_url_content(url):
    cmd = f"wget -O - {url}"
    args = cmd.split()
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = process.communicate()[0].decode()
    # Check if server is available
    if stdout == "":
        raise ValueError(
            "The GES DISC data archive is currently unavailable. Sorry for the inconvenience."
        )
    return stdout


def _get_href_value(input_string):
    """Infer href value."""
    match = re.search(r'<a\s+href="([^"]+)"', input_string)
    # Check if a match was found and extract the value
    if match:
        href_value = match.group(1)
    else:
        href_value = ""
    # Exclude .xml files and doc directory
    if ".xml" in href_value or "doc/" in href_value:
        href_value = ""
    return href_value


def _get_gesc_disc_list_path(url):
    wget_output = _get_ges_disc_url_content(url)
    list_content = [_get_href_value(s) for s in wget_output.split("alt=")[4:]]
    list_content = [s for s in list_content if s != ""]
    if len(list_content) == 0:
        dirname = os.path.basename(url)
        raise ValueError(f"The GES DISC {dirname} directory is empty.")
    list_path = [os.path.join(url, s) for s in list_content]
    return list_path


####--------------------------------------------------------------------------.
#####################
#### Directories ####
#####################

# _get_pps_servers


def get_ges_disc_base_url(product):
    # TRMM
    if is_trmm_product(product):
        ges_disc_base_url = "https://disc2.gesdisc.eosdis.nasa.gov/data/"

    # GPM
    else:
        ges_disc_base_url = "https://gpm1.gesdisc.eosdis.nasa.gov/data"
        ges_disc_base_url = "https://gpm2.gesdisc.eosdis.nasa.gov/data"
    return ges_disc_base_url


def get_ges_disc_product_path(product, version):
    base_url = get_ges_disc_base_url(product)
    dir_pattern = get_ges_disc_dir_key(product)
    if isinstance(dir_pattern, str):
        dir_pattern = f"{dir_pattern}.0{version}"
    url = os.path.join(base_url, dir_pattern)
    return url


####--------------------------------------------------------------------------.
############################
#### Filepath Retrieval ####
############################


def _get_gesdisc_servers(product):
    pass


def _get_gesdisc_directory_tree(product, product_type, date, version):
    pass


def _get_gesdisc_file_list(url_file_list, product, date, version, verbose=True):
    # --> HTTP request !
    pass


def _get_gesdisc_directory(product, product_type, date, version):
    """
    Retrieve the NASA GES DISC server directory paths where the GPM data for
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
        url of the NASA GES DISC server where the data are stored.
    url_data_list: list
        url of the NASA GES DISC server where the data are listed.

    """
    # Retrieve servers URLs
    url_text_server, url_data_server = _get_gesdisc_servers(product_type)

    # Retrieve directory tree structure
    dir_structure = _get_gesdisc_directory_tree(
        product=product, product_type=product_type, date=date, version=version
    )

    # Define url where data are listed
    url_data_list = os.path.join(url_text_server, dir_structure)

    # Return tuple
    return (url_data_server, url_data_list)


def get_gesdisc_daily_filepaths(product, product_type, date, version, verbose=True):
    """
    Retrieve the complete url to the files available on the NASA GES DISC server for a specific day and product.

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
    # Retrieve server urls of NASA GES DISC
    (url_data_server, url_file_list) = _get_gesdisc_directory(
        product=product, product_type=product_type, date=date, version=version
    )
    # Retrieve filepaths
    # - If empty: return []
    filepaths = _get_gesdisc_file_list(
        url_file_list=url_file_list,
        product=product,
        date=date,
        version=version,
        verbose=verbose,
    )

    # Define the complete url of gesdisc filepaths
    # - Need to remove the starting "/" to each filepath
    gesdisc_fpaths = [os.path.join(url_data_server, filepath[1:]) for filepath in filepaths]

    # Return the gesdisc data server filepaths
    return gesdisc_fpaths


def define_gesdisc_filepath(product, product_type, date, version, filename):
    """Define GES DISC filepath from filename."""
    # Retrieve GES DISC directory tree
    dir_tree = _get_gesdisc_directory_tree(
        product=product, product_type=product_type, date=date, version=version
    )
    # Retrieve GES DISC servers URLs
    url_text_server, url_data_server = _get_gesdisc_servers(product_type)
    # Define GES DISC filepath
    fpath = os.path.join(url_data_server, dir_tree, filename)
    return fpath
