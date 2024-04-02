# -----------------------------------------------------------------------------.
# MIT License

# Copyright (c) 2024 GPM-API developers
#
# This file is part of GPM-API.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -----------------------------------------------------------------------------.
"""This module contains the routines required to search data on the NASA PPS servers."""
import datetime
import subprocess

from dateutil.relativedelta import relativedelta

from gpm.configs import get_password_pps, get_username_pps
from gpm.io.checks import (
    check_product_type,
    check_product_validity,
    check_product_version,
)
from gpm.io.products import available_products, get_product_info

####--------------------------------------------------------------------------.
#####################
#### Directories ####
#####################


def _get_pps_text_server(product_type):
    """Return the url to the PPS text servers."""
    if product_type == "NRT":
        url_text_server = "https://jsimpsonhttps.pps.eosdis.nasa.gov/text"
    else:
        url_text_server = "https://arthurhouhttps.pps.eosdis.nasa.gov/text"
    return url_text_server


def _get_pps_data_server(product_type):
    """Return the url to the PPS data servers."""
    if product_type == "NRT":
        url_data_server = "ftps://jsimpsonftps.pps.eosdis.nasa.gov/data"
    else:
        url_data_server = "ftps://arthurhouftps.pps.eosdis.nasa.gov"
    return url_data_server


def _get_pps_nrt_product_folder_name(product):
    """Retrieve NASA PPS server folder name for NRT product_type."""
    folder_name = get_product_info(product).get("pps_nrt_dir", None)
    if folder_name is None:
        raise ValueError(
            f"The pps_nrt_dir key of the {product} product is not specified in the config files.",
        )
    return folder_name


def _get_pps_rs_product_folder_name(product):
    """Retrieve NASA PPS server folder name for RS product_type."""
    folder_name = get_product_info(product).get("pps_rs_dir", None)
    if folder_name is None:
        raise ValueError(
            f"The pps_rs_dir key of the {product} product is not specified in the config files.",
        )
    return folder_name


def _get_pps_nrt_product_dir(product, date):
    """Retrieve the NASA PPS server directory structure where NRT data are stored.

    Parameters
    ----------
    product : str
        GPM product name. See ``gpm.available_products()``.
    date : datetime.date
        Single date for which to retrieve the data.
        Note: this is currently only needed when retrieving IMERG data.

    """
    folder_name = _get_pps_nrt_product_folder_name(product)
    # Specify the directory tree
    if product in available_products(product_types="NRT", product_categories="IMERG"):
        directory_tree = f"{folder_name}/{datetime.datetime.strftime(date, '%Y%m')}"
    else:
        directory_tree = folder_name
    return directory_tree


def _get_pps_rs_product_dir(product, date, version):
    """Retrieve the NASA PPS server directory structure where RS data are stored.

    Parameters
    ----------
    product : str
        GPM product name. See ``gpm.available_products()``. .
    date : datetime.date
        Single date for which to retrieve the data.
    version : int
        GPM version of the data to retrieve if ``product_type = "RS"``.

    """
    version = check_product_version(version, product)
    product = check_product_validity(product, product_type="RS")

    # Retrieve NASA server folder name for RS
    folder_name = _get_pps_rs_product_folder_name(product)

    # Specify the directory tree for current RS version
    if version == 7:
        directory_tree = "/".join(
            [
                "gpmdata",
                datetime.datetime.strftime(date, "%Y/%m/%d"),
                folder_name,
            ],
        )
    # Specify the directory tree for old RS version
    else:  #  version in [4, 5, 6]:
        version_str = "V0" + str(int(version))
        directory_tree = "/".join(
            [
                "gpmallversions",
                version_str,
                datetime.datetime.strftime(date, "%Y/%m/%d"),
                folder_name,
            ],
        )

    # Return the directory tree
    return directory_tree


def _get_pps_directory_tree(product, product_type, date, version):
    """Retrieve the NASA PPS server directory tree where the GPM data are stored.

    The directory tree structure for ``product_type="RS"`` is:
        - ``<gpmallversions>/V0<version>/<pps_rs_dir>/YYYY/MM/DD``
        -  The L3 monthly products are saved in the ``YYYY/MM/01`` directory

    The directory tree structure for ``product_type="NRT"`` is:
        - IMERG-ER and IMERG-FR: ``imerg/<early/late>/YYYY/MM/``
        - Otherwise ``<pps_nrt_dir>/``

    Parameters
    ----------
    product : str
        GPM product name. See ``gpm.available_products()``. .
    product_type : str
        GPM product type. Either ``RS`` (Research) or ``NRT`` (Near-Real-Time).
    date : datetime.date
        Single date for which to retrieve the data.
    version : int
        GPM version of the data to retrieve if ``product_type = "RS"``.

    Returns
    -------
    directory_tree : str
        DIrectory tree on the NASA PPS server where the data are stored.

    """
    product_type = check_product_type(product_type)
    if product_type == "NRT":
        return _get_pps_nrt_product_dir(product, date)
    # product_type == "RS"
    return _get_pps_rs_product_dir(product, date, version)


def get_pps_product_directory(product, product_type, date, version, server_type):
    """Retrieve the NASA PPS server product directory path at specific date.

    The data list is retrieved using https.
    The data stored are retrieved using ftps.

    Parameters
    ----------
    product : str
        GPM product name. See ``gpm.available_products()``.
    product_type : str
        GPM product type. Either ``RS`` (Research) or ``NRT`` (Near-Real-Time).
    date : datetime.date
        Single date for which to retrieve the data.
    version : int
        GPM version of the data to retrieve if ``product_type = "RS"``.
    server_type: str
        Either ``text`` or ``data``

    Returns
    -------
    url_product_dir : str
        url of the NASA PPS server where the data are listed.

    """
    # Retrieve server URL
    url_server = _get_pps_text_server(product_type) if server_type == "text" else _get_pps_data_server(product_type)
    # Retrieve directory tree structure
    dir_structure = _get_pps_directory_tree(
        product=product,
        product_type=product_type,
        date=date,
        version=version,
    )
    # Define product directory where data are listed
    return f"{url_server}/{dir_structure}"


####--------------------------------------------------------------------------.
############################
#### Filepath retrieval ####
############################


def _try_get_pps_file_list(url_product_dir):
    # Retrieve GPM-API configs
    username = get_username_pps()
    password = get_password_pps()
    # Ensure url_file_list ends with "/"
    if url_product_dir[-1] != "/":
        url_product_dir = url_product_dir + "/"
    # Define curl command
    # -k is required with curl > 7.71 otherwise results in "unauthorized access".
    cmd = f"curl -k --user {username}:{password} {url_product_dir}"
    # Run command
    args = cmd.split()
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = process.communicate()[0].decode()
    # Check if server is available
    if stdout == "":
        raise ValueError("The PPS server is currently unavailable. Sorry for the inconvenience.")
    # Check if there are data are available
    if stdout[0] == "<":
        raise ValueError("No data found on PPS.")
    # Retrieve filepaths
    return stdout.split()


def _get_pps_file_list(url_product_dir, product, date, version, verbose=True):
    """Retrieve the filepaths of the files available on the NASA PPS server for a specific day.

    The query is done using https !
    The function does not return the full PPS server url, but the filepath
    from the server root: i.e: ``'/gpmdata/2020/07/05/radar/<...>.HDF5'``
    The returned filepaths can includes more than one product !!!

    Parameters
    ----------
    url_product_dir : str
        The PPS product directory url.
    product : str
        GPM product acronym. See ``gpm.available_products()``.
    date : datetime
        Single date for which to retrieve the data.
    verbose : bool, optional
        Default is ``False``. Whether to specify when data are not available for a specific date.

    """
    try:
        filepaths = _try_get_pps_file_list(url_product_dir)
    except Exception as e:
        # If url not exist, raise an error
        if "The PPS server is currently unavailable." in str(e):
            raise e
        if "No data found on PPS." in str(e):
            # If no filepath (empty directory), print message if verbose=True
            if verbose:
                version_str = str(int(version))
                msg = f"No data found on PPS on date {date} for product {product} (V0{version_str})"
                print(msg)
            filepaths = []
        else:
            raise ValueError(f"Undefined error. The error is {e}.")
    return filepaths


def get_pps_daily_filepaths(product, product_type, date, version, verbose=True):
    """Retrieve the complete url to the files available on the NASA PPS server for a specific day and product.

    Parameters
    ----------
    product : str
        GPM product acronym. See ``gpm.available_products()``.
    date : datetime
        Single date for which to retrieve the data.
    product_type : str, optional
        GPM product type. Either ``RS`` (Research) or ``NRT`` (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if ``product_type = "RS"``.
    verbose : bool, optional
        Whether to specify when data are not available for a specific date.
        The default is ``True``.

    """
    # Retrieve url to product directory
    url_product_dir = get_pps_product_directory(
        product=product,
        product_type=product_type,
        date=date,
        version=version,
        server_type="text",
    )
    # Retrieve filepaths from the PPS base directory of the server
    # - If empty: return []
    # - Example /gpmdata/2020/07/05/radar/<...>.HDF5'
    filepaths = _get_pps_file_list(
        url_product_dir=url_product_dir,
        product=product,
        date=date,
        version=version,
        verbose=verbose,
    )
    # Define the complete url of pps filepaths
    # Filepaths start with a "/"
    url_data_server = _get_pps_data_server(product_type)
    return [f"{url_data_server}{filepath}" for filepath in filepaths]


def define_pps_filepath(product, product_type, date, version, filename):
    """Define PPS filepath from filename.

    This function is called by ``get_filepath_from_filename(filename, storage, product_type)``.
    """
    # Retrieve product directory url
    url_product_dir = get_pps_product_directory(
        product=product,
        product_type=product_type,
        date=date,
        version=version,
        server_type="data",
    )
    # Define PPS filepath
    return f"{url_product_dir}/{filename}"


####--------------------------------------------------------------------------.
#################
#### Utility ####
#################


def find_first_pps_granule_filepath(product: str, product_type: str, version: int) -> str:
    """Return the PPS filepath of the first available granule."""
    from gpm.io.find import find_filepaths

    # Retrieve product start_time from product.yaml file.
    start_time = get_product_info(product).get("start_time", None)
    if start_time is None:
        raise ValueError(f"{product} product start_time is not provided in the product.yaml file.")

    # Find filepath
    end_time = start_time + relativedelta(days=1)
    pps_filepaths = find_filepaths(
        storage="PPS",
        product=product,
        start_time=start_time,
        end_time=end_time,
        version=version,
        product_type=product_type,
    )
    if len(pps_filepaths) == 0:
        raise ValueError(f"No PPS files found for {product} product around {start_time}.")
    pps_filepaths = sorted(pps_filepaths)
    return pps_filepaths[0]
