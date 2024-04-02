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
"""Routines required to search data on the NASA GES DISC servers."""

import datetime
import re
import shlex
import subprocess

from gpm.io.products import get_product_info, is_trmm_product

###---------------------------------------------------------------------------.
###########################
#### GES DISC scraping ####
###########################


def _get_ges_disc_url_content(url):
    # cmd = f"wget -O - {url}"
    cmd = f"curl -L {url}"
    list_cmd = shlex.split(cmd)
    process = subprocess.Popen(list_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = process.communicate()[0].decode()
    # Check if server is available
    if stdout == "":
        raise ValueError(f"The requested url {url} was not found on the GES DISC server.")
    if "The requested URL was not found on this server" in stdout:
        raise ValueError(f"The requested url {url} was not found on the GES DISC server.")
    return stdout


def _get_href_value(input_string):
    """Infer href value."""
    match = re.search(r'<a\s+href="([^"]+)"', input_string)
    # Check if a match was found and extract the value
    href_value = match.group(1) if match else ""
    # Exclude .xml files and doc directory
    if ".xml" in href_value or "doc/" in href_value:
        href_value = ""
    return href_value


def _get_ges_disc_list_path(url):
    # Retrieve url content
    # - If it returns something, means url is correct
    output = _get_ges_disc_url_content(url)
    # Retrieve content
    list_content = [_get_href_value(s) for s in output.split("alt=")[4:]]
    list_content = [s for s in list_content if s != ""]
    if len(list_content) == 0:
        raise ValueError(f"The GES DISC {url} directory is empty.")
    return [f"{url}/{s}" for s in list_content]


####--------------------------------------------------------------------------.
#####################
#### Directories ####
#####################


def _get_ges_disc_server(product):
    # TRMM
    if is_trmm_product(product):
        ges_disc_base_url = "https://disc2.gesdisc.eosdis.nasa.gov/data"

    # GPM
    else:
        # ges_disc_base_url = "https://gpm1.gesdisc.eosdis.nasa.gov/data"
        ges_disc_base_url = "https://gpm2.gesdisc.eosdis.nasa.gov/data"
    return ges_disc_base_url


def _get_ges_disc_product_folder_name(product, version):
    dir_pattern = get_product_info(product)["ges_disc_dir"]
    return f"{dir_pattern}.0{version}"


def get_ges_disc_product_directory_tree(product, date, version):
    """Return the GES DISC product directory tree.

    The directory tree structure is

    - ``<product directory>/YYYY/DOY`` for L1 and L2 products (and IMERG half hourly)
    - ``<product directory>/YYYY/MM`` for L3 daily products
    - ``<product directory>/YYYY`` or ``<product directory>/YYYY/MM`` for L3 monthly products

    Parameters
    ----------
    product : str
        GPM product name. See ``gpm.available_products()``.
    date : datetime.date
        Single date for which to retrieve the data.
    version : int
        GPM version of the data to retrieve.

    Returns
    -------
    directory_tree : str
        DIrectory tree on the NASA GESC DISC server where the data are stored.

    """
    # Retrieve foldername
    folder_name = _get_ges_disc_product_folder_name(product, version)

    # Specify the directory tree
    # --> TODO: currently specified only for L1 and L2
    return "/".join(
        [
            folder_name,
            datetime.datetime.strftime(date, "%Y/%j"),
        ],
    )


def get_ges_disc_product_directory(product, date, version):
    """Retrieve the NASA GES DISC server product directory path at a specific date.

    The data list is retrieved using https.

    Parameters
    ----------
    product : str
        GPM product name. See ``gpm.available_products()``.
    date : datetime.date
        Single date for which to retrieve the data.
    version : int
        GPM version of the data to retrieve.

    Returns
    -------
    url_data_list : str
        url of the NASA GES DISC server where the data are stored.

    """
    # Retrieve server URL
    url_server = _get_ges_disc_server(product)
    # Retrieve directory tree structure
    dir_structure = get_ges_disc_product_directory_tree(product=product, date=date, version=version)
    # Define product directory where data are listed
    return f"{url_server}/{dir_structure}"


####--------------------------------------------------------------------------.
############################
#### Filepath Retrieval ####
############################


def _get_ges_disc_file_list(url_product_dir, product, date, version, verbose=True):
    """Retrieve NASA GES DISC filepaths for a specific day and product.

    The query is done using https !
    The function does return the full GES DISC url file paths.
    The returned file paths refers to a single product !!!

    Parameters
    ----------
    url_product_dir : str
        The GES DISC product directory url.
    product : str
        GPM product acronym. See ``gpm.available_products()``.
    date : datetime
        Single date for which to retrieve the data.
    verbose : bool, optional
        Default is ``False``. Whether to specify when data are not available for a specific date.

    """
    try:
        filepaths = _get_ges_disc_list_path(url_product_dir)
    except Exception as e:
        # If url not exist, raise an error
        if "was not found on the GES DISC server" in str(e):
            raise e
        # If no filepath (empty directory), print message if verbose=True
        if verbose:
            version_str = str(int(version))
            msg = f"No data found on GES DISC on date {date} for product {product} (V0{version_str})"
            print(msg)
        filepaths = []
    return filepaths


def _check_gesc_disc_product_type(product, product_type):
    if product_type == "NRT" and "IMERG" not in product:
        raise ValueError("The only available NRT products on GES DISC are IMERG-ER and IMERG-FR")


def get_ges_disc_daily_filepaths(product, product_type, date, version, verbose=True):
    """Retrieve the NASA GES DISC file paths available at a given date.

    Parameters
    ----------
    product : str
        GPM product acronym. See ``gpm.available_products()``.
    date : datetime
        Single date for which to retrieve the data.
    product_type : str
        GPM product type. Not used for GES DISC.
    version : int
        GPM version of the data to retrieve.
    verbose : bool, optional
        Whether to specify when data are not available for a specific date.
        The default is ``True``.

    """
    _check_gesc_disc_product_type(product=product, product_type=product_type)
    # Retrieve server urls of NASA GES DISC
    url_product_dir = get_ges_disc_product_directory(product=product, date=date, version=version)
    # Retrieve GES DISC filepaths
    # - If empty: return []
    return _get_ges_disc_file_list(
        url_product_dir=url_product_dir,
        product=product,
        date=date,
        version=version,
        verbose=verbose,
    )


def define_ges_disc_filepath(product, product_type, date, version, filename):
    """Define GES DISC filepath from filename.

    This function is called by get_filepath_from_filename(filename, storage, product_type).

    Parameters
    ----------
    product : str
        GPM product acronym. See ``gpm.available_products()``.
    product_type : str
        GPM product type. Not used for GES DISC.
    date : datetime
        Single date for which to retrieve the data.
    version : int
        GPM version of the data to retrieve if ``product_type = "RS"``.
    filename : str
        Name of the GPM file.

    """
    _check_gesc_disc_product_type(product=product, product_type=product_type)
    # Retrieve product directory url
    url_product_dir = get_ges_disc_product_directory(product=product, date=date, version=version)
    # Define GES DISC filepath
    return f"{url_product_dir}/{filename}"
