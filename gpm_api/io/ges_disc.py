#!/usr/bin/env python3
"""
Created on Mon Oct  9 12:44:42 2023

@author: ghiggi
"""
import datetime
import re
import subprocess

from gpm_api.io.products import get_product_info, is_trmm_product

###---------------------------------------------------------------------------.
###########################
#### GES DISC scraping ####
###########################


def _get_ges_disc_url_content(url):
    cmd = f"wget -O - {url}"
    args = cmd.split()
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = process.communicate()[0].decode()
    # Check if server is available
    if stdout == "":
        raise ValueError(f"The requested url {url} was not found on the GES DISC server.")
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


def _get_ges_disc_list_path(url):
    # Retrieve url content
    # - If it returns something, means url is correct
    wget_output = _get_ges_disc_url_content(url)
    # Retrieve content
    list_content = [_get_href_value(s) for s in wget_output.split("alt=")[4:]]
    list_content = [s for s in list_content if s != ""]
    if len(list_content) == 0:
        raise ValueError(f"The GES DISC {url} directory is empty.")
    list_path = [f"{url}/{s}" for s in list_content]
    return list_path


# # Empty directory
# url = "https://gpm2.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.07/"
# url = "https://gpm2.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.07"

# # Unexisting directory
# url = "https://gpm2.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.07/2020"


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
    folder_name = f"{dir_pattern}.0{version}"
    return folder_name


def _get_ges_disc_product_directory_tree(product, date, version):
    """Return the GES DISC product directory tree.

    The directory tree structure is
     - <product directory>/YYYY/DOY for L1 and L2 products (and IMERG half hourly)
     - <product directory>/YYYY/MM for L3 daily products
     - <product directory>/YYYY or <product directory>/YYYY/MM for L3 monthly products

    Parameters
    ----------
    product : str
        GPM product name. See: gpm_api.available_products() .
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
    directory_tree = "/".join(
        [
            folder_name,
            datetime.datetime.strftime(date, "%Y/%j"),
        ]
    )
    return directory_tree


def get_ges_disc_product_directory(product, date, version):
    """
    Retrieve the NASA GES DISC server product directory path at a specific date.

    The data list is retrieved using https.

    Parameters
    ----------
    product : str
        GPM product name. See: gpm_api.available_products() .
    date : datetime.date
        Single date for which to retrieve the data.
    version : int, optional
        GPM version of the data to retrieve.

    Returns
    -------
    url_data_list : str
        url of the NASA GES DISC server where the data are stored.
    """
    # Retrieve server URL
    url_server = _get_ges_disc_server(product)
    # Retrieve directory tree structure
    dir_structure = _get_ges_disc_product_directory_tree(
        product=product, date=date, version=version
    )
    # Define product directory where data are listed
    url_product_dir = f"{url_server}/{dir_structure}"
    return url_product_dir


####--------------------------------------------------------------------------.
############################
#### Filepath Retrieval ####
############################


def _get_gesdisc_file_list(url_product_dir, product, date, version, verbose=True):
    """
    Retrieve NASA GES DISC filepaths for a specific day and product.

    The query is done using https !
    The function does return the full GES DISC url file paths.
    The returned file paths refers to a single product !!!

    Parameters
    ----------
    url_product_dir : str
        The GES DISC product directory url.
    product : str
        GPM product acronym. See gpm_api.available_products() .
    date : datetime
        Single date for which to retrieve the data.
    verbose : bool, optional
        Default is False. Whether to specify when data are not available for a specific date.
    """
    try:
        filepaths = _get_ges_disc_list_path(url_product_dir)
    except Exception as e:
        # If url not exist, raise an error
        if "was not found on the GES DISC server" in str(e):
            raise e
        else:
            # If no filepath (empty directory), print message if verbose=True
            if verbose:
                version_str = str(int(version))
                msg = f"No data found on GES DISC on date {date} for product {product} (V0{version_str})"
                print(msg)
            filepaths = []
    return filepaths


def get_gesdisc_daily_filepaths(product, product_type, date, version, verbose=True):
    """
    Retrieve the NASA GES DISC file paths available at a given date.

    Parameters
    ----------
    product : str
        GPM product acronym. See gpm_api.available_products() .
    date : datetime
        Single date for which to retrieve the data.
    product_type : str
        GPM product type. Not used for GES DISC.
    version : int
        GPM version of the data to retrieve.
    verbose : bool, optional
        Whether to specify when data are not available for a specific date.
        The default is True.
    """
    if product_type == "NRT" and "IMERG" not in product:
        raise ValueError("The only available NRT products on GES DISC are IMERG-ER and IMERG-FR")
    # Retrieve server urls of NASA GES DISC
    url_product_dir = get_ges_disc_product_directory(product=product, date=date, version=version)
    # Retrieve GES DISC filepaths
    # - If empty: return []
    filepaths = _get_gesdisc_file_list(
        url_product_dir=url_product_dir,
        product=product,
        date=date,
        version=version,
        verbose=verbose,
    )
    return filepaths


def define_gesdisc_filepath(product, product_type, date, version, filename):
    """Define GES DISC filepath from filename.

    Parameters
    ----------
    product : str
        GPM product acronym. See gpm_api.available_products().
    product_type : str
            GPM product type. Not used for GES DISC.
    date : datetime
        Single date for which to retrieve the data.
    version : int
        GPM version of the data to retrieve if product_type = 'RS'.
    filename : str
        Name of the GPM file.
    """
    if product_type == "NRT" and "IMERG" not in product:
        raise ValueError("The only available NRT products on GES DISC are IMERG-ER and IMERG-FR")
    # Retrieve product directory url
    url_product_dir = get_ges_disc_product_directory(product=product, date=date, version=version)
    # Define GES DISC filepath
    fpath = f"{url_product_dir}/{filename}"
    return fpath
