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
"""This module contains functions defining where to download GPM data on the local machine."""
import glob
import os
import pathlib

from gpm_api.configs import get_gpm_base_dir
from gpm_api.io.checks import check_base_dir
from gpm_api.io.products import get_product_category

####--------------------------------------------------------------------------.
#####################
#### Directories ####
#####################


def get_time_tree(date):
    """Get time tree path <YYYY>/<MM>/<DD>."""
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    time_tree = os.path.join(year, month, day)
    return time_tree


def _get_local_dir_pattern(product, product_type, version):
    """
    Defines the local (disk) repository base pattern where data are stored and searched.

    Parameters
    ----------
    product : str
        GPM product name. See: gpm_api.available_products()
    product_type : str
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int
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


def _get_local_product_base_directory(base_dir, product, product_type, version):
    """
    Provide the local product base directory path where the requested GPM data are stored.

    Parameters
    ----------
    base_dir : str
        The base directory where to store GPM data.
    product : str
        GPM product name. See: gpm_api.available_products()
    product_type : str
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int
        GPM version of the data to retrieve if product_type = 'RS'.

    Returns
    -------

    product_dir : str
        Product base directory path where data are located.
    """
    base_dir = check_base_dir(base_dir)
    product_dir_pattern = _get_local_dir_pattern(product, product_type, version)
    product_dir = os.path.join(base_dir, product_dir_pattern)
    return product_dir


def _get_local_directory_tree(product, product_type, date, version):
    """Return the local product directory tree.

    The directory tree structure for product_type == "RS" is:
        GPM/RS/V<version>/<product_category>/<product>/YYYY/MM/YY
    The directory tree structure for product_type == "NRT" is:
        GPM/NRT/<product_category>/<product>/YYYY/MM/YY

    Parameters
    ----------
    product : str
        GPM product name. See: gpm_api.available_products().
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    date : datetime.date
        Single date for which to retrieve the data.
    version : int
        GPM version of the data to retrieve.

    Returns
    -------
    directory_tree : str
        DIrectory tree on the NASA GESC DISC server where the data are stored.
    """
    # Define product directory: GPM/RS/V<version>/<product_category>/<product>
    product_dir_tree = _get_local_dir_pattern(product, product_type, version)
    # Define time tree
    time_tree = get_time_tree(date)
    # Define product directory tree for a specific date
    directory_tree = os.path.join(product_dir_tree, time_tree)
    return directory_tree


def get_local_product_directory(base_dir, product, product_type, version, date):
    """
    Provide the local repository path where the requested daily GPM data are stored/need to be saved.

    Parameters
    ----------
    base_dir : str
        The base directory where to store GPM data.
    product : str
        GPM product name. See: gpm_api.available_products()
    product_type : str
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int
        GPM version of the data to retrieve if product_type = 'RS'.
    date : datetime.date
        Single date for which to retrieve the data.

    Returns
    -------

    product_dir_path : str
        Directory path where daily GPM data are located.
        If product_type == "RS": <base_dir>/GPM/RS/V0<version>/<product_category>/<product>/<YYYY>/<MM>/<DD>
        If product_type == "NRT": <base_dir>/GPM/NRT/<product_category>/<product>/<YYYY>/<MM>/<DD>
        <product_category> are: RADAR, PMW, CMB, IMERG.

    """
    dir_structure = _get_local_directory_tree(
        product=product, product_type=product_type, version=version, date=date
    )
    product_dir_path = os.path.join(base_dir, dir_structure)
    return product_dir_path


####--------------------------------------------------------------------------.
############################
#### Filepath retrieval ####
############################


def get_local_daily_filepaths(product, product_type, date, version, verbose=True):
    """
    Retrieve GPM data filepaths on the local disk directory of a specific day and product.

    Parameters
    ----------
    product : str
        GPM product acronym. See gpm_api.available_products()
    product_type : str
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    date : datetime
        Single date for which to retrieve the data.
    version : int
        GPM version of the data to retrieve if product_type = 'RS'.
    verbose : bool, optional
        Whether to print processing details. The default is True.
    """
    # Retrieve the local GPM base directory
    base_dir = get_gpm_base_dir()
    base_dir = check_base_dir(base_dir)

    # Retrieve the directory on disk where the data are stored
    dir_path = get_local_product_directory(
        base_dir=base_dir,
        product=product,
        product_type=product_type,
        date=date,
        version=version,
    )

    # Check if the folder exists
    if not os.path.exists(dir_path):
        return []

    # Retrieve the file names in the directory
    filenames = sorted(os.listdir(dir_path))  # returns [] if empty

    # Retrieve the filepaths
    filepaths = [os.path.join(dir_path, filename) for filename in filenames]

    return filepaths


def define_local_filepath(product, product_type, date, version, filename):
    """Define local file path.

    This function is called by get_filepath_from_filename(filename, storage, product_type).
    """
    # Retrieve the local GPM base directory
    base_dir = get_gpm_base_dir()
    base_dir = check_base_dir(base_dir)

    # Define disk directory path
    dir_tree = get_local_product_directory(
        base_dir=base_dir,
        product=product,
        product_type=product_type,
        date=date,
        version=version,
    )
    # Define disk file path
    filepath = os.path.join(dir_tree, filename)
    return filepath


####--------------------------------------------------------------------------.
#################
#### Utility ####
#################


def _recursive_glob(dir_path, glob_pattern):
    # ** search for all files recursively
    # glob_pattern = os.path.join(base_dir, "**", "metadata", f"{station_name}.yml")
    # metadata_filepaths = glob.glob(glob_pattern, recursive=True)

    dir_path = pathlib.Path(dir_path)
    return [str(path) for path in dir_path.rglob(glob_pattern)]


def list_paths(dir_path, glob_pattern, recursive=False):
    """Return a list of filepaths and directory paths."""
    if not recursive:
        return glob.glob(os.path.join(dir_path, glob_pattern))
    else:
        return _recursive_glob(dir_path, glob_pattern)


def list_files(dir_path, glob_pattern, recursive=False):
    """Return a list of filepaths (exclude directory paths)."""
    paths = list_paths(dir_path, glob_pattern, recursive=recursive)
    filepaths = [f for f in paths if os.path.isfile(f)]
    return filepaths


def get_local_filepaths(product, version=7, product_type="RS"):
    """
    Retrieve all GPM filepaths on the local disk directory for a specific product.

    Parameters
    ----------
    product : str
        GPM product acronym. See gpm_api.available_products()
    version : int
        GPM version of the data to retrieve if product_type = 'RS'.
        The default is 7.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    """
    # Retrieve the local GPM base directory
    base_dir = get_gpm_base_dir()
    base_dir = check_base_dir(base_dir)

    # Retrieve the local directory where the data are stored
    product_dir = _get_local_product_base_directory(
        base_dir=base_dir,
        product=product,
        product_type=product_type,
        version=version,
    )

    # Check if the folder exists
    if not os.path.exists(product_dir):
        return []

    # Retrieve the filepaths
    filepaths = list_files(product_dir, glob_pattern="*", recursive=True)
    return filepaths
