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
import os
import re

from gpm.configs import get_base_dir
from gpm.io.filter import filter_filepaths
from gpm.io.products import get_product_category
from gpm.utils.directories import search_leaf_files

####--------------------------------------------------------------------------.
#####################
#### Directories ####
#####################


def get_time_tree(date):
    """Get time tree path ``<YYYY>/<MM>/<DD>``."""
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    return os.path.join(year, month, day)


def _get_local_dir_pattern(product, product_type, version):
    """Defines the local (disk) repository base pattern where data are stored and searched.

    Parameters
    ----------
    product : str
        GPM product name. See ``gpm.available_products()``.
    product_type : str
        GPM product type. Either ``RS`` (Research) or ``NRT`` (Near-Real-Time).
    version : int
        GPM version of the data to retrieve if ``product_type = "RS"``.

    Returns
    -------
    pattern : str
        Directory base pattern:

        - If ``product_type == "RS"``: ``RS/V<version>/<product_category>/<product>``
        - If ``product_type == "NRT"``: ``NRT/<product_category>/<product>``

        Valid `product_category` are ``RADAR``, ``PMW``, ``CMB``, ``IMERG``.

    """
    # Define pattern
    product_category = get_product_category(product)
    if product_type == "NRT":
        dir_structure = os.path.join(product_type, product_category, product)
    else:  # product_type == "RS"
        version_str = "V0" + str(int(version))
        dir_structure = os.path.join(product_type, version_str, product_category, product)
    return dir_structure


def _get_local_product_base_directory(base_dir, product, product_type, version):
    """Provide the local product base directory path where the requested GPM data are stored.

    Parameters
    ----------
    base_dir : str
        The base directory where to store GPM data.
    product : str
        GPM product name. See ``gpm.available_products()``.
    product_type : str
        GPM product type. Either ``RS`` (Research) or ``NRT`` (Near-Real-Time).
    version : int
        GPM version of the data to retrieve if ``product_type = "RS"``.

    Returns
    -------
    product_dir : str
        Product base directory path where data are located.

    """
    product_dir_pattern = _get_local_dir_pattern(product, product_type, version)
    return os.path.join(base_dir, product_dir_pattern)


def _get_local_directory_tree(product, product_type, date, version):
    """Return the local product directory tree.

    The directory tree structure for ``product_type``:

    - ``RS`` is ``GPM/RS/V<version>/<product_category>/<product>/YYYY/MM/YY``
    - ``NRT`` is ``GPM/NRT/<product_category>/<product>/YYYY/MM/YY``

    Parameters
    ----------
    product : str
        GPM product name. See ``gpm.available_products()``.
    product_type : str
        GPM product type. Either ``RS`` (Research) or ``NRT`` (Near-Real-Time).
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
    return os.path.join(product_dir_tree, time_tree)


def get_local_product_directory(base_dir, product, product_type, version, date):
    """Provide the local repository path where the requested daily GPM data are stored/need to be saved.

    Parameters
    ----------
    base_dir : str
        The base directory where to store GPM data.
    product : str
        GPM product name. See ``gpm.available_products()``.
    product_type : str
        GPM product type. Either ``RS`` (Research) or ``NRT`` (Near-Real-Time).
    version : int
        GPM version of the data to retrieve if ``product_type = "RS"``.
    date : datetime.date
        Single date for which to retrieve the data.

    Returns
    -------
    product_dir_path : str
        Directory path where daily GPM data are located.

        - If ``product_type == "RS"``: ``<base_dir>/GPM/RS/V0<version>/<product_category>/<product>/<YYYY>/<MM>/<DD>``
        - If ``product_type == "NRT"``: ``<base_dir>/GPM/NRT/<product_category>/<product>/<YYYY>/<MM>/<DD>``

        Valid `product_category` are ``RADAR``, ``PMW``, ``CMB``, ``IMERG``.

    """
    dir_structure = _get_local_directory_tree(
        product=product,
        product_type=product_type,
        version=version,
        date=date,
    )
    return os.path.join(base_dir, dir_structure)


####--------------------------------------------------------------------------.
############################
#### Filepath retrieval ####
############################


def get_local_daily_filepaths(product, product_type, date, version, base_dir):
    """Retrieve GPM data filepaths on the local disk directory of a specific day and product.

    Parameters
    ----------
    product : str
        GPM product acronym. See ``gpm.available_products()``.
    product_type : str
        GPM product type. Either ``RS`` (Research) or ``NRT`` (Near-Real-Time).
    date : datetime.date
        Single date for which to retrieve the data.
    version : int
        GPM version of the data to retrieve if ``product_type = "RS"``.
    base_dir : str
        The base directory where to store GPM data.
    """
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
    return [os.path.join(dir_path, filename) for filename in filenames]


def define_local_filepath(product, product_type, date, version, filename, base_dir=None):
    """Define local file path.

    This function is called by get_filepath_from_filename(filename, storage, product_type).
    """
    # Retrieve the local GPM base directory
    base_dir = get_base_dir(base_dir=base_dir)

    # Define disk directory path
    dir_tree = get_local_product_directory(
        base_dir=base_dir,
        product=product,
        product_type=product_type,
        date=date,
        version=version,
    )
    # Define disk file path
    return os.path.join(dir_tree, filename)


def get_local_dir_tree_from_filename(filepath, product_type="RS", base_dir=None):
    """Return directory tree from a GPM filename or filepath."""
    from gpm.io.info import get_info_from_filepath

    base_dir = get_base_dir(base_dir=base_dir)

    # Retrieve file info
    info = get_info_from_filepath(filepath)
    product = info["product"]
    version = int(re.findall("\\d+", info["version"])[0])
    date = info["start_time"].date()
    # Retrieve directory tree
    return get_local_product_directory(
        base_dir=base_dir,
        product=product,
        product_type=product_type,
        date=date,
        version=version,
    )


def get_local_filepath_from_filename(filepath, product_type="RS", base_dir=None):
    """Return the local filepath of a GPM file or filepath."""
    filename = os.path.basename(filepath)
    dir_tree = get_local_dir_tree_from_filename(
        filepath,
        product_type=product_type,
        base_dir=base_dir,
    )
    return os.path.join(dir_tree, filename)


####--------------------------------------------------------------------------.
#################
#### Utility ####
#################


def get_local_filepaths(
    product,
    version=7,
    product_type="RS",
    start_time=None,
    end_time=None,
    base_dir=None,
    groups=None,
):
    """Retrieve all GPM filepaths on the local disk directory for a specific product.

    Parameters
    ----------
    product : str
        GPM product acronym. See ``gpm.available_products()``.
    version : int
        GPM version of the data to retrieve if ``product_type = "RS"``.
        The default is version ``7``.
    product_type : str, optional
        GPM product type. Either ``RS`` (Research) or ``NRT`` (Near-Real-Time).
    groups: list or str, optional
        Whether to group the filepaths in a dictionary by a custom selection of keys.
        Valid group keys are `product_level`, `satellite`, `sensor`, `algorithm`, `start_time`, `end_time`,
        `granule_id`, `version`, `product_type`, `product`, `data_format`,
        `year`, `month`, `day`,  `doy`, `dow`, `hour`, `minute`, `second`, `month_name`, `quarter`, `season`.
        The time components are extracted from `start_time` !
        If groups is ``None`` returns the filepaths list.
        The default is ``None``.

    """
    from gpm.io.info import group_filepaths

    # Retrieve the local GPM base directory
    base_dir = get_base_dir(base_dir=base_dir)

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
    filepaths = search_leaf_files(base_dir=product_dir, parallel=True)

    # Filter files by start_time and end_time
    if start_time is not None or end_time is not None:
        filepaths = filter_filepaths(filepaths, start_time=start_time, end_time=end_time)

    # Sort the filepaths
    filepaths = sorted(filepaths)

    # Group filepaths if groups is not None
    return group_filepaths(filepaths, groups=groups)
