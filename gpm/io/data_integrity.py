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
"""This module contains functions that check the GPM files integrity."""
import os

import xarray as xr

from gpm.io.checks import (
    check_product,
    check_start_end_time,
    check_valid_time_request,
)
from gpm.io.find import find_filepaths


def get_corrupted_filepaths(filepaths):
    """Return the file paths of corrupted files."""
    l_corrupted = []
    for filepath in filepaths:
        try:
            # Try open the HDF file

            # DataTree.close() does not work yet!
            # dt = datatree.open_datatree(filepath, engine="netcdf4")
            # dt.close()

            # h5py it's an heavy dependency !
            # hdf = h5py.File(filepath, "r")  # h5py._hl.files.File
            # hdf.close()

            ds = xr.open_dataset(filepath, engine="netcdf4", group="")
            ds.close()

        except OSError:
            l_corrupted.append(filepath)
    return l_corrupted


def remove_corrupted_filepaths(filepaths, verbose=True):
    for filepath in filepaths:
        if verbose:
            print(f"{filepath} is corrupted and is being removed.")
        os.remove(filepath)


def check_filepaths_integrity(filepaths, remove_corrupted=True, verbose=True):
    """Check the integrity of GPM files.

    Parameters
    ----------
    filepaths : list
        List of file paths.
    remove_corrupted : bool, optional
       Whether to remove the corrupted files.
       The default is ``True``.
    verbose : bool, optional
        Whether to verbose the corrupted files. The default is ``True``.

    Returns
    -------
    l_corrupted : list
        List of corrupted file paths.

    """
    # Loop over files and list file that can't be opened
    l_corrupted = get_corrupted_filepaths(filepaths)

    # Report corrupted and remove if asked
    if remove_corrupted:
        remove_corrupted_filepaths(filepaths=l_corrupted, verbose=verbose)
    else:
        for filepath in l_corrupted:
            print(f"{filepath} is corrupted.")

    return l_corrupted


def check_archive_integrity(
    product,
    start_time,
    end_time,
    version=None,
    product_type="RS",
    remove_corrupted=True,
    verbose=True,
):
    """Check GPM granule file integrity over a given period.

    If remove_corrupted=True, it removes the corrupted files.

    Parameters
    ----------
    product : str
        GPM product acronym.
    start_time : datetime.datetime
        Start time.
    end_time : datetime.datetime
        End time.
    product_type : str, optional
        GPM product type. Either ``RS`` (Research) or ``NRT`` (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if ``product_type = "RS"``.
        GPM data readers currently support version 4, 5, 6 and 7.
    remove_corrupted : bool, optional
        Whether to remove the corrupted files.
        The default is ``True``.

    Returns
    -------
    filepaths, list
        List of file paths which are corrupted.

    """
    # Check valid product and variables
    product = check_product(product, product_type=product_type)
    # Check valid start/end time
    start_time, end_time = check_start_end_time(start_time, end_time)
    start_time, end_time = check_valid_time_request(start_time, end_time, product=product)
    # Find filepaths
    filepaths = find_filepaths(
        storage="LOCAL",
        version=version,
        product=product,
        product_type=product_type,
        start_time=start_time,
        end_time=end_time,
        verbose=False,
    )

    # Check that files have been downloaded on disk
    if len(filepaths) == 0:
        raise ValueError("No files found on disk. Please download them before.")

    # Check the file integrity
    return check_filepaths_integrity(
        filepaths=filepaths,
        remove_corrupted=remove_corrupted,
        verbose=verbose,
    )
