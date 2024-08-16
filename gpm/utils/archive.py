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
"""This module contains utilities for GPM Data Archiving."""
import warnings

import numpy as np

from gpm.io.checks import check_start_end_time
from gpm.io.find import find_filepaths
from gpm.io.info import (
    get_end_time_from_filepaths,
    get_granule_from_filepaths,
    get_start_time_from_filepaths,
)
from gpm.utils.warnings import GPM_Warning

####--------------------------------------------------------------------------.
###########################
#### Data completeness ####
###########################
# TODO: move to io/archiving.py in future


def check_no_duplicated_files(
    product,
    start_time,
    end_time,
    version=None,
    product_type="RS",
    verbose=True,
):
    """Check that there are not duplicated files based on granule number."""
    ##--------------------------------------------------------------------.
    # Find filepaths
    filepaths = find_filepaths(
        storage="LOCAL",
        version=version,
        product=product,
        product_type=product_type,
        start_time=start_time,
        end_time=end_time,
        verbose=verbose,
    )
    ##---------------------------------------------------------------------.
    # Check that files have been downloaded  on disk
    if len(filepaths) == 0:
        raise ValueError("No files found on disk. Please download them before.")
    ##---------------------------------------------------------------------.
    # Retrieve granule id from filename
    filepaths = np.array(filepaths)
    granule_ids = get_granule_from_filepaths(filepaths)

    # Count granule ids occurrence
    ids, counts = np.unique(granule_ids, return_counts=True)

    # Get duplicated indices
    idx_ids_duplicated = np.where(counts > 1)[0].flatten()
    n_duplicated = len(idx_ids_duplicated)
    if n_duplicated > 0:
        duplicated_ids = ids[idx_ids_duplicated]
        for granule_id in duplicated_ids:
            idx_paths_duplicated = np.where(granule_id == granule_ids)[0].flatten()
            tmp_paths_duplicated = filepaths[idx_paths_duplicated].tolist()
            print(f"Granule {granule_id} has duplicated filepaths:")
            for path in tmp_paths_duplicated:
                print(f"- {path}")
        raise ValueError("There are {n_duplicated} duplicated granules.")


def check_time_period_coverage(filepaths, start_time, end_time, raise_error=False):
    """Check time period start_time, end_time is covered.

    If raise_error=True, raise error if time period is not covered.
    If raise_error=False, it raise a GPM warning.

    """
    # Check valid start/end time
    start_time, end_time = check_start_end_time(start_time, end_time)

    # Get first and last timestep from filepaths
    filepaths = sorted(filepaths)
    first_start = get_start_time_from_filepaths(filepaths[0])[0]
    last_end = get_end_time_from_filepaths(filepaths[-1])[0]
    # Check time period is covered
    msg = ""
    if first_start > start_time:
        msg = f"The first file start_time ({first_start}) occurs after the specified start_time ({start_time})"

    if last_end < end_time:
        msg1 = f"The last file end_time ({last_end}) occurs before the specified end_time ({end_time})"
        msg = msg + "; and t" + msg1[1:] if msg != "" else msg1
    if msg != "":
        if raise_error:
            raise ValueError(msg)
        warnings.warn(msg, GPM_Warning, stacklevel=1)


def get_time_period_with_missing_files(filepaths):
    """It returns the time period where the are missing granules.

    It assumes the input filepaths are for a single GPM product.

    Parameters
    ----------
    filepaths : list
        List of GPM file paths.

    Returns
    -------
    list_missing : list
        List of tuple (start_time, end_time).

    """
    from gpm.utils.checks import _is_contiguous_granule
    from gpm.utils.slices import get_list_slices_from_bool_arr

    # Retrieve granule id from filename
    granule_ids = get_granule_from_filepaths(filepaths)

    # Sort filepaths by granule number
    indices = np.argsort(granule_ids)
    filepaths = np.array(filepaths)[indices]
    granule_ids = np.array(granule_ids)[indices]

    # Check if next file granule number is +1
    is_not_missing = _is_contiguous_granule(granule_ids)

    # If there are missing files
    list_missing = []
    if np.any(~is_not_missing):
        # Retrieve slices with unmissing granules
        # - Do not skip consecutive False
        # --> is_not_missing=np.array([False, False, True, True, False, False])
        # --> list_slices = [slice(0, 1, None), slice(1, 2, None), slice(2, 5, None), slice(5, 6, None)]
        list_slices = get_list_slices_from_bool_arr(
            is_not_missing,
            include_false=True,
            skip_consecutive_false=False,
        )
        # Retrieve start and end_time where there are missing files
        for slc in list_slices[0:-1]:
            missing_start = get_end_time_from_filepaths(filepaths[slc.stop - 1])[0]
            missing_end = get_start_time_from_filepaths(filepaths[slc.stop])[0]
            list_missing.append((missing_start, missing_end))
    return list_missing


def check_archive_completeness(
    product,
    start_time,
    end_time,
    version=None,
    product_type="RS",
    download=True,
    transfer_tool="WGET",
    n_threads=4,
    verbose=True,
):
    """Check that the GPM product archive is not missing granules over a given period.

    This function does not require connection to the PPS to search for the missing files.
    However, the start and end period are based on the first and last file found on disk !

    If download=True, it attempt to download the missing granules.

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
    download : bool, optional
        Whether to download the missing files.
        The default is ``True``.
    n_threads : int, optional
        Number of parallel downloads. The default is set to 10.
    transfer_tool : str, optional
        Whether to use ``curl`` or ``wget`` for data download. The default is  ``curl``.
    verbose : bool, optional
        Whether to print processing details. The default is ``False``.

    """
    ##--------------------------------------------------------------------.
    from gpm.io.download import download_archive

    # -------------------------------------------------------------------------.
    # Check valid start/end time
    start_time, end_time = check_start_end_time(start_time, end_time)

    ##--------------------------------------------------------------------.
    # Find filepaths
    filepaths = find_filepaths(
        storage="LOCAL",
        version=version,
        product=product,
        product_type=product_type,
        start_time=start_time,
        end_time=end_time,
        verbose=verbose,
    )
    ##---------------------------------------------------------------------.
    # Check that files have been downloaded on disk
    if len(filepaths) == 0:
        raise ValueError("No files found on disk. Please download them before.")

    ##---------------------------------------------------------------------.
    # Check that the specified time period is covered
    check_time_period_coverage(filepaths, start_time, end_time, raise_error=False)

    ##---------------------------------------------------------------------.
    # Loop over files and retrieve time period with missing granules
    list_missing_periods = get_time_period_with_missing_files(filepaths)

    # If there are missing data,
    if len(list_missing_periods) > 0:
        if download:  # and download=True
            # Attempt to download the missing data
            for s_time, e_time in list_missing_periods:
                download_archive(
                    version=version,
                    product=product,
                    product_type=product_type,
                    start_time=s_time,
                    end_time=e_time,
                    n_threads=n_threads,
                    transfer_tool=transfer_tool,
                    check_integrity=True,
                    remove_corrupted=True,
                    retry=2,
                    verbose=verbose,
                )
        else:
            # Otherwise print time periods with missing data and raise error
            for s_time, e_time in list_missing_periods:
                print(f"- Missing data between {s_time} and {e_time}")
            raise ValueError(
                "The GPM {product} archive is not complete between {start_time} and {end_time}.",
            )


####--------------------------------------------------------------------------.
