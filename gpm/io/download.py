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
"""This module contains the routines required to download data from the NASA PPS and GES DISC servers."""
import datetime
import ftplib
import os
import platform
import re
import shlex
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from packaging.version import Version

from gpm.configs import (
    get_password_earthdata,
    get_password_pps,
    get_username_earthdata,
    get_username_pps,
)
from gpm.io.checks import (
    check_date,
    check_product,
    check_product_type,
    check_product_version,
    check_remote_storage,
    check_start_end_time,
    check_transfer_tool,
    check_valid_time_request,
)
from gpm.io.data_integrity import (
    check_archive_integrity,
    check_filepaths_integrity,
)
from gpm.io.find import find_daily_filepaths
from gpm.io.ges_disc import define_ges_disc_filepath
from gpm.io.info import get_info_from_filepath
from gpm.io.local import define_local_filepath
from gpm.io.pps import define_pps_filepath
from gpm.utils.list import flatten_list
from gpm.utils.timing import print_elapsed_time
from gpm.utils.warnings import GPMDownloadWarning

### Notes
# - Currently we open a connection for every file
# --> Maybe we can improve on that (open once, and then ask many stuffs)
# - Is it possible to download entire directories (instead of per-file?)

## For https connection, it requires Authorization header: <type><credentials>
# - type: "Basic"
# - credentials: <...>
# --> "--header='Authorization: Basic Z2lvbmF0YS5naGlnZ2lAZXBmbC5jaDpnaW9uYXRhLmdoaWdnaUBlcGZsLmNo' "


# WGET options
# -e robots=off : allow wget to work ignoring robots.txt file
# -np           : prevents files from parent directories from being downloaded
# -R .html,.tmp : comma-separated list of rejected extensions
# -nH           : don't create host directories
# -c            :continue from where it left
# --read-timeout=10 --tries=5
#               : if no data arriving for 10 seconds, retry 5 times (0 forever)

# CURL options
# -4: handle IPV6 connections
# --fail : fail silently on server errors. Allow to deal better with failed attempts
#           Return error > 0 when the request fails
# -n or --netrc: flag in curl tells the program to use the user's .netrc
# --silent: hides the progress and error
# --retry 10: retry 10 times
# --retry-delay 5: with 5 secs delays
# --retry-max-time 60*10: total time before it's considered failed
# --connect-timeout 20: limits time curl spend trying to connect to the host to 20 secs
# -o : write to file instead of stdout


####--------------------------------------------------------------------------.
#####################################
#### CURL Backward Compatibility ####
#####################################
def get_curl_version():
    """Get curl version."""
    result = subprocess.run(["curl", "--version"], capture_output=True, text=True, check=False)
    # Example output: curl 7.68.0 (x86_64-pc-linux-gnu) ...
    version_match = re.search(r"curl (\d+\.\d+\.\d+)", result.stdout)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Could not determine curl version")


CURL_VERSION = get_curl_version()
CURL_FTPS_FLAG = "--ftp-ssl" if Version(CURL_VERSION) <= Version("7.20.0") else "--ssl"

####--------------------------------------------------------------------------.
####################
#### PPS Checks ####
####################


def check_valid_pps_credentials(verbose=True):
    """Check validity of PPS credentials."""
    ftp_host = "arthurhouftps.pps.eosdis.nasa.gov"
    username, password = _get_storage_username_password("PPS")
    try:
        with ftplib.FTP_TLS(ftp_host) as ftp:
            ftp.login(user=username, passwd=password)
            ftp.prot_p()  # Switch to secure data connection
        if verbose:
            print("The PPS username and password are valid.")
    except Exception as e:
        str_error = str(e)
        if "Login incorrect" in str_error:
            raise ValueError("Invalid PPS username or password.")
        raise ValueError(f"The PPS server {ftp_host} is not available.")


def check_pps_available(verbose=True):
    """Check PPS server is available."""
    check_valid_pps_credentials(verbose=False)
    if verbose:
        print("The PPS server is available.")


def check_pps_ports_are_open():
    """Check 64000-65000 ports are open for PPS."""
    # Check PPS server is online
    check_pps_available(verbose=False)
    # Test port (64000-65000) are open
    username, password = _get_storage_username_password("PPS")
    command = [
        "curl",
        "--ipv4",
        "--insecure",
        "-n",
        "-v",
        "--retry",
        "0",
        "--retry-delay",
        "1",
        "--user",
        f"{username}:{password}",
        "--ssl",
        "--url",
        "ftp://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/README.TXT",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if "Couldn't connect to server" in result.stderr:
            msg = (
                "The ports in the range of 64000-65000 are not open for the the DNS names "
                + "'arthurhouftps.pps.eosdis.nasa.gov' and 'jsimpsonftps.pps.eosdis.nasa.gov'."
                + "Please modify your router and firewall settings for successful communication "
                + "with the PPS data server."
            )
            raise ValueError(msg)
        msg = (
            "An undetermined connection error occurred while attempting to access the PPS. "
            + "The error is: {result.stderr}"
        )
        raise ValueError(msg)
    print("The ports in the range of 64000-65000 are open. You are ready to use GPM-API !")


####--------------------------------------------------------------------------.
##########################
#### Download Utility ####
##########################


def _get_commands_futures(executor, commands):
    """Submit commands and return futures dictionary."""
    # We use shlex to correctly deal with \\ on windows.  Arguments must be entoured by '<argument>'
    return {
        executor.submit(
            subprocess.check_call,
            shlex.split(cmd),
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ): (i, cmd)
        for i, cmd in enumerate(commands)
    }


def _get_list_failing_commands(dict_futures, pbar=None):
    """List commands that failed.

    pbar is a tqdm progress bar.
    """
    l_cmd_error = []
    for future in as_completed(dict_futures.keys()):
        if pbar:
            pbar.update(1)  # Update the progress bar
        # Collect all commands that caused problems
        if future.exception() is not None:
            _, cmd = dict_futures[future]  # {index: cmd}
            l_cmd_error.append(cmd)
    return l_cmd_error


def _get_list_status_commands(dict_futures, pbar=None):
    """Return a list with the status of the command execution.

    A value of 1 means that the command executed successively.
    A value of 0 means that the command execution failed.

    pbar is a tqdm progress bar.
    """
    # TODO maybe here capture (with -1) if the file does not exists !¨
    n_futures = len(dict_futures)
    status = [1] * n_futures
    for future in as_completed(dict_futures.keys()):
        if pbar:
            pbar.update(1)  # Update the progress bar
        index, cmd = dict_futures[future]
        if future.exception() is not None:
            status[index] = 0
    return status


def run(commands, n_threads=10, progress_bar=True, verbose=True):
    """Run bash commands in parallel using multithreading.

    Parameters
    ----------
    commands : list
        list of commands to execute in the terminal.
    n_threads : int, optional
        Number of parallel download. The default is 10.

    Returns
    -------
    status : list
        Download status of each file. 0=Failed. 1=Success.

    """
    from tqdm import tqdm

    if n_threads < 1:
        n_threads = 1
    n_threads = min(n_threads, 10)
    n_cmds = len(commands)

    # Run with progress bar
    if progress_bar:
        with tqdm(total=n_cmds) as pbar, ThreadPoolExecutor(max_workers=n_threads) as executor:
            dict_futures = _get_commands_futures(executor, commands)
            status = _get_list_status_commands(dict_futures, pbar=pbar)

    # Run without progress bar
    elif (n_threads == 1) and (verbose is True):
        results = [subprocess.run(shlex.split(cmd), shell=False, check=False) for cmd in commands]
        status = [result.returncode == 0 for result in results]
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            # Run commands and list those didn't work
            dict_futures = _get_commands_futures(executor, commands)
            status = _get_list_status_commands(dict_futures)

    return status


####--------------------------------------------------------------------------.
#######################################
#### Download Single File Commands ####
#######################################


def curl_pps_cmd(remote_filepath, local_filepath, username, password):
    """CURL command to download data from PPS through ftps."""
    # Check disk directory exists (if not, create)
    local_dir = os.path.dirname(local_filepath)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Replace ftps with ftp to make curl work !!!
    # - curl expects ftp:// and not ftps://
    remote_filepath = remote_filepath.replace("ftps://", "ftp://", 1)

    # Define CURL command
    # - Base cmd: curl -4 --ftp-ssl --user [user name]:[password] -n [url]
    # - With curl > 7.71 the flag -k (or --insecure) is required  to avoid unauthorized access
    # - Define authentication settings
    auth = f"--ipv4 --insecure -n --user '{username}:{password}' {CURL_FTPS_FLAG} --header 'Connection: close'"
    # - Define options
    options = "--connect-timeout 20 --retry 5 --retry-delay 10"  # --verbose
    # - Define command
    return f"curl {auth} {options} --url {remote_filepath} -o '{local_filepath}'"


def curl_ges_disc_cmd(remote_filepath, local_filepath, username="dummy", password="dummy"):  # noqa
    """CURL command to download data from GES DISC."""
    urs_cookies_path = os.path.join(os.path.expanduser("~"), ".urs_cookies")

    # - Define authentication settings
    auth = f"-n -c '{urs_cookies_path}' -b '{urs_cookies_path}' -L"
    # - Define options
    options = "--connect-timeout 20 --retry 5 --retry-delay 10"
    # - Define command
    return f"curl {auth} {options} --url {remote_filepath} -o '{local_filepath}'"


def wget_pps_cmd(remote_filepath, local_filepath, username, password):
    """WGET command to download data from PPS through ftps."""
    # Check disk directory exists (if not, create)
    local_dir = os.path.dirname(local_filepath)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    ## Define WGET command
    # - Base cmd: wget -4 --ftp-user=[user name] --ftp-password=[password] -O
    # - Define authentication settings
    auth = f"-4 --ftp-user='{username}' --ftp-password='{password}' -e robots=off"
    # - Define options
    options = "-np -R .html,.tmp -nH -c --read-timeout=10 --tries=5"
    # - Define command
    return f"wget {auth} {options} -O '{local_filepath}' {remote_filepath}"


def wget_ges_disc_cmd(remote_filepath, local_filepath, username, password="dummy"):  # noqa
    """WGET command to download data from GES DISC."""
    # Define path to EarthData urs_cookies
    urs_cookies_path = os.path.join(os.path.expanduser("~"), ".urs_cookies")

    # Define authentication settings
    auth = f"--load-cookies '{urs_cookies_path}' --save-cookies '{urs_cookies_path}' --keep-session-cookies"

    # Define wget options
    options = "-c --read-timeout=10 --tries=5 -nH -np --content-disposition"

    # Determine the operating system
    os_name = platform.system()

    # Define command
    if os_name == "Windows":
        window_options = f"--user='{username}' --ask-password"
        cmd = f"wget {auth} {options} {window_options} {remote_filepath} -O '{local_filepath}'"
    else:  # os_name in ["Linux", "Darwin"]:  # Darwin is MacOS
        cmd = f"wget {auth} {options} {remote_filepath} -O '{local_filepath}'"
    return cmd


####--------------------------------------------------------------------------.
#### GPM Download Utility


def _get_single_file_cmd_function(transfer_tool, storage):
    """Return command definition function."""
    dict_fun = {
        "PPS": {"WGET": wget_pps_cmd, "CURL": curl_pps_cmd},
        "GES_DISC": {"WGET": wget_ges_disc_cmd, "CURL": curl_ges_disc_cmd},
    }
    return dict_fun[storage][transfer_tool]


def _get_storage_username_password(storage):
    """Retrieve username and password depending on the 'storage'."""
    # Retrieve username and password
    if storage == "PPS":
        username = get_username_pps()
        password = get_password_pps()
    else:
        username = get_username_earthdata()
        password = get_password_earthdata()
    return username, password


def _ensure_local_directories_exists(local_filepaths):
    _ = [os.makedirs(os.path.dirname(path), exist_ok=True) for path in local_filepaths]


def _download_files(
    remote_filepaths,
    local_filepaths,
    storage,
    transfer_tool,
    n_threads=4,
    progress_bar=True,
    verbose=False,
):
    """Download a list of remote files to their GPM-API local file paths.

    This function open a connection to the server for each file to download !.
    """
    transfer_tool = check_transfer_tool(transfer_tool)
    _ensure_local_directories_exists(local_filepaths)

    # Retrieve username and password
    username, password = _get_storage_username_password(storage)

    # Define command list
    get_single_file_cmd = _get_single_file_cmd_function(transfer_tool, storage)
    list_cmd = [
        get_single_file_cmd(remote_filepath, local_filepath, username, password)
        for remote_filepath, local_filepath in zip(remote_filepaths, local_filepaths)
    ]

    ## Download the data (in parallel)
    return run(list_cmd, n_threads=n_threads, progress_bar=progress_bar, verbose=verbose)


####--------------------------------------------------------------------------.
############################
#### Filtering routines ####
############################


def filter_download_list(remote_filepaths, local_filepaths, force_download=False):
    """Removes filepaths of GPM file already existing on disk.

    Parameters
    ----------
    remote_filepaths : str
        GPM directory on disk for a specific product and date.
    remote_filepaths : str
        Filepaths on which GPM data are stored on PPS servers.
    force_download : boolean, optional
        Whether to redownload data if already existing on disk. The default is ``False``.

    Returns
    -------
    remote_filepaths: list
        List of filepaths on the NASA PPS server.
    local_filepaths: list
        List of filepaths on the local disk.

    """
    # -------------------------------------------------------------------------.
    # Check if data already exists
    if force_download is False:
        # Get index of files which does not exist on disk
        idx_not_existing = [i for i, filepath in enumerate(local_filepaths) if not os.path.exists(filepath)]
        # Select paths of files not present on disk
        local_filepaths = [local_filepaths[i] for i in idx_not_existing]
        remote_filepaths = [remote_filepaths[i] for i in idx_not_existing]
    return (remote_filepaths, local_filepaths)


####--------------------------------------------------------------------------.
###########################
#### Filepaths utility ####
###########################


def _get_func_filepath_definition(storage):
    dict_fun = {
        "LOCAL": define_local_filepath,
        "PPS": define_pps_filepath,
        "GES_DISC": define_ges_disc_filepath,
    }
    return dict_fun[storage]


def _define_filepath(
    product,
    product_type,
    date,
    version,
    filename,
    storage,
):
    """Retrieve the filepath based on the filename."""
    return _get_func_filepath_definition(storage)(
        product=product,
        product_type=product_type,
        date=date,
        version=version,
        filename=filename,
    )


def get_filepath_from_filename(filename, storage, product_type):
    """Convert GPM file names to the <storage> file path."""
    filename = os.path.basename(filename)
    info = get_info_from_filepath(filename)
    product = info["product"]
    version = int(re.findall("\\d+", info["version"])[0])
    date = info["start_time"].date()
    # Retrieve filepath
    return _define_filepath(
        product=product,
        product_type=product_type,
        date=date,
        version=version,
        filename=filename,
        storage=storage,
    )


def get_filepaths_from_filenames(filepaths, storage, product_type):
    """Convert GPM file names or file paths to <storage> file paths.

    Parameters
    ----------
    filepaths : list
        GPM file names or file paths.

    Returns
    -------
    filepaths : list
        List of file paths on <storage> storage.

    """
    return [get_filepath_from_filename(filepath, storage=storage, product_type=product_type) for filepath in filepaths]


####--------------------------------------------------------------------------.
###############################
#### Download by filenames ####
###############################


def download_files(
    filepaths,
    product_type="RS",
    storage="PPS",
    n_threads=4,
    transfer_tool="CURL",
    force_download=False,
    remove_corrupted=True,
    progress_bar=False,
    verbose=True,
    retry=1,
):
    """Download specific GPM files from NASA servers.

    Parameters
    ----------
    filepaths: (str or list)
        List of GPM file names to download.
    product_type : str, optional
        GPM product type. Either ``RS`` (Research) or ``NRT`` (Near-Real-Time).
        The default is "RS".
    storage : str, optional
        The remote repository from where to download.
        Either ``pps`` or ``ges_disc``. The default is "PPS".
    n_threads : int, optional
        Number of parallel downloads. The default is set to 10.
    progress_bar : bool, optional
        Whether to display progress. The default is ``True``.
    transfer_tool : str, optional
        Whether to use ``curl`` or ``wget`` for data download. The default is  ``curl``.
    verbose : bool, optional
        Whether to print processing details. The default is ``False``.
    force_download : boolean, optional
        Whether to redownload data if already existing on disk. The default is ``False``.
    remove_corrupted : boolean, optional
       Whether to remove the corrupted files.
       By default is ``True``.
    retry : int, optional,
        The number of attempts to redownload the corrupted files. The default is 1.

    Returns
    -------
    l_corrupted : list
        List of corrupted file paths.
        If no corrupted files, returns an empty list.

    """
    # TODO list
    # - providing inexisting file names currently behave as if the downloaded file was corrupted
    # - we should provide better error messages

    # Check inputs
    storage = check_remote_storage(storage)
    if isinstance(filepaths, type(None)):
        return None
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    if not isinstance(filepaths, list):
        raise TypeError("Expecting a list of file paths.")
    if len(filepaths) == 0:
        return None

    # Print the number of files to download
    if verbose:
        n_files = len(filepaths)
        print(f"Attempt to download {n_files} files.")

    # Retrieve the remote and local file paths
    remote_filepaths = get_filepaths_from_filenames(
        filepaths,
        storage=storage,
        product_type=product_type,
    )
    local_filepaths = get_filepaths_from_filenames(
        filepaths,
        storage="LOCAL",
        product_type=product_type,
    )

    # If force_download is False, select only data not present on disk
    new_remote_filepaths, new_local_filepaths = filter_download_list(
        local_filepaths=local_filepaths,
        remote_filepaths=remote_filepaths,
        force_download=force_download,
    )
    if len(new_remote_filepaths) == 0:
        if verbose:
            print(f"The requested files are already on disk at {local_filepaths}.")
        return None

    # Download files
    _ = _download_files(
        remote_filepaths=new_remote_filepaths,
        local_filepaths=new_local_filepaths,
        storage=storage,
        transfer_tool=transfer_tool,
        n_threads=n_threads,
        progress_bar=progress_bar,
        verbose=verbose,
    )

    # Get corrupted (and optionally removed) filepaths
    l_corrupted = check_filepaths_integrity(
        filepaths=new_local_filepaths,
        remove_corrupted=remove_corrupted,
        verbose=verbose,
    )

    # Retry download if retry > 1 as input argument
    if len(l_corrupted) > 0 and retry > 1:
        l_corrupted = download_files(
            filepaths=l_corrupted,
            product_type=product_type,
            force_download=force_download,
            n_threads=n_threads,
            transfer_tool=transfer_tool,
            progress_bar=progress_bar,
            verbose=verbose,
            retry=retry - 1,
        )
    elif verbose:
        print(f"The requested files are now available on disk at {new_local_filepaths}.")

    return l_corrupted


####--------------------------------------------------------------------------.
###########################
#### Download routines ####
###########################


def _ensure_files_completness(
    filepaths,
    product_type,
    remove_corrupted,
    verbose,
    transfer_tool,
    retry,
    n_threads,
    progress_bar,
):
    """Check file validity and attempt download if corrupted."""
    l_corrupted = check_filepaths_integrity(
        filepaths=filepaths,
        remove_corrupted=remove_corrupted,
        verbose=verbose,
    )
    if verbose:
        print("Integrity checking of GPM files has completed.")
    if retry > 0 and remove_corrupted and len(l_corrupted) > 0:
        if verbose:
            print("Start attempts to redownload the corrupted files.")
        l_corrupted = download_files(
            filepaths=l_corrupted,
            product_type=product_type,
            force_download=True,
            n_threads=n_threads,
            transfer_tool=transfer_tool,
            progress_bar=progress_bar,
            verbose=verbose,
            retry=retry - 1,
        )
        if verbose:
            if len(l_corrupted) == 0:
                print("All corrupted files have been redownloaded successively.")
            else:
                print("Some corrupted files couldn't been redownloaded.")
                print("Returning the list of corrupted files.")
    return l_corrupted


def _ensure_archive_completness(
    product,
    start_time,
    end_time,
    version,
    product_type,
    remove_corrupted,
    verbose,
    transfer_tool,
    retry,
    n_threads,
    progress_bar,
):
    """Check the archive completeness over the specified time period."""
    l_corrupted = check_archive_integrity(
        product=product,
        start_time=start_time,
        end_time=end_time,
        version=version,
        product_type=product_type,
        remove_corrupted=remove_corrupted,
        verbose=verbose,
    )
    if verbose:
        print("Integrity checking of GPM files has completed.")
    if retry > 0 and remove_corrupted and len(l_corrupted) > 0:
        if verbose:
            print("Start attempts to redownload the corrupted files.")
        l_corrupted = download_files(
            filepaths=l_corrupted,
            product_type=product_type,
            force_download=True,
            n_threads=n_threads,
            transfer_tool=transfer_tool,
            progress_bar=progress_bar,
            verbose=verbose,
            retry=retry - 1,
        )
        if verbose:
            if len(l_corrupted) == 0:
                print("All corrupted files have been redownloaded successively.")
            else:
                print("Some corrupted files couldn't been redownloaded.")
                print("Returning the list of corrupted files.")
    return l_corrupted


def _download_daily_data(
    date,
    version,
    product,
    product_type,
    storage,
    transfer_tool,
    n_threads,
    start_time,
    end_time,
    progress_bar,
    force_download,
    verbose,
    warn_missing_files,
):
    """Download GPM data from NASA servers using curl or wget.

    Parameters
    ----------
    product : str
        GPM product name. See ``gpm.available_products()``.
    date : datetime
        Single date for which to retrieve the data.
    start_time : datetime.datetime
        Filtering start time.
    end_time : datetime.datetime
        Filtering end time.
    product_type : str
        GPM product type. Either ``RS`` (Research) or ``NRT`` (Near-Real-Time).
    version : int
        GPM version of the data to retrieve if ``product_type = "RS"``.
    storage : str
        The remote repository from where to download.
        Either ``pps`` or ``ges_disc``.
    n_threads : int
        Number of parallel downloads.
    progress_bar : bool
        Whether to display progress.
    transfer_tool : str
        Whether to use ``curl`` or ``wget`` for data download.
    force_download : boolean
        Whether to redownload data if already existing on disk.
    verbose : bool
        Whether to print processing details. T

    Returns
    -------
    int
        0 if everything went fine.

    """
    # -------------------------------------------------------------------------.
    ## Check input arguments
    date = check_date(date)
    product_type = check_product_type(product_type=product_type)
    product = check_product(product=product, product_type=product_type)
    storage = check_remote_storage(storage)
    transfer_tool = check_transfer_tool(transfer_tool)
    # -------------------------------------------------------------------------.
    ## Retrieve the list of files available on NASA PPS server
    remote_filepaths, available_version = find_daily_filepaths(
        storage=storage,
        product=product,
        product_type=product_type,
        version=version,
        date=date,
        start_time=start_time,
        end_time=end_time,
        verbose=verbose,
    )
    # -------------------------------------------------------------------------.
    ## If no file to retrieve on NASA PPS, return None
    if len(remote_filepaths) == 0:
        if warn_missing_files:
            msg = f"No data found on PPS on date {date} for product {product}"
            warnings.warn(msg, GPMDownloadWarning, stacklevel=2)
        return [], available_version

    # -------------------------------------------------------------------------.
    # Define disk filepaths
    local_filepaths = get_filepaths_from_filenames(
        remote_filepaths,
        storage="LOCAL",
        product_type=product_type,
    )
    # -------------------------------------------------------------------------.
    ## If force_download is False, select only data not present on disk
    remote_filepaths, local_filepaths = filter_download_list(
        local_filepaths=local_filepaths,
        remote_filepaths=remote_filepaths,
        force_download=force_download,
    )
    if len(remote_filepaths) == 0:
        return [-1], available_version  # flag for already on disk

    # -------------------------------------------------------------------------.
    # Retrieve commands
    status = _download_files(
        remote_filepaths=remote_filepaths,
        local_filepaths=local_filepaths,
        storage=storage,
        transfer_tool=transfer_tool,
        n_threads=n_threads,
        progress_bar=progress_bar,
        verbose=verbose,
    )
    return status, available_version


def _check_download_status(status, product, verbose):
    """Check download status.

    Status values:
    - -1 if all daily files are already on disk
    - 0  download failed
    - 1  download success
    Test: status = [-1,-1,-1] --> All on disk
    Test: status = [] --> No data available
    """
    status = np.array(status)
    no_remote_files = len(status) == 0
    all_already_local = np.all(status == -1).item()
    n_remote_files = np.logical_or(status == 0, status == 1).sum().item()
    n_failed = np.sum(status == 0).item()
    n_downloads = np.sum(status == 1).item()

    # - Return None if no files are available for download
    if no_remote_files:
        print("No files are available for download !")
        return None
    # - Pass if all files are already on disk
    if all_already_local:
        if verbose:
            print(f"All the available GPM {product} product files are already on disk.")
        return True
    # - Files available but all download failed
    if n_failed == n_remote_files:
        print(f"{n_failed} files were available, but the download failed !")
        return None
    if verbose:
        if n_failed != 0:
            print(f"The download of {n_failed} files failed.")
        if n_downloads > 0:
            print(f"{n_downloads} files has been download.")
        if n_remote_files == n_downloads:
            print(f"All the available GPM {product} product files are now on disk.")
        else:
            print(
                f"Not all the available GPM {product} product files are on disk. Retry the download !",
            )
    return True


def download_archive(
    product,
    start_time,
    end_time,
    product_type="RS",
    version=None,
    storage="PPS",
    n_threads=4,
    transfer_tool="CURL",
    progress_bar=False,
    force_download=False,
    check_integrity=True,
    remove_corrupted=True,
    retry=1,
    verbose=True,
):
    """Download GPM data from NASA servers (day by day).

    Parameters
    ----------
    product : str
        GPM product acronym. See ``gpm.available_products()``.
    start_time : (datetime.datetime, datetime.date, np.datetime64, str)
        Start time.
        Accepted types: ``datetime.datetime``, ``datetime.date``, ``np.datetime64`` or ``str``.
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.
    end_time : (datetime.datetime, datetime.date, np.datetime64, str)
        End time.
        Accepted types: ``datetime.datetime``, ``datetime.date``, ``np.datetime64`` or ``str``.
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.
    product_type : str, optional
        GPM product type. Either ``RS`` (Research) or ``NRT`` (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if ``product_type = "RS"``.
    storage : str, optional
        The remote repository from where to download.
        Either ``pps`` or ``ges_disc``. The default is ``pps``.
    n_threads : int, optional
        Number of parallel downloads. The default is set to 10.
    progress_bar : bool, optional
        Whether to display progress. The default is ``True``.
    transfer_tool : str, optional
        Whether to use ``curl`` or ``wget`` for data download. The default is  ``curl``.
    force_download : boolean, optional
        Whether to redownload data if already existing on disk. The default is ``False``.
    verbose : bool, optional
        Whether to print processing details. The default is ``False``.
    check_integrity: bool, optional
        Check integrity of the downloaded files.
        By default is ``True``.
    remove_corrupted: bool, optional
        Whether to remove the corrupted files.
        By default is ``True``.
    retry : int, optional,
        The number of attempts to redownload the corrupted files. The default is 1.
        Only applies if ``check_integrity=True``!

    """
    # -------------------------------------------------------------------------.
    ## Checks input arguments
    storage = check_remote_storage(storage)
    product_type = check_product_type(product_type=product_type)
    product = check_product(product=product, product_type=product_type)
    version = check_product_version(version, product)
    transfer_tool = check_transfer_tool(transfer_tool)
    start_time, end_time = check_start_end_time(start_time, end_time)
    start_time, end_time = check_valid_time_request(start_time, end_time, product)
    # -------------------------------------------------------------------------.
    # Retrieve sequence of dates
    # - Specify start_date - 1 day to include data potentially on previous day directory
    # --> Example granules starting at 23:XX:XX in the day before and extending to 01:XX:XX
    start_date = datetime.datetime(start_time.year, start_time.month, start_time.day)
    start_date = start_date - datetime.timedelta(days=1)
    end_date = datetime.datetime(end_time.year, end_time.month, end_time.day)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    dates = list(date_range.to_pydatetime())

    # -------------------------------------------------------------------------.
    # Loop over dates and download the files
    list_status = []
    list_versions = []
    for i, date in enumerate(dates):
        warn_missing_files = not (i == 0 or i == len(dates) - 1 and date == end_time)

        status, available_version = _download_daily_data(
            date=date,
            version=version,
            product=product,
            product_type=product_type,
            start_time=start_time,
            end_time=end_time,
            storage=storage,
            n_threads=n_threads,
            transfer_tool=transfer_tool,
            progress_bar=progress_bar,
            force_download=force_download,
            verbose=verbose,
            warn_missing_files=warn_missing_files,
        )
        list_status += status
        list_versions += available_version

    # -------------------------------------------------------------------------.
    # Check download status
    download_status = _check_download_status(status=list_status, product=product, verbose=verbose)
    if download_status is None:
        return None

    # -------------------------------------------------------------------------.
    # Check downloaded versions
    versions = np.unique(list_versions).tolist()
    if len(versions) > 1:
        msg = f"Multiple GPM {product} product file versions ({versions}) have been download."
        warnings.warn(msg, GPMDownloadWarning, stacklevel=1)

    # -------------------------------------------------------------------------.
    if check_integrity:
        l_corrupted = [
            _ensure_archive_completness(
                product=product,
                start_time=start_time,
                end_time=end_time,
                version=version,
                product_type=product_type,
                remove_corrupted=remove_corrupted,
                verbose=verbose,
                transfer_tool=transfer_tool,
                retry=retry,
                n_threads=n_threads,
                progress_bar=progress_bar,
            )
            for version in versions
        ]
        l_corrupted = flatten_list(l_corrupted)
        if len(l_corrupted) == 0:
            return l_corrupted
    return None


@print_elapsed_time
def download_daily_data(
    product,
    year,
    month,
    day,
    product_type="RS",
    version=None,
    storage="PPS",
    n_threads=10,
    transfer_tool="CURL",
    progress_bar=False,
    force_download=False,
    check_integrity=True,
    remove_corrupted=True,
    verbose=True,
    retry=1,
):
    from gpm.io.download import download_archive

    start_time = datetime.date(year, month, day)
    end_time = start_time + relativedelta(days=1)

    return download_archive(
        product=product,
        start_time=start_time,
        end_time=end_time,
        product_type=product_type,
        version=version,
        storage=storage,
        n_threads=n_threads,
        transfer_tool=transfer_tool,
        progress_bar=progress_bar,
        force_download=force_download,
        check_integrity=check_integrity,
        remove_corrupted=remove_corrupted,
        verbose=verbose,
        retry=retry,
    )


@print_elapsed_time
def download_monthly_data(
    product,
    year,
    month,
    product_type="RS",
    version=None,
    storage="PPS",
    n_threads=10,
    transfer_tool="CURL",
    progress_bar=False,
    force_download=False,
    check_integrity=True,
    remove_corrupted=True,
    verbose=True,
    retry=1,
):
    from gpm.io.download import download_archive

    start_time = datetime.date(year, month, 1)
    end_time = start_time + relativedelta(months=1)

    return download_archive(
        product=product,
        start_time=start_time,
        end_time=end_time,
        product_type=product_type,
        version=version,
        storage=storage,
        n_threads=n_threads,
        transfer_tool=transfer_tool,
        progress_bar=progress_bar,
        force_download=force_download,
        check_integrity=check_integrity,
        remove_corrupted=remove_corrupted,
        verbose=verbose,
        retry=retry,
    )


####--------------------------------------------------------------------------.
