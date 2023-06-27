#!/usr/bin/env python3
"""
Created on Mon Aug 15 00:18:33 2022

@author: ghiggi
"""

import concurrent.futures
import datetime
import ftplib
import os
import subprocess
import warnings

import pandas as pd
from tqdm import tqdm

from gpm_api.configs import get_gpm_base_dir, get_gpm_password, get_gpm_username
from gpm_api.io import GPM_VERSION  # CURRENT GPM VERSION
from gpm_api.io.checks import (
    check_base_dir,
    check_date,
    check_product,
    check_product_type,
    check_start_end_time,
    check_version,
    is_empty,
)
from gpm_api.io.directories import get_disk_directory, get_pps_directory
from gpm_api.io.info import get_info_from_filepath, get_start_time_from_filepaths
from gpm_api.io.pps import _find_pps_daily_filepaths  # , find_pps_filepaths
from gpm_api.utils.archive import (
    check_file_integrity,
    get_corrupted_filepaths,
    remove_corrupted_filepaths,
)
from gpm_api.utils.utils_string import subset_list_by_boolean
from gpm_api.utils.warnings import GPMDownloadWarning

### Currently we open a connection for every file
### We might want to launch wget in parallel directly
### - And then loop by day or month to display progress

## For https connection, it requires Authorization header: <type><credentials>
# - type: "Basic"
# - credientials: <...>
# --> "--header='Authorization: Basic Z2lvbmF0YS5naGlnZ2lAZXBmbC5jaDpnaW9uYXRhLmdoaWdnaUBlcGZsLmNo' "

##----------------------------------------------------------------------------.
#############################
#### Single file command ####
#############################
def curl_cmd(server_path, disk_path, username, password):
    """Download data using curl via ftps."""
    # -------------------------------------------------------------------------.
    # Check disk directory exists (if not, create)
    disk_dir = os.path.dirname(disk_path)
    if not os.path.exists(disk_dir):
        os.makedirs(disk_dir)
    # -------------------------------------------------------------------------.
    # Replace ftps with ftp to make curl work !!!
    # - curl expects ftp:// and not ftps://
    server_path = server_path.replace("ftps", "ftp", 1)
    # -------------------------------------------------------------------------.
    ## Define command to run
    # Base command: curl -4 --ftp-ssl --user [user name]:[password] -n [url]
    # - -4: handle IPV6 connections
    # - v : verbose
    # --fail : fail silently on server errors. Allow to deal better with failed attemps
    #           Return error > 0 when the request fails
    # --silent: hides the progress and error
    # --retry 10: retry 10 times
    # --retry-delay 5: with 5 secs delays
    # --retry-max-time 60*10: total time before it's considered failed
    # --connect-timeout 20: limits time curl spend trying to connect ot the host to 20 secs
    # --get url: specify the url
    # -o : write to file instead of stdout
    # Important note
    # - -k (or --insecure) is required with curl > 7.71 otherwise unauthorized access
    #
    cmd = "".join(
        [
            "curl ",
            "--verbose ",
            "--ipv4 ",
            "--insecure ",
            "--user ",
            username,
            ":",
            password,
            " ",
            "--ftp-ssl ",
            # Custom settings
            "--header 'Connection: close' ",
            "--connect-timeout 20 ",
            "--retry 5 ",
            "--retry-delay 10 ",
            "-n ",
            server_path,
            " ",
            "-o ",
            disk_path,
        ]
    )
    return cmd


def wget_cmd(server_path, disk_path, username, password):
    """Create wget command to download data via ftps."""
    # -------------------------------------------------------------------------.
    # Check disk directory exists (if not, create)
    disk_dir = os.path.dirname(disk_path)
    if not os.path.exists(disk_dir):
        os.makedirs(disk_dir)
    # -------------------------------------------------------------------------.
    # Base command: wget -4 --ftp-user=[user name] –-ftp-password=[password] -O
    ## Define command to run
    cmd = "".join(
        [
            "wget ",
            "-4 ",
            "--ftp-user=",
            username,
            " ",
            "--ftp-password=",
            password,
            " ",
            "-e robots=off ",  # allow wget to work ignoring robots.txt file
            "-np ",  # prevents files from parent directories from being downloaded
            "-R .html,.tmp ",  # comma-separated list of rejected extensions
            "-nH ",  # don't create host directories
            "-c ",  # continue from where it left
            "--read-timeout=",
            "10",
            " ",  # if no data arriving for 10 seconds, retry
            "--tries=",
            "5",
            " ",  # retry 5 times (0 forever)
            "-O ",
            disk_path,
            " ",
            server_path,
        ]
    )
    # -------------------------------------------------------------------------.
    return cmd


def run(commands, n_threads=10, progress_bar=True, verbose=True):
    """
    Run bash commands in parallel using multithreading.
    Parameters
    ----------
    commands : list
        list of commands to execute in the terminal.
    n_threads : int, optional
        Number of parallel download. The default is 10.

    Returns
    -------
    List of commands which didn't complete.
    """
    if n_threads < 1:
        n_threads = 1
    n_threads = min(n_threads, 10)
    n_cmds = len(commands)
    ##------------------------------------------------------------------------.
    # Run with progress bar
    if progress_bar is True:
        with tqdm(total=n_cmds) as pbar, concurrent.futures.ThreadPoolExecutor(
            max_workers=n_threads
        ) as executor:
            dict_futures = {
                executor.submit(
                    subprocess.check_call,
                    cmd,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ): cmd
                for cmd in commands
            }
            # List cmds that didn't work
            l_cmd_error = []
            for future in concurrent.futures.as_completed(dict_futures.keys()):
                pbar.update(1)  # Update the progress bar
                # Collect all commands that caused problems
                if future.exception() is not None:
                    l_cmd_error.append(dict_futures[future])
    ##------------------------------------------------------------------------.
    # Run without progress bar
    else:
        if (n_threads == 1) and (verbose is True):
            # print("Here")
            # print(commands)
            _ = [subprocess.run(cmd, shell=True) for cmd in commands]
            l_cmd_error = []
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
                # Run commands and list those didn't work
                dict_futures = {
                    executor.submit(
                        subprocess.check_call,
                        cmd,
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    ): cmd
                    for cmd in commands
                }
                # List cmds that didn't work
                l_cmd_error = []
                for future in concurrent.futures.as_completed(dict_futures.keys()):
                    # Collect all commands that caused problems
                    if future.exception() is not None:
                        l_cmd_error.append(dict_futures[future])

    return l_cmd_error


def _download_with_ftlib(server_path, disk_path, username, password):
    # Infer hostname
    hostname = server_path.split("/", 3)[2]  # remove ftps:// and select host

    # Remove hostname from server_path
    server_path = server_path.split("/", 3)[3]

    # Connect to the FTP server using FTPS
    ftps = ftplib.FTP_TLS(hostname)

    # Login to the FTP server using the provided username and password
    ftps.login(username, password)  # /gpmdata base directory

    # Download the file from the FTP server
    try:
        with open(disk_path, "wb") as file:
            ftps.retrbinary(f"RETR {server_path}", file.write)
    except EOFError:
        return f"Impossible to download {server_path}"

    # Close the FTP connection
    ftps.close()
    return None


def ftplib_download(server_paths, disk_paths, username, password, n_threads=10):
    # Download file concurrently
    n_files = len(server_paths)
    with tqdm(total=n_files) as pbar, concurrent.futures.ThreadPoolExecutor(
        max_workers=n_threads
    ) as executor:
        dict_futures = {
            executor.submit(
                _download_with_ftlib,
                server_path=server_path,
                disk_path=disk_path,
                username=username,
                password=password,
            ): server_path
            for server_path, disk_path in zip(server_paths, disk_paths)
        }
        # List cmds that didn't work
        l_cmd_error = []
        for future in concurrent.futures.as_completed(dict_futures.keys()):
            pbar.update(1)  # Update the progress bar
            # Collect all commands that caused problems
            if future.exception() is not None:
                l_cmd_error.append(dict_futures[future])


def _download_files(
    src_fpaths,
    dst_fpaths,
    username,
    password,
    transfer_tool="wget",
    n_threads=4,
    progress_bar=True,
    verbose=False,
):

    if transfer_tool == "curl":
        list_cmd = [
            curl_cmd(src_path, dst_path, username, username)
            for src_path, dst_path in zip(src_fpaths, dst_fpaths)
        ]
    elif transfer_tool == "wget":
        list_cmd = [
            wget_cmd(src_path, dst_path, username, username)
            for src_path, dst_path in zip(src_fpaths, dst_fpaths)
        ]
    else:
        raise NotImplementedError("Download is available with 'wget' or 'curl'.")
    # -------------------------------------------------------------------------.
    ## Download the data (in parallel)
    bad_cmds = run(list_cmd, n_threads=n_threads, progress_bar=progress_bar, verbose=verbose)
    return bad_cmds


####--------------------------------------------------------------------------.
############################
#### Filtering routines ####
############################


def filter_download_list(server_paths, disk_paths, force_download=False):
    """
    Removes filepaths of GPM file already existing on disk.

    Parameters
    ----------
    server_paths : str
        GPM directory on disk for a specific product and date.
    PPS_filepaths : str
        Filepaths on which GPM data are stored on PPS servers.
    force_download : boolean, optional
        Whether to redownload data if already existing on disk. The default is False.

    Returns
    -------
    server_paths: list
        List of filepaths on the NASA PPS server.
    disk_paths: list
        List of filepaths on the local disk.

    """
    # -------------------------------------------------------------------------.
    # Check if data already exists
    if force_download is False:
        # Get index which do not exist
        idx_not_existing = [not os.path.exists(disk_path) for disk_path in disk_paths]
        # Select filepath not existing on disk
        disk_paths = subset_list_by_boolean(disk_paths, idx_not_existing)
        server_paths = subset_list_by_boolean(server_paths, idx_not_existing)
    return (server_paths, disk_paths)


####--------------------------------------------------------------------------.
###########################
#### Download routines ####
###########################


def _convert_disk_to_pps_fpath(filepath):
    """Convert a disk filepath into a PPS filepath."""
    # Extract file information from filepath
    filename = os.path.basename(filepath)
    info_dict = get_info_from_filepath(filepath)
    start_time = info_dict["start_time"]
    date = start_time.date()
    list_dirs = filepath.split(os.path.sep)
    product = list_dirs[-5]
    version = int(list_dirs[-7][2])
    product_type = list_dirs[-8]
    # base_dir = os.path.join(os.path.sep, *list_dirs[0:-8])

    # Retrieve PPS filepath
    url_data_server, url_file_list = get_pps_directory(
        product=product, product_type=product_type, date=date, version=version
    )
    dir_tree = os.path.join(*url_file_list.split(os.path.sep)[4:])
    pps_filepath = os.path.join(url_data_server, dir_tree, filename)
    return pps_filepath


def convert_disk_to_pps_filepaths(filepaths):
    """
    Convert GPM filename or filepaths to PPS filepaths.

    Parameters
    ----------
    filepaths : list
        GPM file paths on disk.

    Returns
    -------
    pps_filepaths : list
        List of file paths on PPS.

    """
    pps_fpaths = [_convert_disk_to_pps_fpath(fpath) for fpath in filepaths]
    return pps_fpaths


def convert_pps_to_disk_filepaths(pps_filepaths, base_dir, product, product_type, version):
    """
    Convert PPS filepaths to local disk filepaths.

    Parameters
    ----------
    pps_filepaths : list
        File paths on the PPS server.
    base_dir : str
        The base directory where to store GPM data.
    product : str
        GPM product acronym. See gpm_api.available_products()
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.

    Returns
    -------
    disk_filepaths : list
        List of filepaths on local disk.

    """
    l_start_time = get_start_time_from_filepaths(pps_filepaths)
    l_dates = [start_time.date() for start_time in l_start_time]
    disk_filepaths = []
    for date, pps_filepath in zip(l_dates, pps_filepaths):
        disk_dir = get_disk_directory(
            base_dir=base_dir,
            product=product,
            product_type=product_type,
            date=date,
            version=version,
        )
        disk_filepath = disk_dir + "/" + os.path.basename(pps_filepath)
        disk_filepaths.append(disk_filepath)
    return disk_filepaths


def _download_daily_data(
    base_dir,
    username,
    password,
    date,
    version,
    product,
    product_type,
    start_time=None,
    end_time=None,
    n_threads=10,
    transfer_tool="curl",
    progress_bar=True,
    force_download=False,
    verbose=True,
    warn_missing_files=True,
):
    """
    Download GPM data from NASA servers using curl or wget.

    Parameters
    ----------
    base_dir : str
        The base directory where to store GPM data.
    username: str
        Email address with which you registered on the NASA PPS.
    password: str
        Password to access the NASA PPS server.
    product : str
        GPM product name. See: gpm_api.available_products()
    date : datetime
        Single date for which to retrieve the data.
    start_time : datetime.datetime
        Filtering start time.
    end_time : datetime.datetime
        Filtering end time.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
        GPM data readers are currently implemented only for GPM V06.
    username : str, optional
        Provide your email for login on GPM NASA servers.
        Temporary default is "gionata.ghiggi@epfl.ch".
    n_threads : int, optional
        Number of parallel downloads. The default is set to 10.
    progress_bar : bool, optional
        Wheter to display progress. The default is True.
    transfer_tool : str, optional
        Wheter to use curl or wget for data download. The default is "curl".
    force_download : boolean, optional
        Whether to redownload data if already existing on disk. The default is False.
    verbose : bool, optional
        Whether to print processing details. The default is True.

    Returns
    -------
    int
        0 if everything went fine.

    """
    # -------------------------------------------------------------------------.
    ## Check input arguments
    date = check_date(date)
    check_product_type(product_type=product_type)
    check_product(product=product, product_type=product_type)

    # -------------------------------------------------------------------------.
    ## Retrieve the list of files available on NASA PPS server
    pps_filepaths = _find_pps_daily_filepaths(
        username=username,
        password=password,
        product=product,
        product_type=product_type,
        version=version,
        date=date,
        start_time=start_time,
        end_time=end_time,
        verbose=verbose,
    )

    disk_filepaths = convert_pps_to_disk_filepaths(
        pps_filepaths=pps_filepaths,
        base_dir=base_dir,
        product=product,
        product_type=product_type,
        version=version,
    )
    # -------------------------------------------------------------------------.
    ## If no file to retrieve on NASA PPS, return None
    if is_empty(pps_filepaths) and warn_missing_files:
        msg = f"No data found on PPS on date {date} for product {product}"
        warnings.warn(msg, GPMDownloadWarning)
        return None

    # -------------------------------------------------------------------------.
    ## If force_download is False, select only data not present on disk
    pps_filepaths, disk_filepaths = filter_download_list(
        disk_paths=disk_filepaths,
        server_paths=pps_filepaths,
        force_download=force_download,
    )
    # -------------------------------------------------------------------------.
    # Retrieve commands
    bad_cmds = _download_files(
        src_fpaths=pps_filepaths,
        dst_fpaths=disk_filepaths,
        username=username,
        password=password,
        n_threads=n_threads,
        progress_bar=progress_bar,
        verbose=verbose,
    )
    return bad_cmds


##-----------------------------------------------------------------------------.
def download_data(
    product,
    start_time,
    end_time,
    product_type="RS",
    version=GPM_VERSION,
    n_threads=10,
    transfer_tool="curl",
    progress_bar=False,
    force_download=False,
    check_integrity=True,
    remove_corrupted=True,
    retry=1,
    verbose=True,
    base_dir=None,
    username=None,
    password=None,
):
    """
    Download GPM data from NASA servers (day by day).

    Parameters
    ----------
    product : str
        GPM product acronym. See gpm_api.available_products()
    start_time : datetime
        Start time.
    end_time : datetime
        End time.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'.
        GPM data readers are currently implemented only for GPM V06.
    n_threads : int, optional
        Number of parallel downloads. The default is set to 10.
    progress_bar : bool, optional
        Wheter to display progress. The default is True.
    transfer_tool : str, optional
        Wheter to use curl or wget for data download. The default is "curl".
    force_download : boolean, optional
        Whether to redownload data if already existing on disk. The default is False.
    verbose : bool, optional
        Whether to print processing details. The default is False.
    check_integrity: bool, optional
        Check integrity of the downloaded files.
        By default is True.
    remove_corrupted: bool, optional
        Whether to remove the corrupted files.
        By default is True.
    retry : int, optional,
        The number of attempts to redownload the corrupted files. The default is 1.
        Only applies if check_integrity is True !
    base_dir : str, optional
        The path to the GPM base directory. If None, it use the one specified
        in the GPM-API config file.
        The default is None.
    username: str, optional
        Email address with which you registered on the NASA PPS.
        If None, it uses the one specified in the GPM-API config file.
        The default is None.
    password: str, optional
        Email address with which you registered on the NASA PPS.
        If None, it uses the one specified in the GPM-API config file.
        The default is None.

    Returns
    -------

    boolean: int
        0 if everything went fine.

    """
    # -------------------------------------------------------------------------.
    # Retrieve GPM-API configs
    base_dir = get_gpm_base_dir(base_dir)
    username = get_gpm_username(username)
    password = get_gpm_password(password)

    # -------------------------------------------------------------------------.
    ## Checks input arguments
    check_product_type(product_type=product_type)
    check_product(product=product, product_type=product_type)
    check_version(version=version)
    base_dir = check_base_dir(base_dir)
    start_time, end_time = check_start_end_time(start_time, end_time)

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
    for i, date in enumerate(dates):
        if i == 0:
            warn_missing_files = False
        else:
            warn_missing_files = True

        _ = _download_daily_data(
            base_dir=base_dir,
            username=username,
            password=password,
            date=date,
            version=version,
            product=product,
            product_type=product_type,
            start_time=start_time,
            end_time=end_time,
            n_threads=n_threads,
            transfer_tool=transfer_tool,
            progress_bar=progress_bar,
            force_download=force_download,
            verbose=verbose,
            warn_missing_files=warn_missing_files,
        )

    # -------------------------------------------------------------------------.
    if verbose:
        print(f"The available GPM {product} product files are on disk.")
    if check_integrity:
        l_corrupted = check_file_integrity(
            base_dir=base_dir,
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
            l_corrupted = redownload_from_filepaths(
                filepaths=l_corrupted,
                username=username,
                n_threads=n_threads,
                transfer_tool=transfer_tool,
                progress_bar=progress_bar,
                verbose=verbose,
                retry=retry,
            )
            if verbose:
                if len(l_corrupted) == 0:
                    print("All corrupted files have been redownloaded successively.")
                else:
                    print("Some corrupted files couldn't been redownloaded.")
                    print("Returning the list of corrupted files.")
                    return l_corrupted
    return None


####--------------------------------------------------------------------------.
def redownload_from_filepaths(
    filepaths,
    n_threads=4,
    transfer_tool="wget",
    progress_bar=False,
    verbose=True,
    retry=1,
    username=None,
    password=None,
):
    """
    Redownload GPM files from the PPS archive.

    Parameters
    ----------
    filepaths : list
        List of disk filepaths.
    n_threads : int, optional
        Number of parallel downloads. The default is set to 10.
    progress_bar : bool, optional
        Wheter to display progress. The default is True.
    transfer_tool : str, optional
        Wheter to use curl or wget for data download. The default is "curl".
    verbose : bool, optional
        Whether to print processing details. The default is False.
    retry : int, optional,
        The number of attempts to redownload the corrupted files. The default is 1.
    username: str, optional
        Email address with which you registered on the NASA PPS.
        If None, it uses the one specified in the GPM-API config file.
        The default is None.
    password: str, optional
        Email address with which you registered on the NASA PPS.
        If None, it uses the one specified in the GPM-API config file.
        The default is None.

    Returns
    -------
    l_corrupted : list
        List of remaining corrupted file paths.
    """
    if isinstance(filepaths, type(None)):
        return None
    if not isinstance(filepaths, list):
        raise TypeError("Expecting a list of filepaths.")
    if len(filepaths) == 0:
        return None

    # -------------------------------------------------------------------------.
    # Retrieve GPM-API configs
    username = get_gpm_username(username)
    password = get_gpm_password(password)

    # -------------------------------------------------------------------------.
    # Attempt to redownload the corrupted files
    if verbose:
        n_files = len(filepaths)
        print(f"Attempt to redownload {n_files}.")

    # Retrieve the PPS file paths
    pps_filepaths = convert_disk_to_pps_filepaths(filepaths)

    # Download files
    _ = _download_files(
        src_fpaths=pps_filepaths,
        dst_fpaths=filepaths,
        username=username,
        password=password,
        n_threads=n_threads,
        progress_bar=progress_bar,
        verbose=verbose,
    )

    # Get corrupted filepaths
    l_corrupted = get_corrupted_filepaths(filepaths)

    # Remove corrupted filepaths
    remove_corrupted_filepaths(filepaths=l_corrupted, verbose=verbose)

    # Retry download if retry > 1 as input argument
    if len(l_corrupted) > 0 and retry > 1:
        l_corrupted = redownload_from_filepaths(
            filepaths=l_corrupted,
            username=username,
            n_threads=n_threads,
            transfer_tool=transfer_tool,
            progress_bar=progress_bar,
            verbose=verbose,
            retry=retry - 1,
        )

    return l_corrupted
