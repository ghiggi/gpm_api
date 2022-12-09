#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 00:18:33 2022

@author: ghiggi
"""

import os
import datetime
import subprocess
import pandas as pd 
import concurrent.futures
from tqdm import tqdm
from gpm_api.io.pps import find_pps_filepaths
from gpm_api.utils.utils_string import subset_list_by_boolean
from gpm_api.utils.archive import check_file_integrity
from gpm_api.io.checks import (
    check_base_dir,
    check_version,
    check_product,
    check_product_type,
    check_start_end_time,
    check_date,
    is_empty,
)
from gpm_api.io.info import get_start_time_from_filepaths
from gpm_api.io.directories import get_disk_directory 


### Currently we open a connection for every file
### We might want to launch wget in parallel directly 
### - And then loop by day or month to display progress

##----------------------------------------------------------------------------.
def curl_cmd(server_path, disk_path, username, password):
    """Download data using curl."""
    #-------------------------------------------------------------------------.
    # Check disk directory exists (if not, create)
    disk_dir = os.path.dirname(disk_path)
    if not os.path.exists(disk_dir):
        os.makedirs(disk_dir)
    #-------------------------------------------------------------------------.
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
    cmd = "".join(["curl ",
                   "--verbose ",
                   "--ipv4 ",
                   "--insecure ", 
                   "--user ", username, ':', password, " ", 
                   "--ftp-ssl ",
                   # TODO: hack to make it temporary work 
                   "--header 'Authorization: Basic Z2lvbmF0YS5naGlnZ2lAZXBmbC5jaDpnaW9uYXRhLmdoaWdnaUBlcGZsLmNo' "
                   # Custom settings
                   "--header 'Connection: close' ",
                   "--connect-timeout 20 ",
                   "--retry 5 ", 
                   "--retry-delay 10 ",
                   '-n ', server_path, " ",
                   "-o ", disk_path])
    return cmd 
    # args = cmd.split()
    # #-------------------------------------------------------------------------.
    # # Execute the command  
    # process = subprocess.Popen(args,
    #                            stdout=subprocess.PIPE,
    #                            stderr=subprocess.PIPE)
    # return process 

def wget_cmd(server_path, disk_path, username, password):
    """Create wget command to download data."""
    server_path = server_path.replace("ftp:", "ftps:")
    #-------------------------------------------------------------------------.
    # Check disk directory exists (if not, create)
    disk_dir = os.path.dirname(disk_path)
    if not os.path.exists(disk_dir):
        os.makedirs(disk_dir)
    #-------------------------------------------------------------------------.
    # Base command: wget -4 --ftp-user=[user name] –-ftp-password=[password] -O
    ## Define command to run
    cmd = "".join(["wget ",
                   "-4 ",
                   "--ftp-user=", username, " ",
                   "--ftp-password=", password, " ",
                   "-e robots=off ", # allow wget to work ignoring robots.txt file 
                   "-np ",           # prevents files from parent directories from being downloaded
                   "-R .html,.tmp ", # comma-separated list of rejected extensions
                   "-nH ",           # don't create host directories
 
                   "-c ",               # continue from where it left 
                   "--read-timeout=", "10", " ", # if no data arriving for 10 seconds, retry
                   "--tries=", "5", " ",         # retry 5 times (0 forever)
                   "-O ", disk_path," ",
                   server_path])
    #-------------------------------------------------------------------------.
    return cmd  

def run(commands, n_threads = 10, progress_bar=True, verbose=True):
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
    if (n_threads < 1):
        n_threads = 1 
    n_threads = min(n_threads, 10) 
    n_cmds = len(commands)
    ##------------------------------------------------------------------------.
    # Run with progress bar
    if progress_bar is True:
        with tqdm(total=n_cmds) as pbar: 
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
                dict_futures = {executor.submit(subprocess.check_call, cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL): cmd for cmd in commands}
                # List cmds that didn't work 
                l_cmd_error = []
                for future in concurrent.futures.as_completed(dict_futures.keys()):
                    pbar.update(1) # Update the progress bar 
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
                dict_futures = {executor.submit(subprocess.check_call, cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL): cmd for cmd in commands}
                # List cmds that didn't work 
                l_cmd_error = []
                for future in concurrent.futures.as_completed(dict_futures.keys()):
                    # Collect all commands that caused problems 
                    if future.exception() is not None:
                        l_cmd_error.append(dict_futures[future])                    

    return l_cmd_error 

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
    #-------------------------------------------------------------------------.    
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


def convert_pps_to_disk_filepaths(pps_filepaths, 
                                  base_dir, 
                                  product,
                                  product_type, 
                                  version):
    """
    Convert PPS filepaths to local disk filepaths.

    Parameters
    ----------
    pps_filepaths : list
        File paths on the PPS server.
    base_dir : str
        The base directory where to store GPM data.
    product : str
        GPM product acronym. See GPM_products()
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
        disk_dir = get_disk_directory(base_dir = base_dir, 
                                      product = product, 
                                      product_type = product_type,
                                      date = date,
                                      version = version)
        disk_filepath = disk_dir + "/" + os.path.basename(pps_filepath)
        disk_filepaths.append(disk_filepath)           
    return disk_filepaths


def _download_daily_data(base_dir,
                        username,
                        product,
                        date,    
                        start_time = None,
                        end_time = None,
                        product_type = 'RS',
                        version = 7,
                        n_threads = 10,
                        transfer_tool = "curl",
                        progress_bar=True, 
                        force_download = False,
                        verbose=True):
    """
    Download GPM data from NASA servers using curl or wget.

    Parameters
    ----------
    base_dir : str
        The base directory where to store GPM data.
    username: str
        Email address with which you registered on on NASA PPS
    product : str
        GPM product name. See: GPM_products()
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
    #-------------------------------------------------------------------------.
    ## Check input arguments
    date = check_date(date)
    check_product_type(product_type = product_type)
    check_product(product = product, product_type = product_type)
    
    #-------------------------------------------------------------------------.
    ## Retrieve the list of files available on NASA PPS server
    pps_filepaths = find_pps_filepaths(username = username,
                                       product = product, 
                                       product_type = product_type,
                                       version = version,          
                                       start_time = date, 
                                       end_time = date + datetime.timedelta(days=1), # TODO CHANGE
                                       verbose = verbose)
    disk_filepaths = convert_pps_to_disk_filepaths(pps_filepaths=pps_filepaths, 
                                                   base_dir=base_dir,
                                                   product=product,
                                                   product_type=product_type, 
                                                   version=version)
    #-------------------------------------------------------------------------.
    ## If no file to retrieve on NASA PPS, return None
    if is_empty(pps_filepaths):
        # print("No data found on PPS on Date", Date, "for product", product)
        return None
    
    #-------------------------------------------------------------------------.
    ## If force_download is False, select only data not present on disk 
    (pps_filepaths, disk_filepaths) = filter_download_list(disk_paths = disk_filepaths, 
                                                           server_paths = pps_filepaths,  
                                                           force_download = force_download)
    #-------------------------------------------------------------------------.
    # Retrieve commands
    if (transfer_tool == "curl"):
        list_cmd = [curl_cmd(pps_path, disk_path, username, username) for pps_path, disk_path in zip(pps_filepaths, disk_filepaths)]
    else:    
        list_cmd = [wget_cmd(pps_path, disk_path, username, username) for pps_path, disk_path in zip(pps_filepaths, disk_filepaths)]
        
    #-------------------------------------------------------------------------.    
    ## Download the data (in parallel)
    bad_cmds = run(list_cmd, n_threads=n_threads, progress_bar=progress_bar, verbose=verbose)
    return bad_cmds


##-----------------------------------------------------------------------------. 
def download_data(base_dir,
                  username,
                  product,
                  start_time,
                  end_time,
                  product_type = 'RS',
                  version = 7,
                  n_threads = 10,
                  transfer_tool = "curl",
                  progress_bar = False, 
                  force_download = False,
                  check_integrity = True, 
                  remove_corrupted = True, 
                  verbose = True):
    """
    Download GPM data from NASA servers (day by day).
    
    Parameters
    ----------
    base_dir : str
        The base directory where to store GPM data.
    username: str
        Email address with which you registered on NASA PPS
    product : str
        GPM product acronym. See GPM_products()
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
        
    Returns
    -------
   
    boolean: int  
        0 if everything went fine.

    """  
    #-------------------------------------------------------------------------.
    ## Checks input arguments
    check_product_type(product_type = product_type) 
    check_product(product = product, product_type = product_type)
    check_version(version = version) 
    base_dir = check_base_dir(base_dir)
    start_time, end_time = check_start_end_time(start_time, end_time)    
    #-------------------------------------------------------------------------.
    # Retrieve sequence of dates
    # - Specify start_date - 1 day to include data potentially on previous day directory 
    # --> Example granules starting at 23:XX:XX in the day before and extending to 01:XX:XX
    start_date = datetime.datetime(start_time.year,start_time.month, start_time.day)
    end_date = datetime.datetime(end_time.year, end_time.month, end_time.day)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    dates = list(date_range.to_pydatetime())
    
    #-------------------------------------------------------------------------.
    # Loop over dates and download the files 
    for date in dates:
        _download_daily_data(base_dir = base_dir,
                             version =  version,
                             username = username,
                             product = product,
                             product_type = product_type,
                             date = date,  
                             start_time = start_time,
                             end_time = end_time,
                             n_threads = n_threads,
                             transfer_tool = transfer_tool,
                             progress_bar = progress_bar, 
                             force_download = force_download,
                             verbose = verbose)
        
    #-------------------------------------------------------------------------. 
    print(f'Download of available GPM {product} product completed.')
    if check_integrity: 
        l_corrupted = check_file_integrity(base_dir=base_dir, 
                                           product=product, 
                                           start_time=start_time, 
                                           end_time=end_time, 
                                           version=version,
                                           product_type=product_type,  
                                           remove_corrupted=remove_corrupted,
                                           verbose=verbose)
        print('Checking integrity of GPM files completed.')
        return l_corrupted
    return None

####--------------------------------------------------------------------------.