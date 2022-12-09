#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:45:37 2022

@author: ghiggi
"""
import os 
import datetime 
import subprocess
import pandas as pd 

from gpm_api.io.checks import (
    check_date,
    check_start_end_time,
    check_product,
    check_product_type,
    check_version,
    is_empty,
)
from gpm_api.io.filter import filter_filepaths
from gpm_api.io.directories import get_pps_directory 


def _get_pps_file_list(username, url_file_list, product, date, version, verbose=True):
    """
    Retrieve the filepaths of the files available on the NASA PPS server for a specific day and product.
    
    Note: This function does not return the complete url ! 
    
    Parameters
    ----------
    username: str
        Email address with which you registered on on NASA PPS.
    product : str
        GPM product acronym. See GPM_products() .
    date : datetime
        Single date for which to retrieve the data.
    verbose : bool, optional   
        Default is False. Wheter to specify when data are not available for a specific date.
    """
    # Ensure url_file_list ends with "/"
    if url_file_list[-1] != "/":
        url_file_list = url_file_list + "/"
        
    # Define curl command 
    # -k is required with curl > 7.71 otherwise results in "unauthorized access".
    cmd = 'curl -k --user ' + username + ':' + username + ' ' + url_file_list
    # cmd = 'curl -k -4 -H "Connection: close" --ftp-ssl --user ' + username + ':' + username + ' -n ' + url_file_list
    # print(cmd)
   
    # Run command
    args = cmd.split()
    process = subprocess.Popen(args,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout = process.communicate()[0].decode()
   
    # Check if server is available
    if stdout == '':
        print("The PPS server is currently unavailable. Data download for product", 
              product, "at date", date,"has been interrupted.")
        raise ValueError("Sorry for the incovenience.")
   
    # Check if data are available
    if stdout[0] == '<':
        if verbose:
            version_str = str(int(version))
            print(f"No data found on PPS on date {date} for product {product} (V0{version_str})")
        return []
    else:
        # Retrieve filepaths
        filepaths = stdout.split() 
    
    # Return file paths 
    return filepaths     


def _get_pps_daily_filepaths(username,
                             product, 
                             product_type, 
                             date,
                             version, 
                             verbose=True): 
    """
    Retrieve the complete url to the files available on the NASA PPS server for a specific day and product.
    
    Parameters
    ----------
    
    username: str
        Email address with which you registered on on NASA PPS.
    product : str
        GPM product acronym. See GPM_products() .
    date : datetime
        Single date for which to retrieve the data.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).    
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'. 
    verbose : bool, optional 
        Whether to specify when data are not available for a specific date.
        The default is True. 
    """
    # Retrieve server urls of NASA PPS
    (url_data_server, url_file_list) = get_pps_directory(product=product, 
                                                         product_type=product_type, 
                                                         date=date,
                                                         version=version)
    # Retrieve filepaths 
    # - If empty: return []
    filepaths = _get_pps_file_list(username=username,
                                   url_file_list=url_file_list, 
                                   product=product, 
                                   date=date, 
                                   version=version, 
                                   verbose=verbose)    
            
    # Define the complete url of pps filepaths 
    # - Need to remove the starting "/" to each filepath
    pps_fpaths = [os.path.join(url_data_server, filepath[1:]) for filepath in filepaths]
    
    # Return the pps data server filepaths 
    return pps_fpaths


##-----------------------------------------------------------------------------.

        
def _find_pps_daily_filepaths(username,
                              product, 
                              date, 
                              start_time = None, 
                              end_time = None,
                              product_type = 'RS',
                              version = 7, 
                              verbose = False):
    """
    Retrieve GPM data filepaths for NASA PPS server for a specific day and product.
    
    Parameters
    ----------
    username: str
        Email address with which you registered on on NASA PPS.
    date : datetime.date
        Single date for which to retrieve the data.
    product : str
        GPM product acronym. See GPM_products()
    start_time : datetime.datetime
        Filtering start time.
    end_time : datetime.datetime
        Filtering end time.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).    
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'. 
    verbose : bool, optional   
        Default is False. Wheter to specify when data are not available for a specific date.
    
    Returns
    -------
    pps_fpaths: list 
        List of file paths on the NASA PPS server.
    disk_fpaths: list
        List of file paths on the local disk.

    """
    ##------------------------------------------------------------------------.
    # Check date
    date = check_date(date)
    
    ##------------------------------------------------------------------------.
    # Retrieve list of available files on pps
    filepaths = _get_pps_daily_filepaths(username=username,
                                         product=product, 
                                         product_type=product_type, 
                                         date=date,
                                         version=version, 
                                         verbose=verbose)          
    if is_empty(filepaths): 
        return (None, None)
    
    ##------------------------------------------------------------------------.
    # Filter the GPM daily file list (for product, start_time & end time)
    filepaths = filter_filepaths(filepaths,
                                 product=product,
                                 product_type=product_type,
                                 version=version,
                                 start_time=start_time,
                                 end_time=end_time)
    
    ##------------------------------------------------------------------------.
    # Print an optional message if data are not available
    if is_empty(filepaths) and verbose: 
        version_str = str(int(version))
        print(f"No data found on PPS on date {date} for product {product} (V0{version_str})")
    
    return filepaths 


##-----------------------------------------------------------------------------.


def find_pps_filepaths(username, 
                       product, 
                       start_time,
                       end_time,
                       product_type='RS',
                       version=7,
                       verbose=True): 
    """
    Retrieve GPM data filepaths on NASA PPS server a specific time period and product.
        
    Parameters
    ----------
    username: str
        Email address with which you registered on on NASA PPS.
    product : str
        GPM product acronym.
    start_time : datetime.datetime
        Start time.
    end_time : datetime.datetime
        End time.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).    
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'. 
        
    Returns
    -------
    filepaths : list 
        List of GPM filepaths.
        
    """
    ## Checks input arguments
    check_version(version=version)
    check_product_type(product_type=product_type) 
    check_product(product=product, product_type=product_type)
    start_time, end_time = check_start_end_time(start_time, end_time) 
    
    # Retrieve sequence of dates 
    # - Specify start_date - 1 day to include data potentially on previous day directory 
    # --> Example granules starting at 23:XX:XX in the day before and extending to 01:XX:XX
    start_date = datetime.datetime(start_time.year,start_time.month, start_time.day)
    start_date = start_date - datetime.timedelta(days=1)
    end_date = datetime.datetime(end_time.year, end_time.month, end_time.day)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    dates = list(date_range.to_pydatetime())
   
    #-------------------------------------------------------------------------.
    # Loop over dates and retrieve available filepaths 
    # TODO LIST 
    # - start_time and end_time filtering could be done only on first and last iteration
    # - misleading error message can occur on last iteration if end_time is close to 00:00:00 
    #   and the searched granule is in previous day directory 
    # - this can be done in parallel !!! 
    list_filepaths = []
    for date in dates: 
        filepaths = _find_pps_daily_filepaths(username=username, 
                                              version=version,
                                              product=product,
                                              product_type=product_type,
                                              date=date, 
                                              start_time=start_time,
                                              end_time=end_time,
                                              verbose=verbose)
        list_filepaths += filepaths
    
    #-------------------------------------------------------------------------.    
    # Return filepaths 
    filepaths = sorted(list_filepaths)
    return filepaths   