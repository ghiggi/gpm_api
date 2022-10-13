#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 00:18:13 2022

@author: ghiggi
"""
import os 
import datetime 
import numpy as np
import subprocess
from gpm_api.io.checks import (
    check_date,
    check_time,
    check_hhmmss, 
    check_product,
    check_product_type,
    check_version,
    is_empty,
    is_not_empty
)
from gpm_api.io.filter import (
    filter_daily_GPM_files,
    granules_time_info, 
    get_time_first_daily_granule, 
    granules_end_hhmmss, 
)
from gpm_api.io.directories import get_GPM_PPS_directory, get_GPM_disk_directory 


def find_daily_GPM_disk_filepaths(base_dir, 
                                  product, 
                                  date, 
                                  start_hhmmss = None, 
                                  end_hhmmss = None,
                                  product_type = 'RS',
                                  version = 7,
                                  provide_only_last_granule = False,
                                  flag_first_date = False):
    """
    Retrieve GPM data filepaths for a specific day and product on user disk.
    
    Parameters
    ----------
    base_dir : str
        The base directory where to store GPM data.
    product : str
        GPM product acronym. See GPM_products()
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).
    date : datetime
        Single date for which to retrieve the data.
    start_hhmmss : str or datetime, optional
        Start time. A datetime object or a string in hhmmss format.
        The default is None (retrieving from 000000)
    end_hhmmss : str or datetime, optional
        End time. A datetime object or a string in hhmmss format.
        The default is None (retrieving to 240000)    
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'. 
        GPM data readers are currently implemented only for GPM V06.
    provide_only_last_granule : bool, optional
        Used to retrieve only the last granule of the day.
    flag_first_date : bool, optional 
        Used to search granules near time 000000 stored in the previous day folder.
        
    Returns
    -------
    list 
        List of GPM data filepaths.
    """
    ##------------------------------------------------------------------------.
    # Check date and time formats 
    date = check_date(date)
    start_hhmmss, end_hhmmss = check_hhmmss(start_hhmmss, end_hhmmss)
    if (end_hhmmss == "000000"):
        return []
    ##------------------------------------------------------------------------.
    # Retrieve the directory on disk where the data are stored
    DIR = get_GPM_disk_directory(base_dir = base_dir, 
                                 product = product, 
                                 product_type = product_type,
                                 date = date,
                                 version = version)
    ##------------------------------------------------------------------------.
    # Check if the folder exists   
    if (not os.path.exists(DIR)):
       print("The GPM product", product, "on date", date, "is unavailable or has not been downloaded !")
       return []
    # Retrieve the file names in the directory
    filenames = sorted(os.listdir(DIR))
    ##------------------------------------------------------------------------.
    # Filter the GPM daily file list (for product, start_time & end time)
    filenames = filter_daily_GPM_files(filenames, product=product,
                                       date = date, # not necessary in reality
                                       start_hhmmss=start_hhmmss, 
                                       end_hhmmss=end_hhmmss)
    ##------------------------------------------------------------------------.
    # Create the filepath 
    filepaths = [os.path.join(DIR,filename) for filename in filenames]
    ##-------------------------------------------------------------------------.
    # Options 1 to deal with data near time 0000000 stored in previous day folder
    # - Return the filepath of the last granule in the daily folder 
    if ((provide_only_last_granule is True) and (not not filepaths)): #and filepaths not empty:
        # Retrieve the start_time of each granules 
        filenames = [os.path.basename(filepath) for filepath in filepaths]
        _, l_start_hhmmss, _ = granules_time_info(filenames)
        # Select filepath with latest start_time
        last_filepath = filepaths[np.argmax(l_start_hhmmss)]
        return last_filepath
    ##-------------------------------------------------------------------------.
    # Options 2 to deal with data near time 0000000 stored in previous day folder
    # - Check if need to retrieve the granule near time 000000 stored in previous day folder
    # - Only needed when asking data for the first date. 
    if (flag_first_date is True):
         # Retrieve start_time of first daily granule
        if (not filepaths):  # if empty list (no data in current day)
            first_start_hhmmss = "000001"
        else:
            first_start_hhmmss, _ = get_time_first_daily_granule(filepaths)
        # To be sure that this is used only to search data on previous day folder  
        if  (int(first_start_hhmmss) > 10000): # 1 am
            first_start_hhmmss = "010000"
        # Retrieve last granules filepath of previous day  
        if (start_hhmmss < first_start_hhmmss):
            last_filepath = find_daily_GPM_disk_filepaths(base_dir = base_dir, 
                                                          product = product, 
                                                          date = date - datetime.timedelta(days=1), 
                                                          start_hhmmss = "210000", 
                                                          end_hhmmss = "240000",
                                                          product_type = product_type,
                                                          version = version,
                                                          provide_only_last_granule = True,
                                                          flag_first_date = False) 
            if is_not_empty(last_filepath):
                # Retrieve last granules end time  
                last_end_hhmmss = granules_end_hhmmss(last_filepath)[0]
                # Append path to filepaths to retrieve if last_end_hhmmss > start_hhmmssS
                if (last_end_hhmmss >= start_hhmmss):
                    filepaths.append(last_filepath)
    ##------------------------------------------------------------------------.
    # If filepaths still empty, return (None,None)
    if is_empty(filepaths):
        return []
    ##------------------------------------------------------------------------.
    return filepaths

def find_daily_GPM_PPS_filepaths(username,
                                 base_dir, 
                                 product, 
                                 date, 
                                 start_hhmmss = None, 
                                 end_hhmmss = None,
                                 product_type = 'RS',
                                 version = 7, 
                                 provide_only_last_granule = False,
                                 flag_first_date = False,
                                 verbose = False):
    """
    Retrieve GPM data filepaths for NASA PPS server for a specific day and product.
    
    Parameters
    ----------
    base_dir : str
        The base directory where to store GPM data.
    username: str
        Email address with which you registered on on NASA PPS    
    product : str
        GPM product acronym. See GPM_products()
    date : datetime
        Single date for which to retrieve the data.
    start_hhmmss : str or datetime, optional
        Start time. A datetime object or a string in hhmmss format.
        The default is None (retrieving from 000000)
    end_hhmmss : str or datetime, optional
        End time. A datetime object or a string in hhmmss format.
        The default is None (retrieving to 240000)
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).    
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'. 
        GPM data readers are currently implemented only for GPM V06.
    provide_only_last_granule : bool, optional
        Used to retrieve only the last granule of the day.
    flag_first_date : bool, optional 
        Used to search granules near time 000000 stored in the previous day folder.
    verbose : bool, optional   
        Default is False. Wheter to specify when data are not available for a specific date.
    
    Returns
    -------
    server_paths: list 
        List of filepaths on the NASA PPS server  
    disk_paths: list
        List of filepaths on disk u 

    """
    # Check product validity 
    check_product(product = product, product_type = product_type)
    ##------------------------------------------------------------------------.
    # Check time format 
    start_hhmmss, end_hhmmss = check_hhmmss(start_hhmmss = start_hhmmss, 
                                            end_hhmmss = end_hhmmss)
    if (end_hhmmss == "000000"):
        return (None, None)
    ##------------------------------------------------------------------------.
    # Retrieve server url of NASA PPS
    (url_data_server, url_file_list) = get_GPM_PPS_directory(product = product, 
                                                             product_type = product_type, 
                                                             date = date,
                                                             version = version)
    ##------------------------------------------------------------------------.
    ## Retrieve the name of available file on NASA PPS servers
    # curl -u username:password
    # -k is required with curl > 7.71 otherwise unauthorized access
    cmd = 'curl -k --user ' + username + ':' + username + ' ' + url_file_list
    # cmd = 'curl -k -4 -H "Connection: close" --ftp-ssl --user ' + username + ':' + username + ' -n ' + url_file_list
    # print(cmd)
    args = cmd.split()
    process = subprocess.Popen(args,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout = process.communicate()[0].decode()
    # Check if server is available
    if (stdout == ''):
        print("The PPS server is currently unavailable. Data download for product", 
              product, "at date", date,"has been interrupted.")
        raise ValueError("Sorry for the incovenience.")
    # Check if data are available
    if (stdout[0] == '<'):
        if verbose is True:
            print("No data found on PPS on date", date, "for product", product)
        return (None,None)
    else:
        # Retrieve filepaths
        filepaths = stdout.split() 
    ##------------------------------------------------------------------------.
    # Filter the GPM daily file list (for product, start_time & end time)
    filepaths = filter_daily_GPM_files(filepaths,
                                       product = product,
                                       date = date, 
                                       product_type = product_type,
                                       start_hhmmss = start_hhmmss,
                                       end_hhmmss = end_hhmmss)
    ##------------------------------------------------------------------------.
    ## Generate server and disk file paths 
    # Generate server file paths 
    server_paths = [url_data_server + filepath for filepath in filepaths]
    # Generate disk file paths 
    disk_dir = get_GPM_disk_directory(base_dir = base_dir, 
                                      product = product, 
                                      product_type = product_type,
                                      date = date,
                                      version = version)
    disk_paths = [disk_dir + "/" + os.path.basename(filepath) for filepath in filepaths]
    ##------------------------------------------------------------------------.
    # Options 1 to deal with data near time 0000000 stored in previous day folder
    # - Return the filepath of the last granule in the daily folder 
    if ((provide_only_last_granule is True) and (is_not_empty(filepaths))): 
        # Retrieve the start_time of each granules 
        _, l_start_hhmmss, _ = granules_time_info(filepaths)
        # Select filepath with latest start_time
        last_disk_path = disk_paths[np.argmax(l_start_hhmmss)]
        last_server_path = server_paths[np.argmax(l_start_hhmmss)]
        return (last_server_path, last_disk_path)
    ##------------------------------------------------------------------------.
    # Options 2 to deal with data near time 0000000 stored in previous day folder
    # - Check if need to retrieve the granule near time 000000 stored in previous day folder
    # - Only needed when asking data for the first date. 
    if (flag_first_date is True):
        # Retrieve start_time of first daily granule
        if not filepaths:  # if empty list (no data in current day)
            first_start_hhmmss = "000001"
        else:
            first_start_hhmmss, _ = get_time_first_daily_granule(filepaths)
        # The be sure that this is used only to search data on previous day folder  
        if  (int(first_start_hhmmss) > 10000): # 1 am
            first_start_hhmmss = "010000"
        # Retrieve if necessary data from last granule of previous day
        if (start_hhmmss < first_start_hhmmss):
             # Retrieve last granules filepath of previous day 
             last_server_path, last_disk_path = find_daily_GPM_PPS_filepaths(username = username,
                                                                               base_dir = base_dir, 
                                                                               product = product, 
                                                                               date = date - datetime.timedelta(days=1), 
                                                                               start_hhmmss = "210000", 
                                                                               end_hhmmss = "240000",
                                                                               product_type = product_type,
                                                                               version = version,
                                                                               provide_only_last_granule = True,
                                                                               flag_first_date = False) 
             if (last_server_path is not None):
                 # Retrieve last granules end time  
                 last_end_hhmmss = granules_end_hhmmss(last_server_path)[0]
                 # Append path to filepaths to retrieve if last_end_hhmmss > start_hhmmssS
                 if (last_end_hhmmss >= start_hhmmss):
                     server_paths.append(last_server_path)
                     disk_paths.append(last_disk_path)
    ##------------------------------------------------------------------------. 
    # If server_paths still empty, return (None,None)
    if is_empty(server_paths):
        return (None, None)
    #--------------------------------------------------------------------------. 
    # Return server paths and disk paths 
    return (server_paths, disk_paths)

##-----------------------------------------------------------------------------.
def find_GPM_files(base_dir, 
                   product, 
                   start_time,
                   end_time,
                   product_type = 'RS',
                   version = 7):
    """
    Retrieve filepaths of GPM data on user disk.
    
    Parameters
    ----------
    base_dir : str
       The base directory where GPM data are stored.
    product : str
        GPM product acronym.
    start_time : datetime
        Start time.
    end_time : datetime
        End time.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).    
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'. 
        GPM data readers are currently implemented only for GPM V06.
        
    Returns
    -------
    List of filepaths of GPM data.

    """
    ## Checks input arguments
    check_product_type(product_type = product_type) 
    check_product(product = product, product_type = product_type)
    check_version(version = version) 
    start_time, end_time = check_time(start_time, end_time) 
    # Retrieve sequence of dates 
    dates = [start_time + datetime.timedelta(days=x) for x in range(0, (end_time-start_time).days + 1)]
    # Retrieve start and end hhmmss
    start_hhmmss = datetime.datetime.strftime(start_time,"%H%M%S")
    end_hhmmss = datetime.datetime.strftime(end_time,"%H%M%S")
    #-------------------------------------------------------------------------.
    # Case 1: Retrieve just 1 day of data 
    if (len(dates)==1):
        filepaths = find_daily_GPM_disk_filepaths(base_dir = base_dir, 
                                                  version = version,
                                                  product = product,
                                                  product_type = product_type,
                                                  date = dates[0], 
                                                  start_hhmmss = start_hhmmss,
                                                  end_hhmmss = end_hhmmss,
                                                  flag_first_date = True)
    #-------------------------------------------------------------------------.
    # Case 2: Retrieve multiple days of data
    if (len(dates) > 1):
        filepaths = find_daily_GPM_disk_filepaths(base_dir = base_dir, 
                                                  version = version,
                                                  product = product,
                                                  product_type = product_type,
                                                  date = dates[0], 
                                                  start_hhmmss = start_hhmmss,
                                                  end_hhmmss = '240000',
                                                  flag_first_date = True)
        if (len(dates) > 2):
            for date in dates[1:-1]:
                filepaths.extend(find_daily_GPM_disk_filepaths(base_dir=base_dir,
                                                               version = version,
                                                               product=product,
                                                               product_type = product_type,
                                                               date=date, 
                                                               start_hhmmss='000000',
                                                               end_hhmmss='240000')
                                 )
        filepaths.extend(find_daily_GPM_disk_filepaths(base_dir = base_dir,
                                                       version = version,
                                                       product = product,
                                                       product_type = product_type,
                                                       date = dates[-1], 
                                                       start_hhmmss='000000',
                                                       end_hhmmss=end_hhmmss)
                         )
    #-------------------------------------------------------------------------. 
    filepaths = sorted(filepaths)
    return filepaths   
