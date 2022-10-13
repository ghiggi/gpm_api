#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 17:45:37 2022

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
    is_not_empty,
)
from gpm_api.io.filter import (
    filter_daily_filepaths,
    granules_time_info, 
    get_time_first_daily_granule, 
    granules_end_hhmmss, 
)
from gpm_api.io.directories import get_pps_directory, get_disk_directory 

#-----------------------------------------------------------------------------.
### TODO: code could be simplified 
# -- return only pps_filepaths 

# -- requires the definition of a function translating pps_filepaths to disk_fpaths 
# --> from a list of filepaths, need special care to identify year/day/month folder around 00
#-----------------------------------------------------------------------------.


####--------------------------------------------------------------------------.
######################
#### Find utility ####
######################
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
            print("No data found on PPS on date", date, "for product", product, "(V0" + version_str + ")")
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

        
def find_pps_daily_filepaths(username,
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
        Email address with which you registered on on NASA PPS.
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
    provide_only_last_granule : bool, optional
        Used to retrieve only the last granule of the day.
    flag_first_date : bool, optional 
        Used to search granules near time 000000 stored in the previous day folder.
    verbose : bool, optional   
        Default is False. Wheter to specify when data are not available for a specific date.
    
    Returns
    -------
    pps_fpaths: list 
        List of file paths on the NASA PPS server.
    disk_fpaths: list
        List of file paths on the local disk.

    """
    # Check product validity 
    check_product(product=product, product_type=product_type)
    ##------------------------------------------------------------------------.
    # Check time format 
    start_hhmmss, end_hhmmss = check_hhmmss(start_hhmmss = start_hhmmss, 
                                            end_hhmmss = end_hhmmss)
    if end_hhmmss == "000000":
        return (None, None)
    
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
    filepaths = filter_daily_filepaths(filepaths,
                                       product = product,
                                       date = date, 
                                       product_type = product_type,
                                       start_hhmmss = start_hhmmss,
                                       end_hhmmss = end_hhmmss)
    
    ##------------------------------------------------------------------------.
    # Print an optional message if data are not available
    if not provide_only_last_granule and is_empty(filepaths) and verbose: 
        version_str = str(int(version))
        print("No data found on PPS on date", date, "for product", product, "(V0" + version_str + ")")
        
    ##------------------------------------------------------------------------.
    # Define disk file paths 
    disk_dir = get_disk_directory(base_dir = base_dir, 
                                  product = product, 
                                  product_type = product_type,
                                  date = date,
                                  version = version)
    disk_fpaths = [disk_dir + "/" + os.path.basename(filepath) for filepath in filepaths]
    
    ##------------------------------------------------------------------------.
    # Options 1 to deal with data near time 0000000 stored in previous day folder
    # - Return the filepath of the last granule in the daily folder 
    if provide_only_last_granule and is_not_empty(filepaths): 
        # Retrieve the start_time of each granules 
        _, l_start_hhmmss, _ = granules_time_info(filepaths)
        # Select filepath with latest start_time
        last_disk_fpath = disk_fpaths[np.argmax(l_start_hhmmss)]
        last_pps_fpath = filepaths[np.argmax(l_start_hhmmss)]
        return (last_pps_fpath, last_disk_fpath)
    
    ##------------------------------------------------------------------------.
    # Options 2 to deal with data near time 0000000 stored in previous day folder
    # - Check if need to retrieve the granule near time 000000 stored in previous day folder
    # - Only needed when asking data for the first date. 
    if flag_first_date:
        # Retrieve start_time of first daily granule
        if is_empty(filepaths):  # if no data in current day
            first_start_hhmmss = "000001"
        else:
            first_start_hhmmss, _ = get_time_first_daily_granule(filepaths)
        # The be sure that this is used only to search data on previous day folder  
        if int(first_start_hhmmss) > 10000: # 1 am
            first_start_hhmmss = "010000"
        # Retrieve if necessary data from last granule of previous day
        if start_hhmmss < first_start_hhmmss:
             # Retrieve last granules filepath of previous day 
             last_pps_fpath, last_disk_fpath = find_pps_daily_filepaths(username = username,
                                                                        base_dir = base_dir, 
                                                                        product = product, 
                                                                        date = date - datetime.timedelta(days=1), 
                                                                        start_hhmmss = "210000", 
                                                                        end_hhmmss = "240000",
                                                                        product_type = product_type,
                                                                        version=version,
                                                                        provide_only_last_granule=True,
                                                                        flag_first_date=False) 
             if is_not_empty(last_pps_fpath):
                 # Retrieve last granules end time  
                 last_end_hhmmss = granules_end_hhmmss(last_pps_fpath)[0]
                 # Append path to filepaths to retrieve if last_end_hhmmss > start_hhmmssS
                 if (last_end_hhmmss >= start_hhmmss):
                     filepaths.append(last_pps_fpath)
                     disk_fpaths.append(last_disk_fpath)
                     
    ##------------------------------------------------------------------------. 
    # If pps_fpaths still empty, return (None,None)
    if is_empty(filepaths):
        return (None, None)
    
    #--------------------------------------------------------------------------. 
    # Return server paths and disk paths 
    return (filepaths, disk_fpaths)

##-----------------------------------------------------------------------------.


# def find_pps_filepaths(username, 
#                        product, 
#                        start_time,
#                        end_time,
#                        product_type = 'RS',
#                        version = 7): 
#     """
#     Retrieve GPM data filepaths on NASA PPS server a specific time period and product.
        
#     Parameters
#     ----------
#     username: str
#        Email address with which you registered on on NASA PPS.
#     product : str
#         GPM product acronym.
#     start_time : datetime.datetime
#         Start time.
#     end_time : datetime.datetime
#         End time.
#     product_type : str, optional
#         GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).    
#     version : int, optional
#         GPM version of the data to retrieve if product_type = 'RS'. 
        
#     Returns
#     -------
#     filepaths : list 
#         List of GPM filepaths.
        
#     """
#     ## Checks input arguments
#     check_product_type(product_type=product_type) 
#     check_product(product=product, product_type=product_type)
#     check_version(version=version) 
#     start_time, end_time = check_time(start_time, end_time) 
    
#     # Retrieve sequence of dates 
#     dates = [start_time + datetime.timedelta(days=x) for x in range(0, (end_time-start_time).days + 1)]
   
#     # Retrieve start and end hhmmss
#     start_hhmmss = datetime.datetime.strftime(start_time,"%H%M%S")
#     end_hhmmss = datetime.datetime.strftime(end_time,"%H%M%S")
   
#     #-------------------------------------------------------------------------.
#     # Case 1: Retrieve just 1 day of data 
#     if (len(dates)==1):
#         filepaths = find_pps_daily_filepaths(username=username, 
#                                              version=version,
#                                              product=product,
#                                              product_type=product_type,
#                                              date=dates[0], 
#                                              start_hhmmss=start_hhmmss,
#                                              end_hhmmss=end_hhmmss,
#                                              flag_first_date=True)
#     #-------------------------------------------------------------------------.
#     # Case 2: Retrieve multiple days of data
#     if (len(dates) > 1):
#         filepaths = find_pps_daily_filepaths(username=username, 
#                                              version=version,
#                                              product=product,
#                                              product_type=product_type,
#                                              date=dates[0], 
#                                              start_hhmmss=start_hhmmss,
#                                              end_hhmmss='240000',
#                                              flag_first_date=True)
#         if (len(dates) > 2):
#             for date in dates[1:-1]:
#                 filepaths.extend(find_pps_daily_filepaths(username=username,
#                                                           version=version,
#                                                           product=product,
#                                                           product_type=product_type,
#                                                           date=date, 
#                                                           start_hhmmss='000000',
#                                                           end_hhmmss='240000')
#                                  )
#         filepaths.extend(find_pps_daily_filepaths(username=username,
#                                                   version=version,
#                                                   product=product,
#                                                   product_type=product_type,
#                                                   date=dates[-1], 
#                                                   start_hhmmss='000000',
#                                                   end_hhmmss=end_hhmmss)
#                          )
#     #-------------------------------------------------------------------------. 
#     filepaths = sorted(filepaths)
#     return filepaths   



