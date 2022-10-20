#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 00:18:13 2022

@author: ghiggi
"""
import os 
import datetime 
import numpy as np
import pandas as pd
import subprocess
from gpm_api.io.checks import (
    check_date,
    check_start_end_time,
    check_hhmmss, 
    check_product,
    check_product_type,
    check_version,
    check_base_dir,
    is_empty,
    is_not_empty,
)
from gpm_api.io.filter import (
    filter_daily_filepaths,
    granules_time_info, 
    get_time_first_daily_granule, 
    granules_end_hhmmss, 
)

from gpm_api.io.directories import get_disk_directory 

####--------------------------------------------------------------------------.
######################
#### Find utility ####
######################
def _get_disk_daily_filepaths(base_dir, 
                             product, 
                             product_type,
                             date,
                             version):
    """
    Retrieve GPM data filepaths on the local disk directory of a specific day and product.
    
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
    version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'. 
    """
    # Retrieve the directory on disk where the data are stored
    dir_path = get_disk_directory(base_dir = base_dir, 
                                  product = product, 
                                  product_type = product_type,
                                  date = date,
                                  version = version)
        
    # Check if the folder exists   
    if not os.path.exists(dir_path):
       print("The GPM product", product, "on date", date, "is unavailable or has not been downloaded !")
       return []
   
    # Retrieve the file names in the directory
    filenames = sorted(os.listdir(dir_path)) # returns [] if empty 
    
    # Retrieve the filepaths 
    filepaths = [os.path.join(dir_path, filename) for filename in filenames]
    
    return filepaths


def find_daily_filepaths(base_dir, 
                         product, 
                         date, 
                         start_hhmmss = None, 
                         end_hhmmss = None,
                         product_type = 'RS',
                         version = 7,
                         provide_only_last_granule = False,
                         flag_first_date = False,
                         verbose=True):
    """
    Retrieve GPM data filepaths on local disk for a specific day and product on user disk.
    
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
    provide_only_last_granule : bool, optional
        Used to retrieve only the last granule of the day.
    flag_first_date : bool, optional 
        Used to search granules near time 000000 stored in the previous day folder.
        
    Returns
    -------
    filepaths : list  
        List of GPM filepaths.
    
    """
    ##------------------------------------------------------------------------.
    # Check date and time formats 
    date = check_date(date)
    start_hhmmss, end_hhmmss = check_hhmmss(start_hhmmss, end_hhmmss)
    if end_hhmmss == "000000":
        return []
    
    ##------------------------------------------------------------------------.
    # Retrieve filepaths 
    filepaths = _get_disk_daily_filepaths(base_dir=base_dir, 
                                          product=product, 
                                          product_type=product_type,
                                          date=date,
                                          version=version)
    
    ##------------------------------------------------------------------------.
    # Filter the GPM daily file list (for product, start_time & end time)
    filepaths = filter_daily_filepaths(filepaths, product=product,
                                       date = date,  
                                       start_hhmmss=start_hhmmss, 
                                       end_hhmmss=end_hhmmss)
    
    ##-------------------------------------------------------------------------.
    # Print an optional message if daily data are not available
    if not provide_only_last_granule and is_empty(filepaths) and verbose: 
        version_str = str(int(version))        
        print("No data found on disk on date", date, "for product", product, "(V0" + version_str + ")")
        
    ##-------------------------------------------------------------------------.
    # Options 1 to deal with data near time 0000000 stored in previous day folder
    # - Return the filepath of the last granule in the daily folder 
    if provide_only_last_granule and is_not_empty(filepaths): 
        # Retrieve the start_time of each granules 
        _, l_start_hhmmss, _ = granules_time_info(filepaths)
        # Select filepath with latest start_time
        last_filepath = filepaths[np.argmax(l_start_hhmmss)]
        return last_filepath
    
    ##-------------------------------------------------------------------------.
    # Options 2 to deal with data near time 0000000 stored in previous day folder
    # - Check if need to retrieve the granule near time 000000 stored in previous day folder
    # - Only needed when asking data for the first date. 
    if flag_first_date:
         # Retrieve start_time of first daily granule
        if  is_empty(filepaths):  # if no data in current day
            first_start_hhmmss = "000001"
        else:
            first_start_hhmmss, _ = get_time_first_daily_granule(filepaths)
        # To be sure that this is used only to search data on previous day folder  
        if int(first_start_hhmmss) > 10000: # 1 am
            first_start_hhmmss = "010000"
        # Retrieve last granules filepath of previous day  
        if start_hhmmss < first_start_hhmmss:
            last_filepath = find_daily_filepaths(base_dir = base_dir, 
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
    # If filepaths still empty, return []
    if is_empty(filepaths):
        return []
    
    ##------------------------------------------------------------------------.
    return filepaths


def find_filepaths(base_dir, 
                   product, 
                   start_time,
                   end_time,
                   product_type = 'RS',
                   version = 7): 
    """
    Retrieve GPM data filepaths on local disk for a specific time period and product.
        
    Parameters
    ----------
    base_dir : str
       The base directory where GPM data are stored.
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
    check_product_type(product_type=product_type) 
    check_product(product=product, product_type=product_type)
    check_version(version=version)
    base_dir = check_base_dir(base_dir)
    start_time, end_time = check_start_end_time(start_time, end_time) 
    
    # Retrieve sequence of dates 
    start_date = datetime.datetime(start_time.year,start_time.month, start_time.day)
    end_date = datetime.datetime(end_time.year, end_time.month, end_time.day)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    dates = list(date_range.to_pydatetime())

    # Retrieve start and end hhmmss
    start_hhmmss = datetime.datetime.strftime(start_time,"%H%M%S")
    end_hhmmss = datetime.datetime.strftime(end_time,"%H%M%S")
   
    #-------------------------------------------------------------------------.
    # Case 1: Retrieve just 1 day of data 
    if len(dates)==1:
        filepaths = find_daily_filepaths(base_dir=base_dir, 
                                         version=version,
                                         product=product,
                                         product_type=product_type,
                                         date=dates[0], 
                                         start_hhmmss=start_hhmmss,
                                         end_hhmmss=end_hhmmss,
                                         flag_first_date=True)
    #-------------------------------------------------------------------------.
    # Case 2: Retrieve multiple days of data
    if len(dates) > 1:
        filepaths = find_daily_filepaths(base_dir=base_dir, 
                                         version=version,
                                         product=product,
                                         product_type=product_type,
                                         date=dates[0], 
                                         start_hhmmss=start_hhmmss,
                                         end_hhmmss='240000',
                                         flag_first_date=True)
        if len(dates) > 2:
            for date in dates[1:-1]:
                filepaths.extend(find_daily_filepaths(base_dir=base_dir,
                                                      version=version,
                                                      product=product,
                                                      product_type=product_type,
                                                      date=date, 
                                                      start_hhmmss='000000',
                                                      end_hhmmss='240000')
                                 )
        filepaths.extend(find_daily_filepaths(base_dir=base_dir,
                                              version=version,
                                              product=product,
                                              product_type=product_type,
                                              date=dates[-1], 
                                              start_hhmmss='000000',
                                              end_hhmmss=end_hhmmss)
                         )
    #-------------------------------------------------------------------------. 
    filepaths = sorted(filepaths)
    return filepaths   


 

