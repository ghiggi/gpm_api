#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:30:46 2022

@author: ghiggi
"""
import os
import datetime
import numpy as np 
from gpm_api.io.checks import (
    check_hhmmss, 
    check_product,
    check_filepaths,
)
from gpm_api.utils.utils_string import (
    str_extract,
    str_subset,
    str_sub,
    str_pad,
    str_detect,
)
from gpm_api.io.patterns import GPM_products_pattern_dict
 

#-----------------------------------------------------------------------------.
##################################
#### Infos from granules name ####
##################################
def granules_time_info(filepaths):
    """
    Retrieve the date, start_hhmmss and end_hhmmss of GPM granules.

    Parameters
    ----------
    filepaths : list, str
        Filepath or filename of a GPM HDF5 file.

    Returns
    -------
    date: list
        List with the date of each granule.
    start_time : list
        List with the start_hhmmss of each granule.
    end_time : list
        List with the end_hhmmss of each granule.

    """
    # Extract filename from filepath (and be sure is a list)
    if isinstance(filepaths,str):
        filepaths = [filepaths]
    filenames = [os.path.basename(filepath) for filepath in filepaths]
    # Check is not 1B DPR product (because different data format)
    is_1B_DPR = str_detect(filenames, "GPMCOR")
    # - Retrieve start_hhmmss and endtime of JAXA 1B DPR reflectivities
    if (all(is_1B_DPR)):  
        # 'GPMCOR_KAR*','GPMCOR_KUR*' # if product not in ['1B-Ka', '1B-Ku']:
        l_YYMMDD = str_sub(str_extract(filenames,"[0-9]{10}"),end=6) 
        dates = [datetime.datetime.strptime(YYMMDD, "%y%m%d").strftime("%Y%m%d") for YYMMDD in l_YYMMDD]
        l_start_hhmmss = str_sub(str_extract(filenames,"[0-9]{10}"),6) 
        l_end_hhmmss = str_sub(str_extract(filenames,"_[0-9]{4}_"),1,5) 
        l_start_hhmmss = str_pad(l_start_hhmmss, width=6, side="right",pad="0")
        l_end_hhmmss = str_pad(l_end_hhmmss, width=6, side="right",pad="0")
    elif (all(list(np.logical_not(is_1B_DPR)))):
        dates = str_sub(str_extract(filenames,"[0-9]{8}-S"), end=-2)
        l_start_hhmmss = str_sub(str_extract(filenames,"S[0-9]{6}"), 1)
        l_end_hhmmss = str_sub(str_extract(filenames,"E[0-9]{6}"), 1)  
    else:
        raise ValueError("BUG... mix of products in filepaths ?")     
    return (dates, l_start_hhmmss, l_end_hhmmss)


def granules_start_hhmmss(filepaths): 
    _, start_hhmmss,_ = granules_time_info(filepaths)
    return(start_hhmmss)


def granules_end_hhmmss(filepaths): 
    _, _, end_hhmmss = granules_time_info(filepaths)
    return(end_hhmmss)


def granules_dates(filepaths): 
    dates, _, _ = granules_time_info(filepaths)
    return(dates)


def get_name_first_daily_granule(filepaths):
    """Retrieve the name of first daily granule in the daily folder."""
    filenames = [os.path.basename(filepath) for filepath in filepaths]
    _, l_start_hhmmss, _ = granules_time_info(filenames)
    first_filename = filenames[np.argmin(l_start_hhmmss)]
    return(first_filename)


def get_time_first_daily_granule(filepaths):
    """Retrieve the start_time and end_time of first daily granule in the daily folder."""
    filename = get_name_first_daily_granule(filepaths)
    _, start_hhmmss, end_hhmmss = granules_time_info(filename)
    return (start_hhmmss[0], end_hhmmss[0])


####--------------------------------------------------------------------------.
##########################
#### Filter filepaths ####
##########################


def filter_by_product(filepaths, product, product_type="RS"): 
    """
    Filter filepaths by product.
    
    Parameters
    ----------
    filepaths : list
        List of filepaths or filenames for a specific day.
    product : str
        GPM product name. See: GPM_products()
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).  
        
    Returns 
    ----------
    filepaths : list 
        List of valid filepaths.
        If no valid filepaths, returns an empty list ! 
    
    """
    #-------------------------------------------------------------------------.
    # Check filepaths 
    if isinstance(filepaths, type(None)):
        return [] 
    filepaths = check_filepaths(filepaths)
    if len(filepaths) == 0: 
        return []

    #-------------------------------------------------------------------------.
    # Check product validity 
    check_product(product = product, 
                  product_type = product_type)
    
    #-------------------------------------------------------------------------.
    # Retrieve GPM filename dictionary 
    GPM_dict = GPM_products_pattern_dict()       
    
    #-------------------------------------------------------------------------. 
    # Subset by specific product 
    filepaths = str_subset(filepaths, GPM_dict[product])
    
    #-------------------------------------------------------------------------. 
    # Return valid filepaths 
    return filepaths
         

def filter_by_time(filepaths, 
                   date = None,  
                   start_hhmmss=None,
                   end_hhmmss=None):
    """
    Filter filepaths by time.
    
    Parameters
    ----------
    filepaths : list
        List of filepaths or filenames for a specific day.
     date : datetime
         Single date for which to retrieve the data.
     start_hhmmss : str or datetime, optional
         Start time. A datetime object or a string in hhmmss format.
         The default is None (retrieving from 000000)
     end_hhmmss : str or datetime, optional
         End time. A datetime object or a string in hhmmss format.
         The default is None (retrieving to 240000)
    
    Returns 
    ----------
    
    filepaths : list 
        List of valid filepaths.
        If no valid filepaths, returns an empty list ! 
    """
    #-------------------------------------------------------------------------.
    # Check filepaths 
    if isinstance(filepaths, type(None)):
        return [] 
    filepaths = check_filepaths(filepaths)
    if len(filepaths) == 0: 
        return []
    
    #-------------------------------------------------------------------------.
    # Check time format 
    start_hhmmss, end_hhmmss = check_hhmmss(start_hhmmss = start_hhmmss, 
                                            end_hhmmss = end_hhmmss)
       
    #-------------------------------------------------------------------------. 
    # - Retrieve start_hhmmss and endtime of GPM granules products (execept JAXA 1B reflectivities)
    l_date, l_s_hhmmss,l_e_hhmmss = granules_time_info(filepaths)
    
    #-------------------------------------------------------------------------. 
    # Check file are available 
    if len(l_date) == 0: 
        return []
   
    #-------------------------------------------------------------------------. 
    # Subset granules by date (required for NRT data)
    if (date is not None):
        idx_valid_date = np.array(l_date) == date.strftime("%Y%m%d")
        filepaths = np.array(filepaths)[idx_valid_date]
        l_s_hhmmss = np.array(l_s_hhmmss)[idx_valid_date]
        l_e_hhmmss = np.array(l_e_hhmmss)[idx_valid_date]
    
    #-------------------------------------------------------------------------. 
    # Convert hhmmss to integer 
    start_hhmmss = int(start_hhmmss)
    end_hhmmss = int(end_hhmmss)
    l_s_hhmmss = np.array(l_s_hhmmss).astype(np.int64)  # to integer 
    l_e_hhmmss = np.array(l_e_hhmmss).astype(np.int64)  # to integer 
    
    # Take care for include in subsetting the last day granule 
    idx_next_day_granule = l_e_hhmmss < l_s_hhmmss
    l_e_hhmmss[idx_next_day_granule] = 240001
    
    # Subset granules files based on start time and end time
    idx_select1 = np.logical_and(l_s_hhmmss <= start_hhmmss, l_e_hhmmss > start_hhmmss)
    idx_select2 = np.logical_and(l_s_hhmmss >= start_hhmmss, l_s_hhmmss < end_hhmmss)
    idx_select = np.logical_or(idx_select1, idx_select2)
    filepaths = list(np.array(filepaths)[idx_select])
    
    return filepaths 
    
    
def filter_daily_filepaths(filepaths,
                           product,
                           product_type = 'RS',
                           date = None,  
                           start_hhmmss=None,
                           end_hhmmss=None):
    """
    Filter the daily GPM filepaths for specific product and daytime period.

    Parameters
    ----------
    filepaths : list
        List of filepaths or filenames for a specific day.
    product : str
        GPM product name. See: GPM_products()
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

    Returns
    -------
    
    filepaths : list 
        Returns the filepaths subset.
        If no valid filepaths, return an empty list. 

    """
    #-------------------------------------------------------------------------.
    # Checks file paths 
    if isinstance(filepaths,str):
        filepaths = [filepaths]
        
    # filenames = [os.path.basename(filepath) for filepath in filepaths]
   
    filepaths = filter_by_product(filepaths=filepaths, 
                                  product=product, 
                                  product_type=product_type)
    
    filepaths = filter_by_time(filepaths, 
                               date=date,  
                               start_hhmmss=start_hhmmss,
                               end_hhmmss=end_hhmmss)
    
    #-------------------------------------------------------------------------.
    # Return filepaths 
    return filepaths

####--------------------------------------------------------------------------.
