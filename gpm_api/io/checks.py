#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 20:02:18 2022

@author: ghiggi
"""
import datetime
import numpy as np 


def is_not_empty(x):
    return not not x


def is_empty(x):
    return not x


def check_filepaths(filepaths):
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    if not isinstance(filepaths, list):
        raise TypeError("Expecting a list of filepaths.")
    return filepaths 

def check_variables(variables):
    if not isinstance(variables, (str, list, np.ndarray, type(None))):
        raise TypeError("'variables' must be a either a str, list, np.ndarray or None.")
    if variables is None: 
        return None
    if isinstance(variables, str): 
        variables = [variables]
    elif isinstance(variables, list): 
        variables = np.array(variables)
    return variables
 
    
def check_groups(groups):
    if not isinstance(groups, (str,list, np.ndarray, type(None))):
        raise TypeError("'groups' must be a either a str, list, np.ndarray or None.")
    if isinstance(groups, str): 
        groups = [groups]
    elif isinstance(groups, list): 
        groups = np.array(groups)
    return groups


def check_version(version):
    if not isinstance(version, int): 
        raise ValueError("Please specify the GPM version with an integer between 5 and 7.")
    if version not in [5,6,7]:
        raise ValueError("Download/Reading have been implemented only for GPM versions 5, 6 and 7.")
              
        
def check_product(product, product_type):
    from gpm_api.io.products import GPM_products
    if not isinstance(product, str):
        raise ValueError("'Ask for a single product at time.'product' must be a single string.")   
    if product not in GPM_products(product_type = product_type):
        raise ValueError("Please provide a valid GPM product --> GPM_products().")  
        
        
def check_product_type(product_type):
    if not isinstance(product_type, str): 
        raise ValueError("Please specify the product_type as 'RS' or 'NRT'.")
    if product_type not in ['RS','NRT']:
        raise ValueError("Please specify the product_type as 'RS' or 'NRT'.")


def check_time(start_time, end_time):
    if not isinstance(start_time, datetime.datetime):
        raise ValueError("start_time must be a datetime object.")
    if not isinstance(end_time, datetime.datetime):
        raise ValueError("end_time must be a datetime object.")
    # Check start_time and end_time are chronological  
    if (start_time > end_time):
        raise ValueError('Provide start_time occuring before of end_time.')   
    return (start_time, end_time)
           
 
def check_date(date):
    if not isinstance(date, (datetime.date, datetime.datetime)):
        raise ValueError("date must be a datetime object")
    if isinstance(date, datetime.datetime):
        date = date.date()
    return date
    

def check_hhmmss(start_hhmmss, end_hhmmss):
    # Check start_hhmmss 
    if start_hhmmss is None:
        start_hhmmss = '000000' 
    elif isinstance(start_hhmmss, datetime.time):
        start_hhmmss = datetime.time.strftime(start_hhmmss, '%H%M%S')    
    elif isinstance(start_hhmmss, datetime.datetime):
        start_hhmmss = datetime.datetime.strftime(start_hhmmss, '%H%M%S')
    elif isinstance(start_hhmmss, str):   
        if len(start_hhmmss) != 6:
            raise ValueError("Please provide start_hhmmss as HHMMSS string or as datetime.time")
        start_hhmmss = start_hhmmss
    else: 
        raise ValueError("Please provide start_hhmmss as HHMMSS string or as datetime.time")
    #-------------------------------------------------------------------------.
    # Check end time 
    if end_hhmmss is None:
        end_hhmmss = '240000'
    elif isinstance(end_hhmmss, datetime.time):
        end_hhmmss = datetime.time.strftime(end_hhmmss, '%H%M%S')  
    elif isinstance(end_hhmmss, datetime.datetime):
        end_hhmmss = datetime.datetime.strftime(end_hhmmss, '%H%M%S')
    elif isinstance(end_hhmmss, str): 
        if len(end_hhmmss) != 6:
            raise ValueError("Please provide end_hhmmss as HHMMSS string or as datetime.time")
        end_hhmmss = end_hhmmss
    else: 
        raise ValueError("Please provide end_hhmmss as HHMMSS string or as datetime.time") 
    return (start_hhmmss, end_hhmmss)


##----------------------------------------------------------------------------.
####################
#### Scan Modes ####
####################


def check_scan_mode(scan_mode, product, version):
    """Checks the validity of scan_mode."""
    #-------------------------------------------------------------------------.
    # Get valid scan modes 
    from gpm_api.io.scan_modes import get_valid_scan_modes
    scan_modes = get_valid_scan_modes(product, version)
    
    # Infer scan mode if not specified 
    if scan_mode is None:
        scan_mode = scan_modes[0]
        if len(scan_modes) > 1: 
            print(f"'scan_mode' has not been specified. Default to {scan_mode}.")
   
    #-------------------------------------------------------------------------.
    # Check that a single scan mode is specified
    if scan_mode is not None:
        if not isinstance(scan_mode, str):
            raise ValueError("Specify a single 'scan_mode'.")
        
    #-------------------------------------------------------------------------.    
    # Check that a valid scan mode is specified  
    if scan_mode is not None:
        if not scan_mode in scan_modes:
            raise ValueError(f"For {product} product, valid scan_modes are {scan_modes}.") 
                                
    #-------------------------------------------------------------------------.      
    return scan_mode


##----------------------------------------------------------------------------.
def check_bbox(bbox):
    """
    Check correctnes of bounding box.

    bbox format: [lon_0, lon_1, lat_0, lat_1]
    bbox should be provided with longitude between -180 and 180, and latitude
    between -90 and 90.
    """
    if bbox is None:
        return bbox
    # If bbox provided
    if not (isinstance(bbox, list) and len(bbox) == 4):
        raise ValueError("Provide valid bbox [lon_0, lon_1, lat_0, lat_1]")
    if bbox[2] > 90 or bbox[2] < -90 or bbox[3] > 90 or bbox[3] < -90:
        raise ValueError("Latitude is defined between -90 and 90")
    # Try to be sure that longitude is specified between -180 and 180
    if bbox[0] > 180 or bbox[1] > 180:
        print("bbox should be provided with longitude between -180 and 180")
        bbox[0] = bbox[0] - 180
        bbox[1] = bbox[1] - 180
    return bbox