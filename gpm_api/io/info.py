#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 20:49:06 2022

@author: ghiggi
"""
import re
import os 
import datetime
from trollsift import Parser
from gpm_api.io.patterns import GPM_products_pattern_dict

# General pattern for all GPM products
GPM_FNAME_PATTERN = "{level:s}.{satellite:s}.{sensor:s}.{algorithm:s}.{start_date:%Y%m%d}-S{start_time:%H%M%S}-E{end_time:%H%M%S}.{id}.{version}.{format}"
# Pattern for 1B Ku and Ka
GPM_FNAME_PATTERN2 = "GPMCOR_{sensor:s}_{start_time:%y%m%d%H%M}_{end_time:%H%M}_{id}_{unknown}_{version}.{format}"

# # IMERG 
# fname = "3B-HHR.MS.MRG.3IMERG.20140422-S043000-E045959.0270.V06B.HDF5"
 
# # 2B
# fname = "2B.GPM.DPRGMI.2HCSHv7-0.20140422-S013047-E030320.000831.V07A.HDF5"
# fname = "2B.GPM.DPRGMI.CORRA2022.20140422-S230649-E003923.000845.V07A.HDF5"
# fname = "2B.TRMM.PRTMI.2HCSHv7-0.20140422-S110725-E123947.093603.V07A.HDF5"

# # 2A
# fname = "2A.MT1.SAPHIR.PRPS2019v2-02.20140422-S000858-E015053.013038.V06A.HDF5"
# fname = "2A.GPM.DPR.V9-20211125.20201028-S075448-E092720.037875.V07A.HDF5"
# # 1C 
# fname = "1C.GPM.GMI.XCAL2016-C.20140422-S013047-E030320.000831.V07A.HDF5"
# # 1B
# fname = "GPMCOR_KUR_2010280754_0927_037875_1BS_DAB_07A.h5"
# fname = "GPMCOR_KAR_2010280754_0927_037875_1BS_DAB_07A.h5"
# fname = "1B.TRMM.PR.V9-20210630.20140422-S002044-E015306.093596.V07A.HDF5"
# # 1A
# fname = "1A.GPM.GMI.COUNT2021.20140422-S230649-E003923.000845.V07A.HDF5"
# fname = "1A.TRMM.TMI.COUNT2021.20140422-S002044-E015306.093596.V07A.HDF5"

# _get_info_from_filename(fname)

def _get_info_from_filename(fname):
    """Retrieve file information dictionary from filename."""    
    try: 
        # Retrieve information from filename 
        p = Parser(GPM_FNAME_PATTERN)
        info_dict = p.parse(fname)
        
        # Retrieve correct start_time and end_time
        start_date = info_dict['start_date']
        start_time = info_dict['start_time']
        end_time = info_dict['end_time']
        start_datetime = start_date.replace(hour=start_time.hour,  minute=start_time.minute, second=start_time.second)
        end_datetime = start_date.replace(hour=end_time.hour,  minute=end_time.minute, second=end_time.second)
        if end_time < start_time:
           end_datetime = end_datetime + datetime.timedelta(days=1)
    except ValueError:
        try:
            p = Parser(GPM_FNAME_PATTERN2)
            info_dict = p.parse(fname)
            info_dict['level'] = "1B"
            info_dict['satellite'] = "GPM"
            # Retrieve correct start_time and end_time
            start_datetime = info_dict['start_time']
            end_time = info_dict['end_time']
            end_datetime = start_datetime.replace(hour=end_time.hour,  minute=end_time.minute, second=end_time.second)
            if end_datetime < start_datetime:
                end_datetime = end_datetime + datetime.timedelta(days=1) 
        except:
            raise ValueError(f"{fname} can not be parsed. Report the issue.")
       
    # Retrieve complete start_time and end_time datetime object 
    info_dict["start_time"] = start_datetime
    info_dict["end_time"] = end_datetime
    _ =  info_dict.pop("start_date", None)
    
    # # Special treatment for  ... 
    # if info_dict.get("algorithm") == "..."
        
    # Return info dictionary
    return info_dict


def _get_info_from_filepath(fpath):
    """Retrieve file information dictionary from filepath."""
    if not isinstance(fpath, str):
        raise TypeError("'fpath' must be a string.")
    fname = os.path.basename(fpath)
    return _get_info_from_filename(fname)



def _get_key_from_filepaths(fpaths, key):
    """Extract specific key information from a list of filepaths."""
    if isinstance(fpaths, str):
        fpaths = [fpaths]
    return [
        _get_info_from_filepath(fpath)[key] for fpath in fpaths
    ]

# TODO_goes_api problem 
# def get_key_from_filepaths(fpaths, key):
#     """Extract specific key information from a list of filepaths."""
#     if isinstance(fpaths, dict):
#         fpaths = {k: _get_key_from_filepaths(v, key=key) for k, v in fpaths.items()}
#     else:
#         fpaths = _get_key_from_filepaths(fpaths, key=key)
#     return fpaths 


def get_product_from_filepath(filepath): 
    GPM_dict = GPM_products_pattern_dict() 
    for product, pattern in GPM_dict.items(): 
        if re.search(pattern, filepath):
            return product 
    else: 
        raise ValueError(f"GPM Product unknown for {filepath}.")
        
def get_version_from_filepath(filepath, integer=True): 
    version = _get_key_from_filepaths(filepath, key="version")[0]
    if integer: 
        version = int(re.findall('\d+', version)[0])        
    return version 
    