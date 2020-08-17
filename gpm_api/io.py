#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:31:08 2020

@author: ghiggi
"""
#----------------------------------------------------------------------------.
import subprocess
import os
import numpy as np 
import datetime
from datetime import timedelta
from .utils.utils_string import str_extract
from .utils.utils_string import str_subset
from .utils.utils_string import str_sub 
from .utils.utils_string import str_pad 
from .utils.utils_string import subset_list_by_boolean
#----------------------------------------------------------------------------.
def curl_download(server, filepath, DIR, username, password):
    """Download data using curl."""
    # Check DIR exists 
    if not os.path.exists(DIR):
        os.mkdirs(DIR)
    # Define url  from which to retrieve data
    url = server + filepath
    # Define command to execute
    # curl -4 --ftp-ssl --user [user name]:[password] -n [url]
    cmd = 'curl -u ' + username + ':' + password + ' -n ' + url + ' -o ' + DIR + "/" + os.path.basename(filepath)
    args = cmd.split()
    # Execute the command  
    process = subprocess.Popen(args,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    return process 
#----------------------------------------------------------------------------.
################################# 
### File Pattern dictionary  ####
################################# 
### DPR
def GPM_DPR_1B_RS_pattern_dict():
     """Return the filename pattern* associated to GPM DPR 1B products."""
     GPM_dict = {'1B-PR': '1B.TRMM.PR*',
                 '1B-Ka': 'GPMCOR_KAR*',
                 '1B-Ku': 'GPMCOR_KUR*'}
     return GPM_dict 
 
def GPM_DPR_2A_RS_pattern_dict(): 
    """Return the filename pattern* associated to GPM DPR 2A RS products."""
    GPM_dict = {'2A-PR': '2A.TRMM.PR.V\d-*',   # to distinguish from SLH
                '2A-DPR': '2A.GPM.DPR.V\d-*', # to distinguish from SLH
                '2A-Ka': '2A.GPM.Ka.V*',
                '2A-Ku': '2A.GPM.Ku.V*',
                
                '2A-ENV-PR':  '2A-ENV.TRMM.PR.V*',
                '2A-ENV-DPR': '2A-ENV.GPM.DPR.V*',
                '2A-ENV-Ka': '2A-ENV.GPM.Ka.V*',
                '2A-ENV-Ku': '2A-ENV.GPM.Ku.V*',
                
                '2A-DPR-SLH': '2A.GPM.DPR.GPM-SLH*',
                '2A-PR-SLH': '2A.TRMM.PR.TRMM-SLH*'}
    return GPM_dict  

def GPM_DPR_2A_NRT_pattern_dict(): 
    """Return the filename pattern* associated to GPM DPR 2A NRT products."""
    GPM_dict = {'2A-DPR': '2A.GPM.DPR.V\d-*', # to distinguish from SLH
                '2A-Ka': '2A.GPM.Ka.V*',
                '2A-Ku': '2A.GPM.Ku.V*'}
    return GPM_dict  

def GPM_DPR_RS_pattern_dict(): 
    """Return the filename pattern* associated to GPM DPR RS products."""
    GPM_dict = GPM_DPR_1B_RS_pattern_dict()    
    GPM_dict.update(GPM_DPR_2A_RS_pattern_dict())                   
    return GPM_dict  

def GPM_DPR_NRT_pattern_dict(): 
    """Return the filename pattern* associated to GPM DPR NRT products."""
    GPM_dict = GPM_DPR_2A_NRT_pattern_dict()                 
    return GPM_dict  

##----------------------------------------------------------------------------.
### PMW
def GPM_PMW_1B_RS_pattern_dict(): 
    """Return the filename pattern* associated to GPM PMW 1B products."""
    GPM_dict = {'1B-TMI': '1B.TRMM.TMI.*',
                '1B-GMI': '1B.GPM.GMI.*'}
    return(GPM_dict)

def GPM_PMW_1B_NRT_pattern_dict(): 
    """Return the filename pattern* associated to GPM PMW 1B products."""
    GPM_dict = {'1B-GMI': '1B.GPM.GMI.*'}
    return(GPM_dict)

def GPM_PMW_1C_RS_pattern_dict():
    """Return the filename pattern* associated to GPM PMW 1C products."""
    GPM_dict = {# Common calibrated brightness temperatures 
                '1C-TMI': '1C.TMI.TRMM.*',
                '1C-GMI': '1C.GPM.GMI.*',
                '1C-SSMI-F11': '1C.F11.SSMIS.*',
                '1C-SSMI-F13': '1C.F13.SSMIS.*',
                '1C-SSMI-F14': '1C.F14.SSMIS.*',
                '1C-SSMI-F15': '1C.F15.SSMIS.*',
                '1C-SSMI-F16': '1C.F16.SSMIS.*',
                '1C-SSMI-F17': '1C.F17.SSMIS.*',
                '1C-SSMI-F18': '1C.F18.SSMIS.*',
                '1C-ASMR2-GCOMW1': '1C.GCOMW1.ASMR2.*',
                '1C-AMSRE-AQUA': '1C.AQUA.AMSRE.*',
                '1C-AMSUB-NOAA15': '1C.AMSUB.NOAA15.*',
                '1C-AMSUB-NOAA16': '1C.AMSUB.NOAA16.*',   
                '1C-SAPHIR-MT1' : '1C.SAPHIR.MT1.*',  
                '1C-MHS-METOPA': '1C.METOPA.MHS.*',
                '1C-MHS-METOPB': '1C.METOPB.MHS.*',
                '1C-MHS-METOPC': '1C.METOPC.MHS.*',
                '1C-MHS-NOAA18': '1C.NOAA18.MHS.*',   
                '1C-MHS-NOAA19': '1C.NOAA19.MHS.*',   
                '1C-ATMS-NPP': '1C.NPP.ATMS.*'}
    return GPM_dict 
     
def GPM_PMW_1C_NRT_pattern_dict():
    """Return the filename pattern* associated to GPM PMW 1C products."""
    GPM_dict = {# Common calibrated brightness temperatures 
                '1C-GMI': '1C.GPM.GMI.*',
                '1C-SSMI-F16': '1C.F16.SSMIS.*',
                '1C-SSMI-F17': '1C.F17.SSMIS.*',
                '1C-SSMI-F18': '1C.F18.SSMIS.*',
                '1C-ASMR2-GCOMW1': '1C.GCOMW1.ASMR2.*', 
                '1C-SAPHIR-MT1' : '1C.SAPHIR.MT1.*',  
                '1C-MHS-METOPB': '1C.METOPB.MHS.*',
                '1C-MHS-METOPC': '1C.METOPC.MHS.*', 
                '1C-MHS-NOAA19': '1C.NOAA19.MHS.*',   
                '1C-ATMS-NOAA20': '1C.NOAA20.ATMS.*',   
                '1C-ATMS-NPP': '1C.NPP.ATMS.*'}
    return GPM_dict 
       
def GPM_PMW_2A_GPROF_RS_pattern_dict(): 
   """Return the filename pattern* associated to GPM PMW GPROF 2A products."""
   GPM_dict = { # Using ERA-I as environment ancillary data
                '2A-GMI-CLIM':  '2A-CLIM.GPM.GMI.*',
                '2A-TMI-CLIM': '2A-CLIM.TMI.TRMM.*',
                '2A-SSMI-F11-CLIM': '2A-CLIM.F11.SSMIS.*',
                '2A-SSMI-F13-CLIM': '2A-CLIM.F13.SSMIS.*',
                '2A-SSMI-F14-CLIM': '2A-CLIM.F14.SSMIS.*',
                '2A-SSMI-F15-CLIM': '2A-CLIM.F15.SSMIS.*',
                '2A-SSMI-F16-CLIM': '2A-CLIM.F16.SSMIS.*',
                '2A-SSMI-F17-CLIM': '2A-CLIM.F17.SSMIS.*',
                '2A-SSMI-F18-CLIM': '2A-CLIM.F18.SSMIS.*',
                '2A-ASMR2-GCOMW1-CLIM': '2A-CLIM.GCOMW1.ASMR2.*',
                '2A-AMSRE-AQUA-CLIM': '2A-CLIM.AQUA.AMSRE.*',
                '2A-AMSUB-NOAA15-CLIM': '2A-CLIM.AMSUB.NOAA15.*',
                '2A-AMSUB-NOAA16-CLIM': '2A-CLIM.AMSUB.NOAA16.*',   
                '2A-SAPHIR-MT1' : '2A-CLIM.SAPHIR.MT1.*',  
                '2A-MHS-METOPA-CLIM': '2A-CLIM.METOPA.MHS.*',
                '2A-MHS-METOPB-CLIM': '2A-CLIM.METOPB.MHS.*',
                '2A-MHS-METOPC-CLIM': '2A-CLIM.METOPC.MHS.*',
                '2A-MHS-NOAA18-CLIM': '2A-CLIM.NOAA18.MHS.*',   
                '2A-MHS-NOAA19-CLIM': '2A-CLIM.NOAA19.MHS.*',   
                '2A-ATMS-NOAA20-CLIM': '2A-CLIM.NOAA20.ATMS.*',    
                '2A-ATMS-NPP-CLIM': '2A-CLIM.NPP.ATMS.*',
                # Using JMA's GANAL as environment ancillary data 
                '2A-GMI': '2A.GPM.GMI.*',
                '2A-TMI': '2A.TMI.TRMM.*',
                '2A-SSMI-F11': '2A.F11.SSMIS.*',
                '2A-SSMI-F13': '2A.F13.SSMIS.*',
                '2A-SSMI-F14': '2A.F14.SSMIS.*',
                '2A-SSMI-F15': '2A.F15.SSMIS.*',
                '2A-SSMI-F16': '2A.F16.SSMIS.*',
                '2A-SSMI-F17': '2A.F17.SSMIS.*',
                '2A-SSMI-F18': '2A.F18.SSMIS.*',
                '2A-ASMR2-GCOMW1': '2A.GCOMW1.ASMR2.*',
                '2A-AMSRE-AQUA': '2A.AQUA.AMSRE.*',
                '2A-AMSUB-NOAA15': '2A.AMSUB.NOAA15.*',
                '2A-AMSUB-NOAA16': '2A.AMSUB.NOAA16.*',   
                '2A-MHS-METOPA': '2A.METOPA.MHS.*',
                '2A-MHS-METOPB': '2A.METOPB.MHS.*',
                '2A-MHS-METOPB': '2A.METOPB.MHS.*',
                '2A-MHS-METOPC': '2A.METOPC.MHS.*',
                '2A-MHS-NOAA18': '2A.NOAA18.MHS.*',   
                '2A-MHS-NOAA19': '2A.NOAA19.MHS.*',  
                '2A-ATMS-NOAA20': '2A.NOAA20.ATMS.*',    
                '2A-ATMS-NPP': '2A.NPP.ATMS.*'}     
   return GPM_dict 

def GPM_PMW_2A_GPROF_NRT_pattern_dict(): 
   """Return the filename pattern* associated to GPM PMW GPROF 2A products."""
   GPM_dict = { # Using JMA's GANAL as environment ancillary data 
                '2A-GMI': '2A.GPM.GMI.*',
                '2A-SSMI-F16': '2A.F16.SSMIS.*',
                '2A-SSMI-F17': '2A.F17.SSMIS.*',
                '2A-SSMI-F18': '2A.F18.SSMIS.*',
                '2A-ASMR2-GCOMW1': '2A.GCOMW1.ASMR2.*',
                '2A-MHS-METOPB': '2A.METOPB.MHS.*',
                '2A-MHS-METOPC': '2A.METOPC.MHS.*',  
                '2A-MHS-NOAA19': '2A.NOAA19.MHS.*',  
                '2A-ATMS-NOAA20': '2A.NOAA20.ATMS.*',   
                '2A-ATMS-NPP': '2A.NPP.ATMS.*'}     
   return GPM_dict 

def GPM_PMW_2A_PRPS_RS_pattern_dict(): 
   """Return the filename pattern* associated to GPM PMW PRPS 2A products."""
   GPM_dict = { # Using ERA-I as environment ancillary data
                '2A-SAPHIR-MT1--CLIM' : '2A-CLIM.SAPHIR.MT1.*',  
                # Using JMA's GANAL as environment ancillary data 
                '2A-SAPHIR-MT1' : '2A.SAPHIR.MT1.*'}                 
   return GPM_dict 

def GPM_PMW_2A_PRPS_NRT_pattern_dict(): 
   """Return the filename pattern* associated to GPM PMW PRPS 2A products."""
   GPM_dict = {'2A-SAPHIR-MT1' : '2A.SAPHIR.MT1.*'}                 
   return GPM_dict 

def GPM_PMW_RS_pattern_dict(): 
    """Return the filename pattern* associated to all PMW RS products."""
    GPM_dict = GPM_PMW_1B_RS_pattern_dict()    
    GPM_dict.update(GPM_PMW_1C_RS_pattern_dict())
    GPM_dict.update(GPM_PMW_2A_GPROF_RS_pattern_dict())
    GPM_dict.update(GPM_PMW_2A_PRPS_RS_pattern_dict())                      
    return GPM_dict  

def GPM_PMW_NRT_pattern_dict(): 
    """Return the filename pattern* associated to all PMW NRT products."""
    GPM_dict = GPM_PMW_1B_NRT_pattern_dict()    
    GPM_dict.update(GPM_PMW_1C_NRT_pattern_dict())
    GPM_dict.update(GPM_PMW_2A_GPROF_NRT_pattern_dict())
    GPM_dict.update(GPM_PMW_2A_PRPS_NRT_pattern_dict())                      
    return GPM_dict  
##----------------------------------------------------------------------------.
### IMERG
def GPM_IMERG_NRT_pattern_dict(): 
   """Return the filename pattern* associated to GPM IMERG products."""
   GPM_dict = {'IMERG-ER': '3B-HHR-*',  # '3B-HHR-L.MS.MRG.3IMERG*'
               'IMERG-LR': '3B-HHR-*'}
   return GPM_dict  

def GPM_IMERG_RS_pattern_dict(): 
   """Return the filename pattern* associated to GPM IMERG products."""
   GPM_dict = {'IMERG-FR': '3B-HHR-*'}  # '3B-HHR-L.MS.MRG.3IMERG*'
   return GPM_dict 

def GPM_IMERG_pattern_dict(): 
   """Return the filename pattern* associated to GPM IMERG products."""
   GPM_dict = GPM_IMERG_NRT_pattern_dict()
   GPM_dict.update(GPM_IMERG_RS_pattern_dict())
   return GPM_dict  

##----------------------------------------------------------------------------.
### Levels     
def GPM_1B_RS_pattern_dict():
    """Return the filename pattern* associated to GPM 1B RS products."""
    GPM_dict = GPM_DPR_1B_RS_pattern_dict()    
    GPM_dict.update(GPM_PMW_1B_RS_pattern_dict())              
    return GPM_dict  


def GPM_1B_NRT_pattern_dict():
    """Return the filename pattern* associated to GPM 1B NRT products."""
    GPM_dict = GPM_PMW_1B_NRT_pattern_dict() # GPM_DPR_1B_NRT_pattern_dict()               
    return GPM_dict  

def GPM_2A_RS_pattern_dict():
    """Return the filename pattern* associated to GPM 2A RS products."""
    GPM_dict = GPM_DPR_2A_RS_pattern_dict()   
    GPM_dict.update(GPM_PMW_2A_GPROF_RS_pattern_dict())
    GPM_dict.update(GPM_PMW_2A_PRPS_RS_pattern_dict())                
    return GPM_dict  

def GPM_2A_NRT_pattern_dict():
    """Return the filename pattern* associated to GPM 2A NRT products."""
    GPM_dict = GPM_DPR_2A_NRT_pattern_dict()    
    GPM_dict.update(GPM_PMW_2A_GPROF_NRT_pattern_dict())
    GPM_dict.update(GPM_PMW_2A_PRPS_NRT_pattern_dict())                
    return GPM_dict  
#----------------------------------------------------------------------------. 
### RS vs. NRT
def GPM_RS_products_pattern_dict():
    """Return the filename pattern* associated to all GPM RS products."""
    GPM_dict = GPM_IMERG_RS_pattern_dict()    
    GPM_dict.update(GPM_DPR_RS_pattern_dict())
    GPM_dict.update(GPM_PMW_RS_pattern_dict())                  
    return GPM_dict  

def GPM_NRT_products_pattern_dict():
    """Return the filename pattern* associated to all GPM NRT products."""
    GPM_dict = GPM_IMERG_NRT_pattern_dict()    
    GPM_dict.update(GPM_DPR_NRT_pattern_dict())
    GPM_dict.update(GPM_PMW_NRT_pattern_dict())                  
    return GPM_dict  

def GPM_products_pattern_dict():
    """Return the filename pattern* associated to all GPM products."""
    GPM_dict = GPM_NRT_products_pattern_dict()    
    GPM_dict.update(GPM_RS_products_pattern_dict())              
    return GPM_dict  

#----------------------------------------------------------------------------. 
###########################
### Available products ####
###########################
##----------------------------------------------------------------------------. 
### DPR 
def GPM_DPR_1B_RS_products():
    """Provide a list of available GPM DPR 1B-level RS data for download."""
    product_list = list(GPM_DPR_1B_RS_pattern_dict().keys())
    return product_list  

def GPM_DPR_1B_NRT_products():
    """Provide a list of available GPM DPR 1B-level NRT data for download."""
    raise ValueError("NRT data for GPM DPR 1B not available !")

def GPM_DPR_2A_RS_products():
    """Provide a list of available GPM DPR 2A-level RS data for download."""
    product_list = list(GPM_DPR_2A_RS_pattern_dict().keys())
    return product_list  

def GPM_DPR_2A_NRT_products():
    """Provide a list of available GPM DPR 2A-level NRT data for download."""
    product_list = list(GPM_DPR_2A_NRT_pattern_dict().keys())
    return product_list  

def GPM_DPR_2A_ENV_RS_products():
    """Provide a list of available GPM DPR 2A-level ENV RS data for download."""
    product_list = ['2A-ENV-DPR',
                    '2A-ENV-Ka',
                    '2A-ENV-Ku']
    return product_list  

def GPM_DPR_2A_ENV_NRT_products():
    """Provide a list of available GPM DPR 2A-level ENV NRT data for download."""
    raise ValueError("NRT data for GPM DPR 2A-ENV not available !")

def GPM_DPR_RS_products():
    """Provide a list of available GPM DPR RS data for download."""
    product_list = list(GPM_DPR_RS_pattern_dict().keys())
    return product_list  

def GPM_DPR_NRT_products():
    """Provide a list of available GPM DPR NRT data for download."""
    product_list = list(GPM_DPR_NRT_pattern_dict().keys())
    return product_list  
##----------------------------------------------------------------------------.
### PMW   
def GPM_PMW_1B_RS_products():
    """Provide a list of available GPM PMW 1B-level RS data for download."""
    product_list = list(GPM_PMW_1B_RS_pattern_dict().keys())
    return product_list

def GPM_PMW_1B_NRT_products():
    """Provide a list of available GPM PMW 1B-level NRT data for download."""
    product_list = list(GPM_PMW_1B_NRT_pattern_dict().keys())
    return product_list

def GPM_PMW_1C_RS_products():
    """Provide a list of available GPM PMW 1C-level RS data for download."""
    product_list = list(GPM_PMW_1C_RS_pattern_dict().keys())
    return product_list

def GPM_PMW_1C_NRT_products():
    """Provide a list of available GPM PMW 1C-level NRT data for download."""
    product_list = list(GPM_PMW_1C_NRT_pattern_dict().keys())
    return product_list

def GPM_PMW_2A_GPROF_RS_products():
    """Provide a list of available GPM PMW 2A-level GPROF RS data for download."""
    product_list = list(GPM_PMW_2A_GPROF_RS_pattern_dict().keys())
    return product_list

def GPM_PMW_2A_GPROF_NRT_products():
    """Provide a list of available GPM PMW 2A-level GPROF NRT data for download."""
    product_list = list(GPM_PMW_2A_GPROF_NRT_pattern_dict().keys())
    return product_list

def GPM_PMW_2A_PRPS_RS_products():
    """Provide a list of available GPM PMW 2A-level PRPS RS data for download."""
    product_list = list(GPM_PMW_2A_PRPS_RS_pattern_dict().keys())
    return product_list

def GPM_PMW_2A_PRPS_NRT_products():
    """Provide a list of available GPM PMW 2A-level PRPS NRT data for download."""
    product_list = list(GPM_PMW_2A_PRPS_NRT_pattern_dict().keys())
    return product_list

def GPM_PMW_RS_products():
    """Provide a list of available RS GPM PMW data for download."""
    product_list = list(GPM_PMW_RS_pattern_dict().keys())
    return product_list 

def GPM_PMW_NRT_products():
    """Provide a list of available NRT GPM PMW data for download."""
    product_list = list(GPM_PMW_NRT_pattern_dict().keys())
    return product_list 

##----------------------------------------------------------------------------.
### IMERG
def GPM_IMERG_NRT_products():
    """Provide a list of available GPM IMERG NRT data for download."""
    product_list = list(GPM_IMERG_NRT_pattern_dict().keys())
    return product_list

def GPM_IMERG_RS_products():
    """Provide a list of available GPM IMERG RS data for download."""
    product_list = list(GPM_IMERG_RS_pattern_dict().keys())
    return product_list

def GPM_IMERG_products():
    """Provide a list of available GPM IMERG data for download."""
    product_list = list(GPM_IMERG_pattern_dict().keys())
    return product_list
##----------------------------------------------------------------------------.
### 1B, 1C, 2A levels
def GPM_1B_RS_products():
    """Provide a list of available GPM 1B-level RS data for download."""
    product_list = list(GPM_1B_RS_pattern_dict().keys())
    return product_list 

def GPM_1B_NRT_products():
    """Provide a list of available GPM 1B-level NRT data for download."""
    product_list = list(GPM_1B_NRT_pattern_dict().keys())
    return product_list 

def GPM_1C_RS_products():
    """Provide a list of available GPM PMW 1C-level RS data for download."""
    product_list = list(GPM_PMW_1C_RS_pattern_dict().keys())
    return product_list 

def GPM_1C_NRT_products():
    """Provide a list of available GPM PMW 1C-level NRT data for download."""
    product_list = list(GPM_PMW_1C_NRT_pattern_dict().keys())
    return product_list 

def GPM_2A_RS_products():
    """Provide a list of available GPM PMW 1C-level RS data for download."""
    product_list = list(GPM_2A_RS_pattern_dict().keys())
    return product_list 

def GPM_2A_NRT_products():
    """Provide a list of available GPM PMW 1C-level NRT data for download."""
    product_list = list(GPM_2A_NRT_pattern_dict().keys())
    return product_list 

##----------------------------------------------------------------------------.
### RS vs. NRT   
def GPM_RS_products():   
    """Provide a list of available GPM RS data for download."""
    return GPM_DPR_RS_products() + GPM_PMW_RS_products() + GPM_IMERG_RS_products() 

def GPM_NRT_products():
    """Provide a list of available GPM NRT data for download."""
    return GPM_DPR_NRT_products() + GPM_PMW_NRT_products() + GPM_IMERG_NRT_products() 

##----------------------------------------------------------------------------.
### ALL       
def GPM_products(product_type=None):
    """
    Provide a list of all/NRT/RS GPM data for download.

    Parameters
    ----------
    product_type : str, optional
        If None (default), provide all products (RS and NRT).
        If 'RS', provide a list of all GPM RS data for download.
        If 'NRT', provide a list of all GPM NRT data for download.
    Returns
    -------
    List
        List of GPM products name.

    """
    if product_type is None: 
        return list(np.unique(GPM_RS_products() + GPM_NRT_products()))
    else:
        if (product_type == 'RS'):
            return(GPM_RS_products())
        if (product_type == 'NRT'):
            return(GPM_NRT_products())
        else:
            raise ValueError("Please specify 'product_type' either 'RS' or 'NRT'")
#-----------------------------------------------------------------------------.
###############################
### Code for data download ####
###############################
def get_GPM_disk_directory(base_DIR,
                           product,
                           product_type, 
                           Date, 
                           GPM_version=6):
    """Retrieve directory path where to save GPM data."""
    GPM_folder_name = "GPM_V" + str(GPM_version)
    if (product_type == 'RS'):
        if product in GPM_PMW_RS_products():
            product_type_folder = 'PMW_RS'
        elif product in GPM_IMERG_RS_products():
            product_type_folder = 'IMERG_RS'
        elif product in GPM_DPR_RS_products():
            product_type_folder = 'DPR_RS'
        else:
            raise ValueError("If this error appear, BUG when checking product")
    elif (product_type == 'NRT'):
        if product in GPM_PMW_NRT_products():
            product_type_folder = 'PMW_NRT'
        elif product in GPM_IMERG_NRT_products():
            product_type_folder = 'IMERG'
        elif product in GPM_DPR_NRT_products():
            product_type_folder = 'DPR_NRT'
        else:
            raise ValueError("If this error appear, BUG when checking product")
    else:
       raise ValueError("If this error appear, BUG when checking product_type") 
        
    DIR = os.path.join(base_DIR, 
                       GPM_folder_name, 
                       product_type_folder, 
                       product, 
                       Date.strftime('%Y'), Date.strftime('%m'), Date.strftime('%d'))
    return(DIR)

def get_GPM_PPS_directory(product, 
                          product_type, 
                          Date, 
                          GPM_version=6):
    # return (server_data, url)
    ##------------------------------------------------------------------------.
    ### NRT data 
    if (product_type == 'NRT'):
        if (product not in GPM_NRT_products()):
            raise ValueError("Please specify a valid NRT product: GPM_NRT_products()")
        ## Specify servers 
        server_text = 'https://jsimpsonhttps.pps.eosdis.nasa.gov/text'
        server_data = 'ftp://jsimpsonftps.pps.eosdis.nasa.gov'
        ## Retrieve NASA server folder name for NRT
        # GPM PMW 1B
        if (product in GPM_PMW_1B_NRT_products()):
            folder_name = 'GMI1B'
        # GPM PMW 1C
        elif (product in GPM_1C_NRT_products()):
            if (product == '1C-GMI'):
                folder_name = '/1C/GMI'
            elif (product in ['1C-SSMI-F16','1C-SSMI-F17','1C-SSMI-F18']):
                folder_name = '/1C/SSMIS'
            elif (product == '1C-ASMR2-GCOMW1'): 
                folder_name = '/1C/AMSR2'
            elif (product == '1C-SAPHIR-MT1'): 
                folder_name = '/1C/SAPHIR'     
            elif (product in ['1C-MHS-METOPB', '1C-MHS-METOPC','1C-MHS-NOAA19']):
                folder_name = '/1C/MHS'
            elif (product in ['1C-ATMS-NOAA20', '1C-ATMS-NPP']): 
                folder_name = '1C/ATMS'
            else: 
                raise ValueError('BUG - Some product option is missing.')  
        # GPM PMW 2A GPROF
        elif (product in GPM_PMW_2A_GPROF_NRT_products()):
           if (product == '2A-GMI'):
               folder_name = '/GPROF/GMI'
           elif (product in ['2A-SSMI-F16','2A-SSMI-F17','2A-SSMI-F18']):
               folder_name = '/GPROF/SSMIS'
           elif (product == '2A-ASMR2-GCOMW1'): 
               folder_name = '/GPROF/AMSR2'
           elif (product in ['2A-MHS-METOPB', '2A-MHS-METOPC','2A-MHS-NOAA19']):
               folder_name = '/GPROF/MHS'
           elif (product in ['2A-ATMS-NOAA20', '2A-ATMS-NPP']): 
               folder_name = 'GPROF/ATMS'
           else: 
               raise ValueError('BUG - Some product option is missing.') 
        # GPM PMW 2A PPRS
        elif (product in GPM_PMW_2A_PRPS_NRT_products()):    
            folder_name = 'PRPS'
        # GPM DPR 2A  
        elif (product in GPM_DPR_2A_NRT_products()):
            if (product == '2A-Ku'):
                folder_name = '/radar/KuL2'
            elif (product == '2A-Ka'):
                folder_name = '/radar/KaL2'
            elif (product == '2A-DPR'): 
                folder_name = '/radar/DprL2'
            else:
                raise ValueError('BUG - Some product option is missing.') 
        # GPM IMERG NRT   
        elif (product in GPM_IMERG_NRT_products()):    
            if product == 'IMERG-ER':
               folder_name = 'imerg/early'
            elif product == 'IMERG-LR':
                folder_name = 'imerg/late'
            else: 
                raise ValueError('BUG - Some product option is missing.') 
            # Specify the special url for IMERG NRT products
            url = server_text + '/' + folder_name + '/'+ datetime.datetime.strftime(Date, '%Y%m') + '/'
            return (server_data, url)
        else:
            raise ValueError('BUG - Some product option is missing.') 
        # Specify server url for NRT data
        url = server_text + '/' + folder_name + '/'   
    ##------------------------------------------------------------------------.
    ### RS data      
    elif (product_type == 'RS'):    
        if (product not in GPM_RS_products()):
            raise ValueError("Please specify a valid NRT product: GPM_RS_products()")
        ## Specify servers 
        server_text = 'https://arthurhouhttps.pps.eosdis.nasa.gov/text'
        server_data = 'ftp://arthurhou.pps.eosdis.nasa.gov'
        ## Retrieve NASA server folder name for RS
        # GPM DPR 1B (and GMI)
        if product in GPM_1B_RS_products():
            folder_name = '1B'
        # GPM DPR 2A
        elif product in GPM_DPR_2A_RS_products():
            folder_name = 'radar'
        # GPM PMW 2A PRPS
        elif product in GPM_PMW_2A_PRPS_RS_products():
             folder_name = 'prps' 
        # GPM PMW 2A GPROF
        elif product in GPM_PMW_2A_GPROF_RS_products():
             folder_name = 'gprof'   
        # GPM PMW 1C 
        elif product in GPM_PMW_1C_RS_products():
             folder_name = '1C'  
        # GPM IMERG
        elif product == 'IMERG-FR':   
            folder_name = 'imerg' 
        else: 
            raise ValueError('BUG - Some product is missing.')
         # Specify server url for RS data (based on GPM version)
        if (GPM_version == 6): 
            url = server_text + '/gpmdata/' + datetime.datetime.strftime(Date, '%Y/%m/%d') + '/' + folder_name + "/"
        elif (GPM_version >= 4 & GPM_version <= 5):
            GPM_version_str = "V0" + str(int(GPM_version)) 
            url = server_text + '/gpmallversions/' + GPM_version_str + '/'+ datetime.datetime.strftime(Date, '%Y/%m/%d') + '/' + folder_name + "/"     
        else: 
            raise ValueError('Please specify either GPM_version 4, 5 or 6.')
    ##------------------------------------------------------------------------.   
    return (server_data, url)

    
#-----------------------------------------------------------------------------.  
def filter_daily_GPM_files(file_list,
                           product,
                           start_HHMMSS=None,
                           end_HHMMSS=None):
    """
    Filter the daily GPM file list for specific product and daytime period.

    Parameters
    ----------
    file_list : list
        List of filenames for a specific day.
    product : str
        GPM product name. See: GPM_products()
    start_HHMMSS : str or datetime, optional
        Start time. A datetime object or a string in HHMMSS format.
        The default is None (retrieving from 000000)
    end_HHMMSS : str or datetime, optional
        End time. A datetime object or a string in HHMMSS format.
        The default is None (retrieving to 240000)

    Returns
    -------
    Returns a subset of file_list

    """
    #-------------------------------------------------------------------------.
    # Check valid product 
    if product not in GPM_products():
        raise ValueError("Please provide a valid GPM product --> GPM_products()")   
    #-------------------------------------------------------------------------.
    # Check start_HHMMSS 
    if start_HHMMSS is None:
        start_HHMMSS = 0  
    elif isinstance(start_HHMMSS, datetime.datetime):
        start_HHMMSS = int(datetime.datetime.strftime(start_HHMMSS, '%H%M%S'))
    elif isinstance(start_HHMMSS, str):   
        if len(start_HHMMSS) != 6:
            raise ValueError("Please provide start_HHMMSS as HHMMSS format")
        start_HHMMSS = int(start_HHMMSS)
    else: 
        raise ValueError("Please provide start_HHMMSS as HHMMSS string format or as datetime")    
    #-------------------------------------------------------------------------.
    # Check end time 
    if end_HHMMSS is None:
        end_HHMMSS = 240000  
    elif isinstance(end_HHMMSS, datetime.datetime):
        end_HHMMSS = int(datetime.datetime.strftime(end_HHMMSS, '%H%M%S'))
    elif isinstance(end_HHMMSS, str): 
        if len(end_HHMMSS) != 6:
            raise ValueError("Please provide end_HHMMSS as HHMMSS format")
        end_HHMMSS = int(end_HHMMSS)
    else: 
        raise ValueError("Please provide end_HHMMSS as HHMMSS string format or as datetime")    
    #-------------------------------------------------------------------------.
    # Retrieve GPM filename dictionary 
    GPM_dict = GPM_products_pattern_dict()       
    #-------------------------------------------------------------------------. 
    # Subset specific product 
    l_files = str_subset(file_list, GPM_dict[product])
    #-------------------------------------------------------------------------. 
    # Subset specific time period     
    # - Retrieve start_HHMMSS and endtime of GPM granules products (execept JAXA 1B reflectivities)
    if product not in ['1B-Ka', '1B-Ku']:
        l_s_HHMMSS = str_sub(str_extract(l_files,"S[0-9]{6}"), 1)
        l_e_HHMMSS = str_sub(str_extract(l_files,"E[0-9]{6}"), 1)
    # - Retrieve start_HHMMSS and endtime of JAXA 1B reflectivities
    else: 
        # Retrieve start_HHMMSS of granules   
        l_s_HHMM = str_sub(str_extract(l_files,"[0-9]{10}"),6) 
        l_e_HHMM = str_sub(str_extract(l_files,"_[0-9]{4}_"),1,5) 
        l_s_HHMMSS = str_pad(l_s_HHMM, width=6, side="right",pad="0")
        l_e_HHMMSS = str_pad(l_e_HHMM, width=6, side="right",pad="0")
    # Subset granules files based on start time and end time 
    l_s_HHMMSS = np.array(l_s_HHMMSS).astype(np.int64)  # to integer 
    l_e_HHMMSS = np.array(l_e_HHMMSS).astype(np.int64)  # to integer 
    idx_select1 = np.logical_and(l_s_HHMMSS <= start_HHMMSS, l_e_HHMMSS > start_HHMMSS)
    idx_select2 = np.logical_and(l_s_HHMMSS >= start_HHMMSS, l_s_HHMMSS < end_HHMMSS)
    idx_select = np.logical_or(idx_select1, idx_select2)
    l_files = np.array(l_files)[idx_select]
    l_files = l_files.tolist()
    return(l_files)

def find_daily_GPM_disk_filepaths(base_DIR, product, Date, 
                                  start_HHMMSS = None, 
                                  end_HHMMSS = None,
                                  product_type = 'RS',
                                  GPM_version = 6):
    """
    Retrieve GPM data filepaths for a specific day and product on user disk.
    
    Parameters
    ----------
    base_DIR : str
        The base directory where to store GPM data.
    product : str
        GPM product acronym. See GPM_products()
    Date : datetime
        Single date for which to retrieve the data.
    start_HHMMSS : str or datetime, optional
        Start time. A datetime object or a string in HHMMSS format.
        The default is None (retrieving from 000000)
    end_HHMMSS : str or datetime, optional
        End time. A datetime object or a string in HHMMSS format.
        The default is None (retrieving to 240000)
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).    
    GPM_version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'. 
        GPM data readers are currently implemented only for GPM V06.

    Returns
    -------
    list 
        List of GPM data filepaths.
    """
    ##------------------------------------------------------------------------.
    # Retrieve the directory on disk where the data are stored
    DIR = get_GPM_disk_directory(base_DIR = base_DIR, 
                                 product = product, 
                                 product_type = product_type,
                                 Date = Date,
                                 GPM_version = GPM_version)
    ##------------------------------------------------------------------------.
    # Retrieve the file names in the directoy
    filenames = sorted(os.listdir(DIR))
    ##------------------------------------------------------------------------.
    # Filter the GPM daily file list (for product, start_time & end time)
    filenames = filter_daily_GPM_files(filenames, product=product,
                                       start_HHMMSS=start_HHMMSS, 
                                       end_HHMMSS=end_HHMMSS)
    ##------------------------------------------------------------------------.
    # Create the filepath 
    filepaths = [os.path.join(DIR,filename) for filename in filenames]
    return(filepaths)

def find_daily_GPM_PPS_filepaths(username,
                                 product, 
                                 Date, 
                                 start_HHMMSS = None, 
                                 end_HHMMSS = None,
                                 product_type = 'RS',
                                 GPM_version = 6):
    """
    Retrieve GPM data filepaths for NASA PPS server for a specific day and product.
    
    Parameters
    ----------
    product : str
        GPM product acronym. See GPM_products()
    Date : datetime
        Single date for which to retrieve the data.
    start_HHMMSS : str or datetime, optional
        Start time. A datetime object or a string in HHMMSS format.
        The default is None (retrieving from 000000)
    end_HHMMSS : str or datetime, optional
        End time. A datetime object or a string in HHMMSS format.
        The default is None (retrieving to 240000)
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).    
    GPM_version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'. 
        GPM data readers are currently implemented only for GPM V06.

    Returns
    -------
    (url_data_server, file_list) 
        (URL of the server from which to retrieve data, list of GPM filepaths)
    """
    ##------------------------------------------------------------------------.
    # Retrieve server url of NASA PPS
    (url_data_server, url_file_list) = get_GPM_PPS_directory(product = product, 
                                                             product_type = product_type, 
                                                             Date = Date,
                                                             GPM_version = GPM_version)
    #-------------------------------------------------------------------------.
    # Retrieve the name of available file on NASA PPS servers
    # curl -u username:password
    cmd = 'curl -u ' + username + ':' + username + ' -n ' + url_file_list
    args = cmd.split()
    process = subprocess.Popen(args,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout = process.communicate()[0].decode()
    if stdout[0] == '<':
        if verbose is True:
            print('No data available the', datetime.datetime.strftime(Date, "%Y/%m/%d"))
        return []
    else:
        # Retrieve file list 
        file_list = stdout.split() 
    #-------------------------------------------------------------------------.
    # Filter the GPM daily file list (for product, start_time & end time)
    file_list = filter_daily_GPM_files(file_list,
                                       product=product,
                                       start_HHMMSS=start_HHMMSS,
                                       end_HHMMSS=end_HHMMSS)

    return (url_data_server, file_list)

##-----------------------------------------------------------------------------.
def find_GPM_files(base_DIR, 
                   product, 
                   start_time,
                   end_time,
                   product_type = 'RS',
                   GPM_version = 6):
    """
    Retrieve filepath of GPM data on user disk.
    
    Parameters
    ----------
    base_DIR : str
       The base directory where GPM data are stored.
    product : str
        GPM product acronym.
    start_time : datetime
        Start time.
    end_time : datetime
        End time.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).    
    GPM_version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'. 
        GPM data readers are currently implemented only for GPM V06.
        
    Returns
    -------
    List of filepaths of GPM data.

    """
    # Check start_time and end_time are chronological
    if (start_time > end_time):
        raise ValueError('Provide start_time occuring before of end_time')
    # Retrieve sequence of Dates 
    Dates = [start_time + timedelta(days=x) for x in range(0, (end_time-start_time).days + 1)]
    # Retrieve start and end HHMMSS
    start_HHMMSS = datetime.datetime.strftime(start_time,"%H%M%S")
    end_HHMMSS = datetime.datetime.strftime(end_time,"%H%M%S")
    if (end_HHMMSS == '000000'):
        end_HHMMSS == '240000'
    #-------------------------------------------------------------------------.
    # Case 1: Retrieve just 1 day of data 
    if (len(Dates)==1):
        filepaths = find_daily_GPM_disk_filepaths(base_DIR = base_DIR, 
                                                  GPM_version = GPM_version,
                                                  product = product,
                                                  product_type = product_type,
                                                  Date = Dates[0], 
                                                  start_HHMMSS = start_HHMMSS,
                                                  end_HHMMSS = end_HHMMSS)
    #-------------------------------------------------------------------------.
    # Case 2: Retrieve multiple days of data
    if (len(Dates) > 1):
        filepaths = find_daily_GPM_disk_filepaths(base_DIR = base_DIR, 
                                                  GPM_version = GPM_version,
                                                  product = product,
                                                  product_type = product_type,
                                                  Date = Dates[0], 
                                                  start_HHMMSS = start_HHMMSS,
                                                  end_HHMMSS = '240000')
        if (len(Dates) > 2):
            for Date in Dates[1:-1]:
                filepaths.extend(find_daily_GPM_disk_filepaths(base_DIR=base_DIR,
                                                               GPM_version = GPM_version,
                                                               product=product,
                                                               product_type = product_type,
                                                               Date=Date, 
                                                               start_HHMMSS='000000',
                                                               end_HHMMSS='240000')
                                 )
        filepaths.extend(find_daily_GPM_disk_filepaths(base_DIR = base_DIR,
                                                       GPM_version = GPM_version,
                                                       product = product,
                                                       product_type = product_type,
                                                       Date = Dates[-1], 
                                                       start_HHMMSS='000000',
                                                       end_HHMMSS=end_HHMMSS)
                         )
    #-------------------------------------------------------------------------. 
    return(filepaths)
##-----------------------------------------------------------------------------.






#-----------------------------------------------------------------------------.
def check_which_GPM_file_exist(DIR, PPS_filepaths, force_download=False):
    """
    Check if the file to be retrieve from NASA PPS server already exists on disk.

    Parameters
    ----------
    DIR : str
        GPM directory on disk for a specific product and date.
    PPS_filepaths : str
        Filepaths on which GPM data are stored on PPS servers.
    force_download : boolean, optional
        Whether to redownload data if already existing on disk. The default is False.

    Returns
    -------
    PPS_filepaths.
        Filepaths of the files to download from PPS servers

    """
    # Create directory if does not exist
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    #-------------------------------------------------------------------------.    
    # Check if data already exists 
    if force_download is False: 
        # Retrieve filepath on user disk 
        filepaths = [os.path.join(DIR, os.path.basename(file)) for file in PPS_filepaths]
        # Get index which do not exist
        idx_not_existing = [not os.path.exists(filepath) for filepath in filepaths]
        # Select filepath not existing on disk
        PPS_filepaths = subset_list_by_boolean(PPS_filepaths, idx_not_existing)
    return(PPS_filepaths)
     
    
#----------------------------------------------------------------------------.
# Download of GPM data from NASA servers 
def download_daily_GPM_data(base_DIR,
                            username,
                            product,
                            Date,    
                            start_HHMMSS = None,
                            end_HHMMSS = None,
                            product_type = 'RS',
                            GPM_version = 6,
                            n_parallel = 10,
                            force_download=False,
                            verbose=True):
    """
    Download GPM data from NASA servers using curl.

    Parameters
    ----------
    base_DIR : str
        The base directory where to store GPM data.
    username: str
        Email address with which you registered on on NASA PPS
    product : str
        GPM product name. See: GPM_products()
    Date : datetime
        Single date for which to retrieve the data.
    start_HHMMSS : str or datetime, optional
        Start time. A datetime object or a string in HHMMSS format.
        The default is None (retrieving from 000000)
    end_HHMMSS : str or datetime, optional
        End time. A datetime object or a string in HHMMSS format.
        The default is None (retrieving to 240000)
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).    
    GPM_version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'. 
        GPM data readers are currently implemented only for GPM V06.
    username : str, optional
        Provide your email for login on GPM NASA servers. 
        Temporary default is "gionata.ghiggi@epfl.ch".
    n_parallel : int, optional
        Number of parallel downloads. The default is set to 10.
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
    # Check Date is datetime 
    if not isinstance(Date, (datetime.date, datetime.datetime)):
        raise ValueError("Date must be a datetime object")
    # Check just a single product is provided 
    if not (isinstance(product, str)):
        raise ValueError("'product' must be a single string")   
    # Check product type 
    if (product_type not in ['RS','NRT']):
        raise ValueError("'product_type' must be either 'RS' or 'NRT'") 
    #-------------------------------------------------------------------------.
    ## Retrieve the list of files available on NASA PPS server
    (server_url, filepaths) = find_daily_GPM_PPS_filepaths(username = username,
                                                           product = product, 
                                                           product_type = product_type,
                                                           GPM_version = GPM_version,
                                                           Date = Date, 
                                                           start_HHMMSS = start_HHMMSS, 
                                                           end_HHMMSS = end_HHMMSS)
    #-------------------------------------------------------------------------.
    ## If no file to retrieve on NASA PPS, return 0
    if (len(filepaths) == 0):
        return 0 
    #-------------------------------------------------------------------------.   
    ## Define disk directory where to story the data 
    DIR = get_GPM_disk_directory(base_DIR = base_DIR, 
                                 product = product, 
                                 product_type = product_type,
                                 Date = Date,
                                 GPM_version = GPM_version)
    #-------------------------------------------------------------------------.
    ## If force_download is False, select only data not present on disk 
    # - Here also create DIR if does not exist already
    filepaths = check_which_GPM_file_exist(DIR = DIR, 
                                           PPS_filepaths = filepaths, 
                                           force_download = force_download)
    #-------------------------------------------------------------------------.
    ## Download the data (in parallel)
    # - Wait all n_parallel jobs ended before restarting download
    # - TODO: change to max synchronous n_jobs with multiprocessing
    process_list = []
    process_idx = 0
    if (len(filepaths) >= 1):
        for filepath in filepaths:
            process = curl_download(server = server_url,
                                    filepath = filepath, 
                                    DIR = DIR,
                                    username = username,
                                    password = username)
            process_list.append(process)
            process_idx = process_idx + 1
            # Wait that all n_parallel job ended before restarting downloading 
            if (process_idx == n_parallel):
                [process.wait() for process in process_list]
                process_list = []
                process_idx = 0
        # Before exiting, be sure that download have finished
        [process.wait() for process in process_list]
    return 0

#-----------------------------------------------------------------------------. 
def download_GPM_data(base_DIR,
                      username,
                      product,
                      start_time,
                      end_time,
                      product_type = 'RS',
                      GPM_version = 6):
    """
    Download GPM data from NASA servers.
    
    Parameters
    ----------
    base_DIR : str
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
    GPM_version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'. 
        GPM data readers are currently implemented only for GPM V06.
        
    Returns
    -------
    int 
        0 if everything went fine.

    """  
    # Check start_time and end_time are chronological
    if (start_time > end_time):
        raise ValueError('Provide start_time occuring before of end_time')
    # Retrieve sequence of Dates 
    Dates = [start_time + timedelta(days=x) for x in range(0, (end_time-start_time).days + 1)]
    # Retrieve start and end HHMMSS
    start_HHMMSS = datetime.datetime.strftime(start_time,"%H%M%S")
    end_HHMMSS = datetime.datetime.strftime(end_time,"%H%M%S")
    if (end_HHMMSS == '000000'):
        end_HHMMSS == '240000'
    #-------------------------------------------------------------------------.
    # Case 1: Retrieve just 1 day of data 
    if (len(Dates)==1):
        download_daily_GPM_data(base_DIR = base_DIR,
                                GPM_version =  GPM_version,
                                username = username,
                                product = product,
                                product_type = product_type,
                                Date = Dates[0],  
                                start_HHMMSS = start_HHMMSS,
                                end_HHMMSS = end_HHMMSS)
    #-------------------------------------------------------------------------.
    # Case 2: Retrieve multiple days of data
    if (len(Dates) > 1):
        download_daily_GPM_data(base_DIR = base_DIR, 
                                GPM_version =  GPM_version,
                                username = username,
                                product = product,
                                product_type = product_type,
                                Date = Dates[0],
                                start_HHMMSS = start_HHMMSS,
                                end_HHMMSS = '240000')
        if (len(Dates) > 2):
            for Date in Dates[1:-1]:
                download_daily_GPM_data(base_DIR = base_DIR,
                                        GPM_version =  GPM_version,
                                        username = username,
                                        product = product,
                                        product_type = product_type,
                                        Date = Date, 
                                        start_HHMMSS = '000000',
                                        end_HHMMSS = '240000')
        download_daily_GPM_data(base_DIR=base_DIR, 
                                GPM_version =  GPM_version,
                                username = username,
                                product = product,
                                product_type = product_type,
                                Date = Dates[-1], 
                                start_HHMMSS ='000000',
                                end_HHMMSS = end_HHMMSS)
    #-------------------------------------------------------------------------. 
    print('Download of GPM', product, 'completed')
    return 0
#-----------------------------------------------------------------------------.
#########################################
### Code for querying data from disk ####
#########################################

