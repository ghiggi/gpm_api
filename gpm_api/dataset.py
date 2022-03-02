#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:50:39 2020

@author: ghiggi
"""
import os 
import h5py
import yaml 
import pandas as pd
import numpy as np 
import xarray as xr
import dask.array

from .io import check_product
from .io import find_GPM_files
from .io import GPM_DPR_RS_products
from .io import GPM_DPR_2A_ENV_RS_products
from .io import GPM_PMW_2A_GPROF_RS_products
from .io import GPM_PMW_2A_PRPS_RS_products
from .io import GPM_IMERG_products
from .io import GPM_products

# For create_GPM_Class
from .DPR.DPR import create_DPR
from .DPR.DPR_ENV import create_DPR_ENV
from .PMW.GMI import create_GMI
from .IMERG.IMERG import create_IMERG

from .utils.utils_HDF5 import hdf5_file_attrs
from .utils.utils_string import str_remove

##----------------------------------------------------------------------------.
def subset_dict(x, keys):
    return dict((k, x[k]) for k in keys)

def remove_dict_keys(x, keys):
    if (isinstance(keys,str)):
        keys = [keys]
    for key in keys:
        x.pop(key, None)
    return
 
def flip_boolean(x):
    # return list(~numpy.array(x))
    # return [not i for i in x]
    # return list(np.invert(x))
    return list(np.logical_not(x))
    
    
def parse_GPM_ScanTime(h):    
    df = pd.DataFrame({'year': h['Year'][:],
                       'month': h['Month'][:],
                       'day': h['DayOfMonth'][:],
                       'hour': h['Hour'][:],
                       'minute': h['Minute'][:],
                       'second': h['Second'][:]})
    return pd.to_datetime(df).to_numpy()
#----------------------------------------------------------------------------.
def GPM_variables_dict(product,
                       scan_mode,
                       GPM_version = 6):
    """
    Return a dictionary with variables information for a specific GPM product.
    ----------
    product : str
        GPM product acronym.
    scan_mode : str
        'NS' = Normal Scan --> For Ku band and DPR 
        'MS' = Matched Scans --> For Ka band and DPR 
        'HS' = High-sensitivity Scans --> For Ka band and DPR
        For products '1B-Ku', '2A-Ku' and '2A-ENV-Ku', specify 'NS'.
        For products '1B-Ka', '2A-Ka' and '2A-ENV-Ka', specify either 'MS' or 'HS'.
        For product '2A-DPR', specify either 'NS', 'MS' or 'HS'.
        For product '2A-ENV-DPR', specify either 'NS' or 'HS'.
        For product '2A-SLH', specify scan_mode = 'Swath'.
        For product 'IMERG-ER','IMERG-LR' and 'IMERG-FR', specify scan_mode = 'Grid'.
    GPM_version : int, optional
        GPM version of the data to retrieve. Only GPM V06 currently implemented.

    Returns
    -------
    dict

    """
    if (product not in GPM_products()):
        raise ValueError('Retrievals not yet implemented for product', product)
    if (GPM_version != 6):     
        raise ValueError('Retrievals currently implemented only for GPM V06.')
    dict_path = os.path.dirname(os.path.abspath(__file__)) + '/CONFIG/' # './CONFIG/'
    filename = "GPM_V" + str(GPM_version) + "_" + product + "_" + scan_mode
    filepath = dict_path + filename + '.yaml'
    with open(filepath) as file:
        d = yaml.safe_load(file)
    return(d)

def GPM_variables(product, scan_modes=None, GPM_version = 6):
    """
    Return a list of variables available for a specific GPM product.

    Parameters
    ----------
    product : str
        GPM product acronym.
    GPM_version : int, optional
        GPM version of the data to retrieve. Only GPM V06 currently implemented.    
    Returns
    -------
    list

    """
    if (scan_modes is None):
        # Retrieve scan modes 
        scan_modes = initialize_scan_modes(product)
    if (isinstance(scan_modes, str)):
        scan_modes = [scan_modes]
    # If more than one, retrieve the union of the variables 
    if (len(scan_modes) > 1):
        l_vars = []
        for scan_mode in scan_modes:
            GPM_vars = list(GPM_variables_dict(product = product, 
                                              scan_mode = scan_mode, 
                                              GPM_version=GPM_version).keys())
            l_vars = l_vars + GPM_vars
        GPM_vars = list(np.unique(np.array(l_vars)))
    else:
        GPM_var_dict = GPM_variables_dict(product = product, 
                                      scan_mode = scan_modes[0], 
                                      GPM_version=GPM_version)
        GPM_vars = list(GPM_var_dict.keys())
    return GPM_vars

#-----------------------------------------------------------------------------.
################
### Classes ####
################
def initialize_scan_modes(product): 
    if (product in ['1B-Ku', '2A-Ku','2A-ENV-Ku']):
        scan_modes = ['NS']
    elif (product in ['1B-Ka', '2A-Ka','2A-ENV-Ka']):    
        scan_modes = ['MS','HS']
    elif (product in ['2A-DPR']):
        scan_modes = ['NS','MS','HS']
    elif product in ['2A-ENV-DPR']:
        scan_modes = ['NS','HS']
    elif (product == "2A-SLH"):
        scan_modes = ['Swath']    
    elif (product in GPM_IMERG_products()):
        scan_modes = ['Grid']
    elif (product in GPM_PMW_2A_GPROF_RS_products()):
        scan_modes = ['S1']
    elif (product in GPM_PMW_2A_PRPS_RS_products()):      
        scan_modes = ['S1']
    else:
        raise ValueError("Retrievals for", product,"not yet implemented")
    return(scan_modes)

def create_GPM_class(base_DIR, product, bbox=None, start_time=None, end_time=None):
    # TODO add for ENV and SLH
    if (product in ['1B-Ka','1B-Ku','2A-Ku','2A-Ka','2A-DPR','2A-SLH']):
        x = create_DPR(base_DIR=base_DIR, product=product,
                       bbox=bbox, start_time=start_time, end_time=end_time)
    elif (product in GPM_IMERG_products()):
        x = create_IMERG(base_DIR=base_DIR, product=product,
                         bbox=bbox, start_time=start_time, end_time=end_time)
    elif (product in GPM_PMW_2A_GPROF_RS_products()):
        x = create_GMI(base_DIR=base_DIR, product=product,
                       bbox=bbox, start_time=start_time, end_time=end_time)
    elif (product in GPM_PMW_2A_PRPS_RS_products()):
        x = create_GMI(base_DIR=base_DIR, product=product,
                       bbox=bbox, start_time=start_time, end_time=end_time)
    elif (product in GPM_DPR_2A_ENV_RS_products()):
        x = create_DPR_ENV(base_DIR=base_DIR, product=product,
                           bbox=bbox, start_time=start_time, end_time=end_time)
    else:
        raise ValueError("Class method for such product not yet implemented")
    return(x)

##----------------------------------------------------------------------------.
###############
### Checks ####
###############
def is_not_empty(x):
    return(not not x)

def is_empty(x):
    return( not x)

def check_GPM_version(GPM_version):
    if (GPM_version != 6): 
        raise ValueError("Only GPM V06 data are currently read correctly")

##----------------------------------------------------------------------------.
def check_scan_mode(scan_mode, product):
    """Checks the validity of scan_mode."""
    # Check that the scan mode is specified if asking for radar data
    if ((scan_mode is None) and (product in ['1B-Ku','1B-Ka','2A-Ku','2A-Ka','2A-DPR',
                                             '2A-ENV-DPR','2A-ENV-Ka','2A-ENV-Ku'])):
        raise ValueError('Please specify a valid scan_mode: NS, MS, HS')
    # Check that a single scan mode is specified 
    if ((scan_mode is not None) and not (isinstance(scan_mode, str))):
        raise ValueError('Specify a single scan_mode at time') 
    # Check that a valid scan mode is specified if asking for radar data
    if (scan_mode is not None):
        if ((product in ['1B-Ku', '2A-Ku','2A-ENV-Ku']) and scan_mode != "NS"):
            raise ValueError("For '1B-Ku','2A-Ku'and '2A-ENV-Ku' products, specify scan_mode = 'NS'")
        if ((product in ['1B-Ka', '2A-Ka','2A-ENV-Ka']) and (scan_mode not in ['MS','HS'])):    
            raise ValueError("For '1B-Ka', '2A-Ka' and '2A-ENV-Ka' products, specify scan_mode either 'MS' or 'HS'")
        if ((product in ['2A-DPR']) and (scan_mode not in ['NS','MS','HS'])):    
            raise ValueError("For '2A-DPR' product, specify scan_mode either 'NS', 'MS' or 'HS'") 
        if ((product in ['2A-ENV-DPR']) and (scan_mode not in ['NS','HS'])):    
            raise ValueError("For '2A-ENV-DPR' products, specify scan_mode either 'NS' or 'HS'") 
    # Specify HDF group name for 2A-SLH and IMERG products
    if (product == "2A-SLH"):
        scan_mode = 'Swath'   
    if (product in GPM_IMERG_products()):
        scan_mode = 'Grid'
    if (product in GPM_PMW_2A_GPROF_RS_products()):
        scan_mode = 'S1'
    if (product in GPM_PMW_2A_PRPS_RS_products()):
        scan_mode = 'S1'
    if (scan_mode is None):
        raise ValueError('scan_mode is still None. This should not occur!')
    return(scan_mode)  

##----------------------------------------------------------------------------.
def check_variables(variables, product, scan_mode, GPM_version=6):
    """Checks the validity of variables."""
    # Make sure variable is a list (if str --> convert to list)     
    if (isinstance(variables, str)):
        variables = [variables]
    # Check variables are valid 
    valid_variables = GPM_variables(product=product,
                                    scan_modes=scan_mode,
                                    GPM_version=GPM_version) 
    idx_valid = [var in valid_variables for var in variables]
    if not all(idx_valid): 
        idx_not_valid = np.logical_not(idx_valid)
        if (all(idx_not_valid)):
            raise ValueError('All variables specified are not valid')
        else:
            variables = list(np.array(variables)[idx_valid])
            # variables_not_valid = list(np.array(variables)[idx_not_valid])
            # raise ValueError('The following variable are not valid:', variables_not_valid)
    ##------------------------------------------------------------------------.    
    # Treat special cases for variables not available for specific products
    # This is now done using the YAML file !
    # 1B products    
    # 2A products
    # if ('flagAnvil' in variables): 
    #     if ((product == '2A-Ka') or (product == '2A-DPR' and scan_mode in ['MS','HS'])): 
    #         # print('flagAnvil available only for Ku-band and DPR NS.\n Silent removal from the request done.')
    #         variables = str_remove(variables, 'flagAnvil')
    # if ('binDFRmMLBottom' in variables):         
    #     if ((product in ['2A-Ka','2A-Ku']) or (product == '2A-DPR' and scan_mode == 'NS')):  
    #         # print('binDFRmMLBottom available only for 2A-DPR.\n Silent removal from the request done.')
    #         variables = str_remove(variables, 'binDFRmMLBottom')
    # if ('binDFRmMLTop' in variables):         
    #     if ((product in ['2A-Ka','2A-Ku']) or (product == '2A-DPR' and scan_mode == 'NS')):  
    #         # print('binDFRmMLTop available only for 2A-DPR.\n Silent removal from the request done.')
    #         variables = str_remove(variables, 'binDFRmMLTop') 
    # if (product == '2A-DPR' and scan_mode == 'MS'): 
    #     variables = str_remove(variables, 'precipRate') 
    #     variables = str_remove(variables, 'paramDSD') 
    #     variables = str_remove(variables, 'phase') 
    ##------------------------------------------------------------------------.  
    # Check that there are still some variables to retrieve
    if (len(variables) == 0):
        raise ValueError('No valid variables to retrieve')        
    return(variables)   

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
        raise ValueError('Provide valid bbox [lon_0, lon_1, lat_0, lat_1]')
    if (bbox[2] > 90 or bbox[2] < -90 or bbox[3] > 90 or bbox[3] < -90):
        raise ValueError('Latitude is defined between -90 and 90')
    # Try to be sure that longitude is specified between -180 and 180        
    if (bbox[0] > 180 or bbox[1] > 180):
        print('bbox should be provided with longitude between -180 and 180')
        bbox[0] = bbox[0] - 180      
        bbox[1] = bbox[1] - 180        
    return(bbox)

#----------------------------------------------------------------------------.     
def GPM_granule_Dataset(hdf, product, variables, 
                        scan_mode = None, 
                        variables_dict = None,
                        GPM_version = 6,
                        bbox=None, enable_dask=True, chunks='auto'):
    """
    Create a lazy xarray.Dataset with relevant GPM data and attributes 
    for a specific granule.   

    Parameters
    ----------
    hdf : h5py.File
        HFD5 object read with h5py.
    scan_mode : str
        'NS' = Normal Scan --> For Ku band and DPR 
        'MS' = Matched Scans --> For Ka band and DPR 
        'HS' = High-sensitivity Scans --> For Ka band and DPR
        For products '1B-Ku', '2A-Ku' and '2A-ENV-Ku', specify 'NS'.
        For products '1B-Ka', '2A-Ka' and '2A-ENV-Ka', specify either 'MS' or 'HS'.
        For product '2A-DPR', specify either 'NS', 'MS' or 'HS'.
        For product '2A-ENV-DPR', specify either 'NS' or 'HS'.
        For product '2A-SLH', specify scan_mode = 'Swath'.
        For product 'IMERG-ER','IMERG-LR' and 'IMERG-FR', specify scan_mode = 'Grid'.
    product : str
        GPM product acronym.                           
    variables : list, str
         Datasets names to extract from the HDF5 file.
         Hint: utils_HDF5.hdf5_datasets_names() to see available datasets.
    variables_dict : dict, optional    
         Expect dictionary from GPM_variables_dict(product, scan_mode)
         Provided to avoid recomputing it at every call.
         If variables_dict is None --> Perform also checks on the arguments.
    GPM_version : int, optional
        GPM version of the data to retrieve. Only GPM V06 currently implemented.         
    bbox : list, optional 
         Spatial bounding box. Format: [lon_0, lon_1, lat_0, lat_1]  
         For radar products it subset only along_track !
    dask : bool, optional
         Wheter to lazy load data (in parallel) with dask. The default is True.
         Hint: xarray’s lazy loading of remote or on-disk datasets is often but not always desirable.
         Before performing computationally intense operations, load the Dataset
         entirely into memory by invoking the Dataset.load() 
    chunks : str, list, optional
        Chunck size for dask. The default is 'auto'.
        Alternatively provide a list (with length equal to 'variables') specifying
        the chunk size option for each variable.
        
    Returns
    -------
    xarray.Dataset

    """   
    ##------------------------------------------------------------------------.
    ## Arguments checks are usually done in GPM _Dataset()       
    if variables_dict is None:
        ## Check valid product 
        check_product(product, product_type=None)
        ## Check scan_mode 
        scan_mode = check_scan_mode(scan_mode=scan_mode, product=product)      
        ## Check variables 
        variables = check_variables(variables=variables, 
                                    product=product,
                                    scan_mode=scan_mode,
                                    GPM_version=GPM_version)   
        ## Check bbox
        bbox = check_bbox(bbox)
        ##--------------------------------------------------------------------.   
        ## Retrieve variables dictionary 
        variables_dict = GPM_variables_dict(product=product, 
                                            scan_mode=scan_mode,
                                            GPM_version = GPM_version)  
    ##------------------------------------------------------------------------.     
    ## Retrieve basic coordinates 
    hdf_attr = hdf5_file_attrs(hdf)
    # - For GPM radar products
    if (product not in GPM_IMERG_products()):
        granule_id = hdf_attr['FileHeader']["GranuleNumber"]
        lon = hdf[scan_mode]['Longitude'][:]
        lat = hdf[scan_mode]['Latitude'][:]
        tt = parse_GPM_ScanTime(hdf[scan_mode]['ScanTime'])
        coords = {'lon': (['along_track','cross_track'],lon),
                  'lat': (['along_track','cross_track'],lat),
                  'time': (['along_track'], tt),
                  'granule_id': (['along_track'], np.repeat(granule_id, len(tt))),
                 }
      
    # - For IMERG products
    else:
        lon = hdf[scan_mode]['lon'][:]
        lat = hdf[scan_mode]['lat'][:]
        tt = hdf5_file_attrs(hdf)['FileHeader']['StartGranuleDateTime'][:-1]  
        tt = np.array(np.datetime64(tt) + np.timedelta64(30, 'm'), ndmin=1)
        coords = {'time': tt,
                  'lon': lon,
                  'lat': lat
                  }
    
    ##------------------------------------------------------------------------.
    ## Check if there is some data in the bounding box
    if (bbox is not None):
        # - For GPM radar products
        if (product not in GPM_IMERG_products()):
            idx_row, idx_col = np.where((lon >= bbox[0]) & (lon <= bbox[1]) & (lat >= bbox[2]) & (lat <= bbox[3]))              
        # - For IMERG products
        else:   
            idx_row = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]    
            idx_col = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]    
        # If no data in the bounding box in current granule, return empty list
        if (idx_row.size == 0 or idx_col.size == 0):
            return(None)
    ##------------------------------------------------------------------------.
    # Retrieve each variable 
    flag_first = True # Required to decide if to create/append to Dataset
    for var in variables: 
        # print(var)
        ##--------------------------------------------------------------------.
        # Prepare attributes for the DataArray 
        dict_attr = subset_dict(variables_dict[var], ['description','units','standard_name'])
        dict_attr['product'] = product
        dict_attr['scan_mode'] = scan_mode
        ##--------------------------------------------------------------------.
        # Choose if using dask 
        if enable_dask is True:
            hdf_obj = dask.array.from_array(hdf[scan_mode][variables_dict[var]['path']], chunks=chunks)
        else:
            hdf_obj = hdf[scan_mode][variables_dict[var]['path']]
        ##--------------------------------------------------------------------.    
        # Create the DataArray
        da = xr.DataArray(hdf_obj,
                          dims = variables_dict[var]['dims'],
                          coords = coords,
                          attrs = dict_attr)
        da.name = var
        ## -------------------------------------------------------------------.
        ## Subsetting based on bbox (lazy with dask)
        if bbox is not None:
            # - For GPM radar products
            # --> Subset only along_track to allow concat on cross_track  
            if (product not in GPM_IMERG_products()):
                da = da.isel(along_track = slice((min(idx_row)),(max(idx_row)+1))) 
            # - For IMERG products
            else: 
                da = da.isel(lon = idx_row, lat = idx_col)
        ## -------------------------------------------------------------------.
        ## Convert to float explicitly (needed?)
        # hdf_obj.dtype  ## int16
        # da = da.astype(np.float)
        ## -------------------------------------------------------------------.
        ## Parse missing values and errors
        da = xr.where(da.isin(variables_dict[var]['_FillValue']), np.nan, da)     
        # for value in dict_attr['_FillValue']:
        #     da = xr.where(da == value, np.nan, da)
        ## -------------------------------------------------------------------.
        ## Add scale and offset 
        if len(variables_dict[var]['offset_scale'])==2:
            da = da/variables_dict[var]['offset_scale'][1] - variables_dict[var]['offset_scale'][0]
        ## --------------------------------------------------------------------.    
        ## Create/Add to Dataset 
        if flag_first is True: 
            ds = da.to_dataset()
            flag_first = False
        else:
            ds[var] = da
        ##--------------------------------------------------------------------. 
        ## Special processing for specific fields     
        if var == 'precipWaterIntegrated':
            ds['precipWaterIntegrated_Liquid'] = ds['precipWaterIntegrated'][:,:,0]
            ds['precipWaterIntegrated_Solid'] = ds['precipWaterIntegrated'][:,:,1]
            ds['precipWaterIntegrated'] = ds['precipWaterIntegrated_Liquid']+ds['precipWaterIntegrated_Solid']
        if var == 'paramDSD':
            ds['DSD_dBNw'] = ds['paramDSD'][:,:,:,0]
            ds['DSD_m'] = ds['paramDSD'][:,:,:,1]
            ds = ds.drop_vars(names='paramDSD')
            # Modify attributes 
            ds['DSD_m'].attrs['units'] = 'mm' 
        if (var == 'flagBB' and product == '2A-DPR'):
            ds['flagBB'].attrs['description'] = ''' Flag for Bright Band: 
                                                    0 : BB not detected
                                                    1 : Bright Band detected by Ku and DFRm
                                                    2 : Bright Band detected by Ku only
                                                    3 : Bright Band detected by DFRm only
                                               '''
        # TODO ENV                                 
        # if (var == 'cloudLiquidWater'):
        #     # nwater , 
        # if (var == 'waterVapor'):
        #     # nwater
        # TODO DECODING  
        # if (var == 'phase'):
            # print('Decoding of phase not yet implemented')
        # if (var == 'typePrecip'): 
            # print('Decoding of typePrecip not yet implemented')    
    ##------------------------------------------------------------------------. 
    # Add optional coordinates 
    # - altitude...
    # - TODO 
    ##------------------------------------------------------------------------.
    # Add other stuffs to dataset    
    return(ds) 
      
def GPM_Dataset(base_DIR,
                product, 
                variables, 
                start_time, 
                end_time,
                scan_mode=None, 
                GPM_version = 6,
                product_type = 'RS',
                bbox=None, enable_dask=False, chunks='auto'):
    """
    Lazily map HDF5 data into xarray.Dataset with relevant GPM data and attributes. 
   
    Parameters
    ----------
    base_DIR : str
       The base directory where GPM data are stored.
    product : str
        GPM product acronym.                           
    variables : list, str
         Datasets names to extract from the HDF5 file.
         Hint: GPM_variables(product) to see available variables.
    start_time : datetime
        Start time.
    end_time : datetime
        End time.
    scan_mode : str, optional
        'NS' = Normal Scan --> For Ku band and DPR 
        'MS' = Matched Scans --> For Ka band and DPR 
        'HS' = High-sensitivity Scans --> For Ka band and DPR
        For products '1B-Ku', '2A-Ku' and '2A-ENV-Ku', specify 'NS'.
        For products '1B-Ka', '2A-Ka' and '2A-ENV-Ka', specify either 'MS' or 'HS'.
        For product '2A-DPR', specify either 'NS', 'MS' or 'HS'.
        For product '2A-ENV-DPR', specify either 'NS' or 'HS'.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).    
    GPM_version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'. 
        GPM data readers are currently implemented only for GPM V06.
    bbox : list, optional 
         Spatial bounding box. Format: [lon_0, lon_1, lat_0, lat_1]  
    dask : bool, optional
         Wheter to lazy load data (in parallel) with dask. The default is True.
         Hint: xarray’s lazy loading of remote or on-disk datasets is often but not always desirable.
         Before performing computationally intense operations, load the Dataset
         entirely into memory by invoking the Dataset.load() 
    chunks : str, list, optional
        Chunck size for dask. The default is 'auto'.
        Alternatively provide a list (with length equal to 'variables') specifying
        the chunk size option for each variable.

    Returns
    -------
    xarray.Dataset

    """
    ##------------------------------------------------------------------------.
    ## Check valid product 
    check_product(product, product_type=product_type)
    ## Check scan_mode 
    scan_mode = check_scan_mode(scan_mode, product)      
    ## Check variables 
    variables = check_variables(variables=variables, 
                                product=product,
                                scan_mode=scan_mode,
                                GPM_version=GPM_version)  
    ## Check bbox
    bbox = check_bbox(bbox)
    ##------------------------------------------------------------------------.
    ## Check for chuncks    
    # TODO smart_autochunck per variable (based on dim...)
    # chunks = check_chuncks(chunks) 
    ##------------------------------------------------------------------------.
    # Find filepaths
    filepaths = find_GPM_files(base_DIR = base_DIR,
                               GPM_version =  GPM_version,
                               product = product, 
                               product_type = product_type,
                               start_time = start_time,
                               end_time = end_time)
    ##------------------------------------------------------------------------.
    # Check that files have been downloaded  on disk 
    if is_empty(filepaths):
        raise ValueError('Requested files are not found on disk. Please download them before')
    ##------------------------------------------------------------------------.
    # Initialize list (to store Dataset of each granule )
    l_Datasets = list() 
    # Retrieve variables dictionary 
    variables_dict = GPM_variables_dict(product=product, 
                                        scan_mode=scan_mode,
                                        GPM_version = GPM_version)   
    for filepath in filepaths:  
        # Load hdf granule file  
        try: 
            hdf = h5py.File(filepath,'r') # h5py._hl.files.File
        except OSError:
            if not os.path.exists(filepath):
                raise ValueError("This is a gpm_api bug. This filepath should not have been included in filepaths.")
            else:
                print("The following file is corrupted and is being removed: {}".format(filepath))
                print("Redownload the file !!!")
                os.remove(filepath)
                continue 
        hdf_attr = hdf5_file_attrs(hdf)
        # --------------------------------------------------------------------.
        ## Decide if retrieve data based on JAXA quality flags 
        # Do not retrieve data if TotalQualityCode not ... 
        if (product in GPM_DPR_RS_products()):
            DataQualityFiltering = {'TotalQualityCode': ['Good']} # TODO future fun args
            if (hdf_attr['JAXAInfo']['TotalQualityCode'] not in DataQualityFiltering['TotalQualityCode']):
                continue
        #---------------------------------------------------------------------.
        # Retrieve data if granule is not empty 
        if (hdf_attr['FileHeader']['EmptyGranule'] == 'NOT_EMPTY'):
            ds = GPM_granule_Dataset(hdf = hdf,
                                     GPM_version =  GPM_version,
                                     product = product, 
                                     scan_mode = scan_mode,  
                                     variables = variables,
                                     variables_dict = variables_dict,
                                     bbox = bbox,
                                     enable_dask = enable_dask, 
                                     chunks = 'auto')
            if ds is not None:
                l_Datasets.append(ds)
    ##-------------------------------------------------------------------------.
    # Concat all Datasets
    if (len(l_Datasets) >= 1):
        if (product in GPM_IMERG_products()):
            ds = xr.concat(l_Datasets, dim="time")
        else:
            ds = xr.concat(l_Datasets, dim="along_track")
        print('GPM Dataset loaded successfully !')    
    else:
        print("No data available for current request. Try for example to modify the bbox.")
        return 
    ##------------------------------------------------------------------------.
    # Return Dataset
    return ds
##----------------------------------------------------------------------------.

def read_GPM(base_DIR,
             product, 
             start_time, 
             end_time,
             variables = None, 
             scan_modes = None, 
             GPM_version = 6,
             product_type = 'RS',
             bbox = None, 
             enable_dask = True,
             chunks = 'auto'):
    """
    Construct a GPM object (DPR,GMI, IMERG) depending on the product specified.
    Map HDF5 data on disk lazily into dask arrays with relevant data and attributes. 
   
    Parameters
    ----------
    base_DIR : str
       The base directory where GPM data are stored.
    product : str
        GPM product acronym.                           
    variables : str, list, optional
         Datasets names to extract from the HDF5 file.
         By default all variables available.
         Hint: GPM_variables(product) to see available variables.
    start_time : datetime
        Start time.
    end_time : datetime
        End time.
    scan_modes : str,list optional
        None --> Default to all the available for the specific GPM product
        'NS' = Normal Scan --> For Ku band and DPR 
        'MS' = Matched Scans --> For Ka band and DPR 
        'HS' = High-sensitivity Scans --> For Ka band and DPR
        For products '1B-Ku', '2A-Ku' and '2A-ENV-Ku', specify 'NS'.
        For products '1B-Ka', '2A-Ka' and '2A-ENV-Ka', specify 'MS' or 'HS'.
        For product '2A-DPR', specify 'NS', 'MS' or 'HS'.
        For product '2A-ENV-DPR', specify either 'NS' or 'HS'.
    product_type : str, optional
        GPM product type. Either 'RS' (Research) or 'NRT' (Near-Real-Time).    
    GPM_version : int, optional
        GPM version of the data to retrieve if product_type = 'RS'. 
        GPM data readers are currently implemented only for GPM V06.
    bbox : list, optional 
         Spatial bounding box. Format: [lon_0, lon_1, lat_0, lat_1]  
    dask : bool, optional
         Wheter to lazy load data (in parallel) with dask. The default is True.
         Hint: xarray’s lazy loading of remote or on-disk datasets is often but not always desirable.
         Before performing computationally intense operations, load the Dataset
         entirely into memory by invoking the Dataset.load() 
    chunks : str, list, optional
        Chunck size for dask. The default is 'auto'.
        Alternatively provide a list (with length equal to 'variables') specifying
        the chunk size option for each variable.

    Returns
    -------
    xarray.Dataset

    """
    ##------------------------------------------------------------------------.
    ## Check GPM version 
    check_GPM_version(GPM_version)
    ## Check product is valid
    check_product(product)
    # Initialize variables if not provided
    if (variables is None):
        variables = GPM_variables(product=product, GPM_version=GPM_version)
    ## Initialize or check the scan_modes  
    if (scan_modes is not None):
        if isinstance(scan_modes, str):
            scan_modes = [scan_modes]
        scan_modes = [check_scan_mode(scan_mode, product) for scan_mode in scan_modes]  
    else: 
        scan_modes = initialize_scan_modes(product) 
    ##------------------------------------------------------------------------.
    # Initialize GPM class
    x = create_GPM_class(base_DIR=base_DIR, product=product,
                         bbox=bbox, start_time=start_time, end_time=end_time)
    # Add the requested scan_mode to the GPM DPR class object
    for scan_mode in scan_modes:
        print("Retrieving", product,scan_mode,"data")
        x.__dict__[scan_mode] = GPM_Dataset(base_DIR = base_DIR,
                                            GPM_version = GPM_version,
                                            product = product, 
                                            product_type = product_type,
                                            variables = variables, 
                                            scan_mode = scan_mode, 
                                            start_time = start_time, 
                                            end_time = end_time,
                                            bbox = bbox, 
                                            enable_dask = enable_dask,
                                            chunks = chunks)
    ##------------------------------------------------------------------------.
    # Return the GPM class object with the requested  GPM data 
    return(x)
    
    
