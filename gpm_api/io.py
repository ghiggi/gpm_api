#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 18:31:08 2020

@author: ghiggi
"""
#----------------------------------------------------------------------------.
######################
### GPM functions ####
######################
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
def get_GPM_file_dict():
    """Return the filename pattern* associated to GPM products."""
    GPM_dict = {'1B-Ka': 'GPMCOR_KAR*',
                '1B-Ku': 'GPMCOR_KUR*',
                #'1B-GMI': '1B.GPM.GMI*', 
                '2A-ENV-DPR': '2A-ENV.GPM.DPR.V*',
                '2A-ENV-Ka': '2A-ENV.GPM.Ka.V*',
                '2A-ENV-Ku': '2A-ENV.GPM.Ku.V*',
                '2A-SLH': '2A.GPM.DPR.GPM-SLH*',
                '2A-DPR': '2A.GPM.DPR.V\d-*',
                '2A-Ka': '2A.GPM.Ka.V*',
                '2A-Ku': '2A.GPM.Ku.V*',
                'IMERG-ER': '3B-HHR-*',  # '3B-HHR-L.MS.MRG.3IMERG*'
                'IMERG-LR': '3B-HHR-*',
                'IMERG-FR': '3B-HHR-*'}
    return GPM_dict 

def get_GPM_directory(base_DIR, product, Date):
    """Retrieve directory path where to save GPM data."""
    DIR = os.path.join(base_DIR, product, Date.strftime('%Y'), Date.strftime('%m'), Date.strftime('%d'))
    return(DIR)
  
#----------------------------------------------------------------------------. 
def GPM_NRT_IMERG_available():
    """Provide a list of available NRT GPM IMERG data for download."""
    product_list = ['IMERG-ER', 
                    'IMERG-LR']
    return product_list   
def GPM_RS_IMERG_available():
    """Provide a list of available RS GPM IMERG data for download."""
    product_list = ['IMERG-FR']
    return product_list    
    
def GPM_RS_1B_available():
    """Provide a list of available RS GPM 1B-level data for download."""
    product_list = ['1B-Ka',
                    '1B-Ku']
                    #'1B-GMI']
    return product_list 

def GPM_RS_2A_available():
    """Provide a list of available RS GPM 2A-level data for download."""
    product_list = ['2A-ENV-DPR',
                    '2A-ENV-Ka',
                    '2A-ENV-Ku',
                    '2A-SLH',
                    '2A-DPR',
                    '2A-Ka',
                    '2A-Ku']
    return product_list  

def GPM_RS_ENV_available():
    """Provide a list of available RS GPM ENV data for download."""
    product_list = ['2A-ENV-DPR',
                    '2A-ENV-Ka',
                    '2A-ENV-Ku']
    return product_list  

def GPM_RS_DPR_available():
    """Provide a list of available RS GPM DPR data for download."""
    product_list = ['1B-Ka',
                    '1B-Ku',
                    '2A-DPR',
                    '2A-Ka',
                    '2A-Ku']
    return product_list  
##----------------------------------------------------------------------------.   
def GPM_RS_available():   
    """Provide a list of available RS GPM data for download."""
    return GPM_RS_1B_available() + GPM_RS_2A_available() + GPM_RS_IMERG_available()

def GPM_NRT_available():
    """Provide a list of available NRT GPM data for download."""
    return GPM_NRT_IMERG_available()

def GPM_IMERG_available():
    """Provide a list of available GPM IMERG data for download."""
    return GPM_NRT_IMERG_available() + GPM_RS_IMERG_available()
    
def GPM_products_available():
    """Provide a list of available GPM data for download."""
    return GPM_RS_available() + GPM_NRT_available()
#-----------------------------------------------------------------------------.
# Filtering of GPM files 
def filter_daily_GPM_file_list(file_list,
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
        GPM product name. See: GPM_products_available()
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
    if product not in GPM_products_available():
        raise ValueError("Please provide a valid GPM product --> GPM_products_available()")   
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
    GPM_dict = get_GPM_file_dict()       
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
  
##----------------------------------------------------------------------------.
# Download of GPM data from NASA servers 
def download_daily_GPM_data(base_DIR,
                            username,
                            product,
                            Date, 
                            start_HHMMSS=None,
                            end_HHMMSS=None,
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
        GPM product name. See: GPM_products_available()
    Date : datetime
        Single date for which to retrieve the data.
    start_HHMMSS : str or datetime, optional
        Start time. A datetime object or a string in HHMMSS format.
        The default is None (retrieving from 000000)
    end_HHMMSS : str or datetime, optional
        End time. A datetime object or a string in HHMMSS format.
        The default is None (retrieving to 240000)
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
        raise ValueError('Date must be a datetime object')
    # Check just a single product is provided 
    if not (isinstance(product, str)):
        raise ValueError('product must be a single string')   
    # Check is product available and select server and filepath    
    if product in GPM_RS_available():            
        server_text = 'https://arthurhouhttps.pps.eosdis.nasa.gov/text'
        server_data = 'ftp://arthurhou.pps.eosdis.nasa.gov'
        flag = "RS"
    elif product in GPM_NRT_available():  
        server_text = 'https://jsimpsonhttps.pps.eosdis.nasa.gov/text'
        server_data = 'ftp://jsimpsonftps.pps.eosdis.nasa.gov'
        flag = "NRT"
    else:
        print('Currently available GPM products:', GPM_products_available())
        raise ValueError('Provide a valid GPM product')
    #-------------------------------------------------------------------------.
    # Retrieve NASA server folder name 
    if product in GPM_RS_2A_available():
        folder_name = 'radar'
    elif product in GPM_RS_1B_available():
        folder_name = '1B'
    elif product == 'IMERG-ER':
        folder_name = 'imerg/early'
    elif product == 'IMERG-LR':
        folder_name = 'imerg/late'
    elif product == 'IMERG-FR':   
        folder_name = 'imerg' 
    else: 
        raise ValueError('BUG - Something missing')
    #-------------------------------------------------------------------------.        
    # Retrieve NASA server url   
    if flag == 'NRT':
        url = server_text + '/' + folder_name + '/'+ datetime.datetime.strftime(Date, '%Y%m') + '/'
    else:    
        url = server_text + '/gpmdata/' + datetime.datetime.strftime(Date, '%Y/%m/%d') + '/' + folder_name + "/"
    #-------------------------------------------------------------------------.
    # Retrieve available filepaths in NASA servers
    # curl -u username:password
    cmd = 'curl -u ' + username + ':' + username + ' -n ' + url
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
    # Filter file list
    file_list = filter_daily_GPM_file_list(file_list,
                                           product=product,
                                           start_HHMMSS=start_HHMMSS,
                                           end_HHMMSS=end_HHMMSS)
    #-------------------------------------------------------------------------.
    # Define directory where to story the data 
    DIR = get_GPM_directory(base_DIR, product, Date)
    # Create directory if does not exist
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    #-------------------------------------------------------------------------.    
    # Check if data already exists 
    if force_download is False: 
        # Retrieve filepath on user disk 
        filepaths = [os.path.join(DIR, os.path.basename(file)) for file in file_list]
        # Get index which do not exist
        idx_not_existing = [not os.path.exists(filepath) for filepath in filepaths]
        # Select filepath on NASA server not existing on user disk
        file_list = subset_list_by_boolean(file_list, idx_not_existing)
    #-------------------------------------------------------------------------.
    # Download the data (in parallel)
    # - Wait all n_parallel jobs ended before restarting download
    # - TODO: change to max synchronous n_jobs with multiprocessing
    process_list = []
    process_idx = 0
    if (len(file_list) >= 1):
        for filepath in file_list:
            process = curl_download(server=server_data,
                                    filepath=filepath, 
                                    DIR=DIR,
                                    username=username,
                                    password=username)
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

##-----------------------------------------------------------------------------.
def find_daily_GPM_filepaths(base_DIR, product, Date, 
                             start_HHMMSS=None, end_HHMMSS=None):
    """
    Retrieve GPM data filepaths for a specific day and product on user disk.
    
    Parameters
    ----------
    base_DIR : str
        The base directory where to store GPM data.
    product : str
        GPM product acronym. See GPM_products_available()
    Date : datetime
        Single date for which to retrieve the data.
    start_HHMMSS : str or datetime, optional
        Start time. A datetime object or a string in HHMMSS format.
        The default is None (retrieving from 000000)
    end_HHMMSS : str or datetime, optional
        End time. A datetime object or a string in HHMMSS format.
        The default is None (retrieving to 240000)

    Returns
    -------
    list 
        List of GPM data filepaths.
    """

    DIR = get_GPM_directory(base_DIR, product, Date)
    filenames = sorted(os.listdir(DIR))
    filenames = filter_daily_GPM_file_list(filenames, product=product,
                                           start_HHMMSS=start_HHMMSS, 
                                           end_HHMMSS=end_HHMMSS)
    filepaths = [os.path.join(DIR,filename) for filename in filenames]
    return(filepaths)
#-----------------------------------------------------------------------------. 
def download_GPM_data(base_DIR,
                      username,
                      product,
                      start_time,
                      end_time):
    """
    Download GPM data from NASA servers.
    
    Parameters
    ----------
    base_DIR : str
        The base directory where to store GPM data.
    username: str
        Email address with which you registered on NASA PPS
    product : str
        GPM product acronym. See GPM_products_available()
    start_time : datetime
        Start time.
    end_time : datetime
        End time.

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
                                username = username,
                                product = product,
                                Date = Dates[0],  
                                start_HHMMSS = start_HHMMSS,
                                end_HHMMSS = end_HHMMSS)
    #-------------------------------------------------------------------------.
    # Case 2: Retrieve multiple days of data
    if (len(Dates) > 1):
        download_daily_GPM_data(base_DIR = base_DIR, 
                                username = username,
                                product = product,
                                Date = Dates[0],
                                start_HHMMSS = start_HHMMSS,
                                end_HHMMSS = '240000')
        if (len(Dates) > 2):
            for Date in Dates[1:-1]:
                download_daily_GPM_data(base_DIR=base_DIR, 
                                        username = username,
                                        product=product,
                                        Date=Date, 
                                        start_HHMMSS='000000',
                                        end_HHMMSS='240000')
        download_daily_GPM_data(base_DIR=base_DIR, 
                                username = username,
                                product=product,
                                Date=Dates[-1], 
                                start_HHMMSS='000000',
                                end_HHMMSS=end_HHMMSS)
    #-------------------------------------------------------------------------. 
    print('Download of GPM', product, 'completed')
    return 0
##----------------------------------------------------------------------------.
def find_GPM_files(base_DIR, 
                   product, 
                   start_time,
                   end_time):
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
        filepaths = find_daily_GPM_filepaths(base_DIR = base_DIR, 
                                             product = product,
                                             Date = Dates[0], 
                                             start_HHMMSS = start_HHMMSS,
                                             end_HHMMSS = end_HHMMSS)
    #-------------------------------------------------------------------------.
    # Case 2: Retrieve multiple days of data
    if (len(Dates) > 1):
        filepaths = find_daily_GPM_filepaths(base_DIR = base_DIR, 
                                            product = product,
                                            Date = Dates[0], 
                                            start_HHMMSS = start_HHMMSS,
                                            end_HHMMSS = '240000')
        if (len(Dates) > 2):
            for Date in Dates[1:-1]:
                filepaths.extend(find_daily_GPM_filepaths(base_DIR=base_DIR, 
                                                         product=product,
                                                         Date=Date, 
                                                         start_HHMMSS='000000',
                                                         end_HHMMSS='240000')
                                 )
        filepaths.extend(find_daily_GPM_filepaths(base_DIR = base_DIR, 
                                                 product = product,
                                                 Date = Dates[-1], 
                                                 start_HHMMSS='000000',
                                                 end_HHMMSS=end_HHMMSS)
                         )
    #-------------------------------------------------------------------------. 
    return(filepaths)
#-----------------------------------------------------------------------------.
