#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 21:46:46 2022

@author: ghiggi
"""

import h5py 
import glob
import os
from gpm_api.utils.utils_HDF5 import hdf5_groups_names
from gpm_api.io import download_GPM_data
base_dir = "/tmp/"

start_time = datetime.datetime.strptime("2018-07-01 08:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2018-07-01 09:00:00", '%Y-%m-%d %H:%M:%S')

start_time = datetime.datetime.strptime("2014-07-01 08:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2014-07-01 09:00:00", '%Y-%m-%d %H:%M:%S')

start_time = datetime.datetime.strptime("2004-07-01 08:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2004-07-01 09:00:00", '%Y-%m-%d %H:%M:%S')

products = list(GPM_PMW_1C_RS_pattern_dict())

product_type = 'RS'
version = 7

products = product[0]

#### Download products 
for product in products:
    print(product)
    download_GPM_data(base_dir=base_dir, 
                      username=username,
                      product=product, 
                      product_type=product_type, 
                      version=version, 
                      start_time=start_time,
                      end_time=end_time, 
                      force_download=False, 
                      transfer_tool="curl",
                      progress_bar=True,
                      verbose = True, 
                      n_threads=1)
    
pmw_dir_path = os.path.join(dir_path, f"GPM_V{version}", "PMW_RS")
pmw_products = os.listdir(pmw_dir_path)
pmw_dict = {}
for pmw_product in pmw_products: 
    filepath = glob.glob(os.path.join(pmw_dir_path, pmw_product, "*/*/*/*.HDF5"))[0]
    hdf = h5py.File(filepath, "r") 
    list_groups = hdf5_groups_names(hdf)
    scan_modes = np.unique([s.split("/")[0] for s in list_groups])
    pmw_dict[pmw_product] = scan_modes

pmw_dict
