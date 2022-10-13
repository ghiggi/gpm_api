#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 13:52:46 2022

@author: ghiggi
"""
import os
import datetime
import gpm_api
import numpy as np 
import xarray as xp
from dask.diagnostics import ProgressBar
from gpm_api.io import download_GPM_data
from gpm_api.io_future.dataset import open_dataset 

base_dir = "/home/ghiggi"
username = "gionata.ghiggi@epfl.ch"

#### Define analysis time period 
start_time = datetime.datetime.strptime("2020-10-28 08:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2020-10-28 09:00:00", '%Y-%m-%d %H:%M:%S')

products = ['2A-DPR', "2A-GMI", '2B-GPM-CORRA', '2B-GPM-CSH', '2A-GPM-SLH', 
            '2A-ENV-DPR', '1C-GMI']
           
version = 7 
product_type = "RS"

#### Download products 
# for product in products:
#     print(product)
#     download_GPM_data(base_dir=base_dir, 
#                       username=username,
#                       product=product, 
#                       product_type=product_type, 
#                       version = version, 
#                       start_time=start_time,
#                       end_time=end_time, 
#                       force_download=False, 
#                       transfer_tool="curl",
#                       progress_bar=True,
#                       verbose = True, 
#                       n_threads=1)


product = products[0]
groups = None
variables = None 
scan_mode = None
decode_cf=False
chunks="auto"
prefix_group=True
version=7

product_var_dict = {'2A-DPR': ["airTemperature","heightZeroDeg", "precipRate",
                               "precipRateNearSurface","precipRateESurface","precipRateESurface2",
                               "zFactorFinalESurface","zFactorFinalNearSurface",
                               "zFactorFinal", "binEchoBottom", "landSurfaceType"],
                    "2A-GMI": ["rainWaterPath", "surfacePrecipitation", 
                               "cloudWaterPath", "iceWaterPath"],
                    '2B-GPM-CORRA': ["precipTotRate", "precipTotWaterCont",
                                     "cloudIceWaterCont", "cloudLiqWaterCont", 
                                     "nearSurfPrecipTotRate", "estimSurfPrecipTotRate",
                                     # "OEestimSurfPrecipTotRate", "OEsimulatedBrightTemp",
                                     # "OEcolumnCloudLiqWater", "OEcloudLiqWaterCont", "OEcolumnWaterVapor"],
                                     # lowestClutterFreeBin, surfaceElevation
                                    ],
                    '2B-GPM-CSH': ["latentHeating", "surfacePrecipRate"],
                    '2A-GPM-SLH': ["latentHeating", "nearSurfacePrecipRate"],
                    '2A-ENV-DPR': ["cloudLiquidWater", "waterVapor", "airPressure"],
                    '1C-GMI': ["Tc", "Quality"]
                    } 
                                     

dict_product = {}
# product, variables = list(product_var_dict.items())[0]
# product, variables = list(product_var_dict.items())[2]
for product, variables in product_var_dict.items():
    ds = open_dataset(base_dir=base_dir,
                     product=product, 
                     start_time=start_time,
                     end_time=end_time,
                     # Optional 
                     variables=variables, 
                     groups=groups,  
                     scan_mode=scan_mode,
                     version=version,
                     product_type=product_type,
                     chunks="auto",
                     decode_cf = True, 
                     prefix_group = False)
    dict_product[product] = ds             

# GPM CORRA and GPM DPR same coords ;) 
np.testing.assert_equal(dict_product['2A-DPR'].lon.data, dict_product['2B-GPM-CORRA'].lon.data)

np.unique(dict_product['2A-DPR']["precipRateESurface2"].data.compute())
np.unique(dict_product['2B-GPM-CORRA']["nearSurfPrecipTotRate"].data.compute()) # only -9999 
np.unique(dict_product['2B-GPM-CORRA']["estimSurfPrecipTotRate"].data.compute()) # only -9999 
np.unique(dict_product['2A-GMI']["surfacePrecipitation"].data.compute()) # only -9999 

# Check difference between products 
np.set_printoptions(suppress=True)

diff = dict_product['2A-DPR']["precipRateESurface2"] - dict_product['2B-GPM-CORRA']["nearSurfPrecipTotRate"]
diff = dict_product['2A-DPR']["precipRateESurface2"] - dict_product['2A-DPR']["precipRateESurface"]

diff = diff.compute()
np.unique(diff.data.flatten().round(1), return_counts=True)
plt.hist(diff.data.flatten(), bins=100, range=[0,10])
plt.xlim([0,300])

 
plot_product_var_dict = {'2A-DPR': ["precipRateNearSurface",
                                    # "precipRateESurface",
                                    # "precipRateESurface2"
                                    ],
                         '2B-GPM-CORRA': ["nearSurfPrecipTotRate", 
                                           # "estimSurfPrecipTotRate"
                                           ],
                         "2A-GMI": ["surfacePrecipitation"],                
                         } 

product = "2A-DPR"
variable = "precipRateESurface2"
bbox = [-110, -70, 18, 32]
bbox_extent = [-100,-85, 18, 32]

#-----------------------------------------------------------------------------.
    
product = "2B-GPM-CORRA"
variable = "nearSurfPrecipTotRate"
da_subset = dict_product[product][variable].isel(along_track=slice((min(idx_row)), (max(idx_row) + 1)))
da_subset = da_subset.compute()
values = da_subset.data.flatten().round(1)
np.unique(values, return_counts=True) # max: 153.9, min: 0 


product = "2A-DPR"
variable = "precipRateESurface"
da_subset = dict_product[product][variable].isel(along_track=slice((min(idx_row)), (max(idx_row) + 1)))
da_subset = da_subset.compute()
values = da_subset.data.flatten().round(1)
np.unique(values, return_counts=True) # max: 286, min: 0 

#-----------------------------------------------------------------------------.


 


 

 