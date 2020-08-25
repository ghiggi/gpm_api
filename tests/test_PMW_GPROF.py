#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:52:07 2020

@author: ghiggi
"""

# The output is referenced to one of 80 typical structures for
#  each hydrometeor or heating profile. 

# These vertical structures are referenced to as profiles in the output structure. 

# Vertical hydrometeor profiles can be reconstructed to 28 layers by knowing the profile number (i.e. shape) of the profile and a scale factor that is written
# for each pixel.

# GANAL data vs ECMWF (CLIM product)

# sddim
# 21 Number of characters in each species description.
# ntemps
# 12 Number of profile temperature indeces.
# nlyrs
# 28 Number of profiling layers.
# nprf
# 80 Number of unique profiles for each species

## Check pixelStatus for a valid retrieval

def GPM_2A_GPROF_dict(): 
    """Return a dictionary with 2A-GPROF variables information."""
    dict_var = {
    'surfaceTypeIndex': {'path': 'surfaceTypeIndex', 
                      'description': """ Surface type:
                        1 : Ocean
                        2 : Sea-Ice
                        3-7 : Decreasing vegetation
                        8-11 : Decreasing snow cover
                        12 : Standing Water
                        13 : Land/ocean or water Coast
                        14 : Sea-ice edge """,
                      'ndims': 2, 
                      'dims': ['along_track','cross_track'],
                      '_FillValue': -99,
                      'offset_scale': [],
                      'units': '-',
                      'standard_name': 'surfaceTypeIndex'},  
    'sunGlintAngle': {'path': 'sunGlintAngle', 
                      'description': """ The angle between the sun and the instrument view direction as reflected
off the Earth’s surface. sunGlintAngle is the angular separation between the reflected
satellite view vector and the sun vector. When sunGlintAngle is zero, the instrument
views the center of the specular (mirror-like) sun reflection. If this angle is less than ten
degrees, the pixel is affected by sunglint and the pixel’s qualityFlag is lowered to 1. Values
range from 0 to 127 degrees.""",
                      'ndims': 2, 
                      'dims': ['along_track','cross_track'],
                      '_FillValue': -99,
                      'offset_scale': [],
                      'units': '-',
                      'standard_name': 'sunGlintAngle'},
    'probabilityOfPrecip': {'path': 'probabilityOfPrecip', 
                    'description': """A diagnostic variable, in percent, defining the fraction of raining vs. non-raining Database
profiles that make up the final solution. Values range from 0 to 100 percent.""",
                    'ndims': 2, 
                    'dims': ['along_track','cross_track'],
                    '_FillValue': -99,
                    'offset_scale': [],
                    'units': '-',
                    'standard_name': 'probabilityOfPrecip'},
    'surfacePrecipitation': {'path': 'surfacePrecipitation', 
                    'description': 'The instantaneous precipitation rate at the surface',
                    'ndims': 2, 
                    'dims': ['along_track','cross_track'],
                    '_FillValue': -9999.9,
                    'offset_scale': [],
                    'units': 'mm/hr',
                    'standard_name': 'surfacePrecipitation'},
    'frozenPrecipitation': {'path': 'frozenPrecipitation', 
                       'description': 'The instantaneous frozen precipitation rate at the surface',
                       'ndims': 2, 
                       'dims': ['along_track','cross_track'],
                       '_FillValue': -9999.9,
                       'offset_scale': [],
                       'units': 'mm/hr',
                       'standard_name': 'frozenPrecipitation'},
    'convectivePrecipitation': {'path': 'convectivePrecipitation', 
                   'description': 'The instantaneous convective precipitation rate at the surface',
                   'ndims': 2, 
                   'dims': ['along_track','cross_track'],
                   '_FillValue': -9999.9,
                   'offset_scale': [],
                   'units': 'mm/hr',
                   'standard_name': 'convectivePrecipitation'},
    'rainWaterPath': {'path': 'rainWaterPath', 
                      'description': 'Total integrated rain water in the vertical atmospheric column.',
                      'ndims': 2, 
                      'dims': ['along_track','cross_track'],
                      '_FillValue': -9999.9,
                      'offset_scale': [],
                      'units': 'kg/m^2',
                      'standard_name': 'rainWaterPath'},
    'cloudWaterPath': {'path': 'cloudWaterPath', 
                       'description': 'Total integrated cloud liquid water in the vertical atmospheric column.',
                       'ndims': 2, 
                       'dims': ['along_track','cross_track'],
                       '_FillValue': -9999.9,
                       'offset_scale': [],
                       'units': 'kg/m^2',
                       'standard_name': 'cloudWaterPath'},
    'iceWaterPath': {'path': 'iceWaterPath', 
                     'description': 'Total integrated ice water in the vertical atmospheric column.',
                     'ndims': 2, 
                     'dims': ['along_track','cross_track'],
                     '_FillValue': -9999.9,
                     'offset_scale': [],
                     'units': 'kg/m^2',
                     'standard_name': 'iceWaterPath'},
    'mostLikelyPrecipitation': {'path': 'mostLikelyPrecipitation', 
                           'description': 'The surface precipitation value with the closest Tb match within the Bayesian retrieval.',
                           'ndims': 2, 
                           'dims': ['along_track','cross_track'],
                           '_FillValue': -9999.9,
                           'offset_scale': [],
                           'units': 'mm/hr',
                           'standard_name': 'mostLikelyPrecipitation'},
    'pixelStatus': {'path': 'pixelStatus', 
                    'description': """Explains the reason of no retrieval:
0 : Valid pixel
1 : Invalid Latitude / Longitude
2 : Channel Tbs out of range
3 : Surface code / histogram mismatch
4 : Missing TCWV, T2m, or sfccode from preprocessor
5 : No Bayesian Solution""",
                    'ndims': 2, 
                    'dims': ['along_track','cross_track'],
                    '_FillValue': -99,
                    'offset_scale': [],
                    'units': '-',
                    'standard_name': 'pixelStatus'},
    'qualityFlag': {'path': 'qualityFlag', 
                    'description': """Generalized quality of the retrieved pixel:
                         0 : Pixel is "good" and has the highest confidence of the best retrieval.
1 : Use with caution." Pixels can be set to 1 for the following reasons:
    - Sunglint is present, RFI, geolocate, warm load or other L1C ’positive value’ quality warning flags.
    - All sea-ice covered surfaces.
    - All snow covered surfaces.
    - Sensor channels are missing, but not critical ones.
2 : Use pixel with extreme care over snow covered surface."
3 : Use with extreme caution. Critical channels missing for the retrieval. """,
                   'ndims': 2, 
                   'dims': ['along_track','cross_track'],
                   '_FillValue': -99,
                   'offset_scale': [],
                   'units': '-',
                   'standard_name': 'qualityFlag'},
    } # close dictionary here 
    return dict_var

def GPM_2A_PRPS_dict(): 
    """Return a dictionary with 2A-PRPS variables information."""
    dict_var = {
    'surfacePrecipitation': {'path': 'surfacePrecipitation', 
                    'description': 'The instantaneous precipitation rate at the surface',
                    'ndims': 2, 
                    'dims': ['along_track','cross_track'],
                    '_FillValue': -9999.9,
                    'offset_scale': [],
                    'units': 'mm/hr',
                    'standard_name': 'surfacePrecipitation'},
    'qualityFlag': {'path': 'qualityFlag', 
                    'description': """Generalized quality of the retrieved pixel:
                        0 : All is OK
                        1 : Bad Tcs
                        2 : Altitude too high """,
                   'ndims': 2, 
                   'dims': ['along_track','cross_track'],
                   '_FillValue': -999,
                   'offset_scale': [],
                   'units': '-',
                   'standard_name': 'qualityFlag'},
    } # close dictionary here 
    return dict_var
##---------------------------------------------------------------------------.
import os
import numpy as np
import yaml
import datetime
os.chdir('/home/ghiggi/gpm_api') 
from gpm_api.io import GPM_PMW_2A_GPROF_RS_products
from gpm_api.io import find_GPM_files
from gpm_api.dataset import initialize_scan_modes
from gpm_api.dataset import GPM_variables_dict, GPM_variables
##----------------------------------------------------------------------------.
# Write the PRPS YAML files 
GPM_version = 6
products = GPM_PMW_2A_PRPS_RS_products()
product = '2A-SAPHIR-MT1'
product_type = 'RS'

for product in products:
    scan_modes = initialize_scan_modes(product)
    for scan_mode in scan_modes:
        dict_file = GPM_2A_PRPS_dict()
        filename = "GPM_V" + str(GPM_version) + "_" + product + "_" + scan_mode
        filepath = '/home/ghiggi/gpm_api/gpm_api/CONFIG/' + filename + '.yaml' 
        with open(filepath, 'w') as file:
            documents = yaml.dump(dict_file, file)
##----------------------------------------------------------------------------.
# Write the GPROF YAML files 
GPM_version = 6
products = GPM_PMW_2A_GPROF_RS_products()
product = '2A-GMI'
product_type = 'RS'

for product in products:
    scan_modes = initialize_scan_modes(product)
    for scan_mode in scan_modes:
        dict_file = GPM_2A_GPROF_dict()
        filename = "GPM_V" + str(GPM_version) + "_" + product + "_" + scan_mode
        filepath = '/home/ghiggi/gpm_api/gpm_api/CONFIG/' + filename + '.yaml' 
        with open(filepath, 'w') as file:
            documents = yaml.dump(dict_file, file)
                        
GPM_variables_dict(product, scan_mode)            
GPM_variables(product)   
##----------------------------------------------------------------------------.  
# Find file paths 
base_DIR = '/home/ghiggi/tmp'            
start_time = datetime.datetime.strptime("2014-08-09 00:00:00", '%Y-%m-%d %H:%M:%S')
end_time = datetime.datetime.strptime("2014-08-09 03:00:00", '%Y-%m-%d %H:%M:%S')
filepaths = find_GPM_files(base_DIR = base_DIR,
                           product = product, 
                           start_time = start_time,
                           end_time = end_time,
                           product_type = product_type,
                           GPM_version = GPM_version)            
##----------------------------------------------------------------------------.  
# Test data loading to xarray  
import h5py 
from gpm_api.utils.utils_HDF5 import hdf5_datasets_names, hdf5_file_attrs
from gpm_api.dataset import GPM_variables 
from gpm_api.dataset import GPM_granule_Dataset, GPM_Dataset, read_GPM

filepath = filepaths[0]
hdf = h5py.File(filepath,'r') # h5py._hl.files.File
hdf5_datasets_names(hdf)
hdf5_file_attrs(hdf)       
scan_mode = initialize_scan_modes(product)[0]       
variables = GPM_variables(product)      
bbox = None

ds = GPM_granule_Dataset(hdf=hdf,
                         product = product, 
                         scan_mode = scan_mode,  
                         variables = variables,
                         enable_dask=True, chunks='auto')
ds
ds = GPM_Dataset(base_DIR = base_DIR,
                 product = product, 
                 scan_mode = scan_mode,  
                 variables = variables,
                 start_time = start_time,
                 end_time = end_time,
                 bbox = bbox, enable_dask = True, chunks = 'auto')

##----------------------------------------------------------------------------.  
# Test loading of all 2AGPROF products 
products = ['2A-GMI-CLIM', 
            '2A-SSMI-F16-CLIM',
            '2A-SSMI-F17-CLIM',
            '2A-SSMI-F18-CLIM',
            '2A-MHS-METOPA-CLIM',
            '2A-MHS-METOPB-CLIM',
            '2A-MHS-NOAA18-CLIM',
            '2A-MHS-NOAA19-CLIM',
            '2A-MHS-METOPA',
            '2A-MHS-METOPB',
            '2A-MHS-NOAA18',
            '2A-MHS-NOAA19',
            '2A-ATMS-NPP']
for product in products:
    print(product)
    scan_modes = initialize_scan_modes(product)
    for scan_mode in scan_modes:
        ds = GPM_Dataset(base_DIR = base_DIR,
                         product = product, 
                         scan_mode = scan_mode,  
                         variables = variables,
                         start_time = start_time,
                         end_time = end_time,
                         bbox = bbox, enable_dask = True, chunks = 'auto')
        print(ds)


PMW = read_GPM(base_DIR = base_DIR,
               product = product,
               start_time = start_time,
               end_time = end_time,
               variables = None, 
               scan_modes = None, 
               GPM_version = 6,
               product_type = 'RS',
               bbox = None, 
               enable_dask = True,
               chunks = 'auto')








##----------------------------------------------------------------------------.       
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.patheffects as PathEffects
import matplotlib.ticker as mticker

from mpl_toolkits.axes_grid1 import AxesGrid

#make figure
f_size = 10
fig = plt.figure(figsize=(1.6*f_size, 0.9*f_size))
#add the map
ax = fig.add_subplot(1, 1, 1,projection=ccrs.PlateCarree())

ax.add_feature(cartopy.feature.OCEAN.with_scale('50m'))
ax.add_feature(cartopy.feature.LAND.with_scale('50m'), edgecolor='black',lw=0.5,facecolor=[0.95,0.95,0.95])
pm = ax.scatter(ds.lon,ds.lat,c=ds.surfacePrecipitation,vmin=12,vmax=50,s=0.1,zorder=2)
plt.colorbar(pm,ax=ax,shrink=0.33)
plt.show()           
##----------------------------------------------------------------------------.       