#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 23:02:04 2022

@author: ghiggi
"""
# -----------------------------------------------------------------------------.
# Implement valid geolocation checks
# --> In tests/test_dataset_valid_geolocation.py
# --> Use SSMIS, MHS and GMI for testing

# In checks
# - check_valid_geolocation
# - ensure_valid_geolocation (1 spurious pixel)
# - ds_gpm.gpm_api.has_valid_geolocation

# check ax is geoaxes or mpl.axes ! 

###---------------------------------------------------------------------------.
# Refactor geospatial.py

###---------------------------------------------------------------------------.
# prefix_group=False to save data to netcdf ! 
# SLV/precipRateESurface: variables with / can not be saved to netcdf ! 

# -----------------------------------------------------------------------------.
# Solve TODOs in dataset.py
# - Add granule_id also to GRID (imerg) dataset

### Orbit quality flags
# --> Add as coordinate?
# ScanStatus/dataQuality
# ScanStatus/geoError
# ScanStatus/modeStatus
# ScanStatus/dataWarning
# ScanStatus/operationalMode

### Granule attributes 
# DataQualityFiltering = {'TotalQualityCode' : ['Good'],  # ”Fair” or ”EG”

#### Masking functions
# - Masking when NA on other variable
# - Drop scan with FLG/qualityFlag low or bad

# -----------------------------------------------------------------------------.
#### Investigate chunking of a granule

# -----------------------------------------------------------------------------.
## Download GPM data after May 21 2018

# - SwathDefinition(ds_template['lons'], ds_template['lats']).plot()
# gpm_api lon, lat --> longitude-latitude?

# -----------------------------------------------------------------------------.
# list: start_time end_time per satellite
# --> dev/list_products.py

# download from filename

###--------------------------------------------------------------------------.
# Add channels frequency coordinates to PMW 1B and 1C 
# - Derive YAML with frequencies from 1C files 
# --> test/test_pmw_channels_coords.py

# -----------------------------------------------------------------------------.
# pyresample accessor
# - pyresample.area
# - ds_gpm.pyresample.area.plot()  # property !!!

# -----------------------------------------------------------------------------.
### Refactor patterns.py and products.py

# -----------------------------------------------------------------------------.
# If data alread download, print a better message than:
# --> Now: Download of available GPM 2A-DPR product completed.
# --> Future: All data already on disk

# import gpm_api
# gpm_api.download(base_dir='/home/ghiggi/GPM',
#                  username="gionata.ghiggi@epfl.ch",
#                  product='2A-DPR',
#                  product_type='RS',
#                  start_time='2020-07-05 00:58:02',
#                  end_time='2020-07-05 01:14:37',
#                  version=7,
#                  force_download=False,
#                  verbose=True)

# -----------------------------------------------------------------------------.
# Improve decoding and DataArray attributes
# --> Decode before or after each Dataset

##----------------------------------------------------------------------------.
#### Test download NRT data

# -----------------------------------------------------------------------------.
# Download set of background .... 
# - Callable from ax.background_img(name='BM', resolution='high')  
# - Add BlueMarble to IMERG example 

# Add PlateCarre backgrounds at /home/ghiggi/anaconda3/envs/satpy39/lib/python3.9/site-packages/cartopy/data/raster/natural_earth
# https://neo.gsfc.nasa.gov/
# https://neo.gsfc.nasa.gov/view.php?datasetId=BlueMarbleNG

# bg_dict = {
#    "__comment__": """JSON file specifying the image to use for a given type/name and
#                      resolution. Read in by cartopy.mpl.geoaxes.read_user_background_images.""",
#   "BM": {
#     "__comment__": "Blue Marble Next Generation, July ",
#     "__source__": "https://neo.sci.gsfc.nasa.gov/view.php?datasetId=BlueMarbleNG-TB",
#     "__projection__": "PlateCarree",
#     "low": "BM_July_low.png",
#     "high": "BM_July_high.png"
#   },
# }

# import json
# fpath = "/home/ghiggi/Backgrounds/images.json"
# with open(fpath, 'w', encoding ='utf8') as f:
#     json.dump(bg_dict, f, allow_nan=False)

# -----------------------------------------------------------------------------.

# gpm_api/bucket
# gpm_api/geometry (grid, swath)
# gpm_api/sensors

# -----------------------------------------------------------------------------.
### Create Jupyter Notebooks
# - Copy to Google Colab and GitHub Codespace
# - Figure out a valid cartopy installation in Colab
# https://colab.research.google.com/drive/14SFtTM5BydEElTgy83F_74-J9MJBCznb?usp=sharing
# https://colab.research.google.com/drive/1vptHQjopOYi0HohHCRqVcmQiWM5xSgZ8?usp=sharing
# https://colab.research.google.com/drive/1OYW2KXvBUT7lexrBXd71njU1zjQsKSQ5?usp=sharing

# -----------------------------------------------------------------------------.
#### GPM High-Level Classes (sensors)
# gpm = gpm_api.GPM(...)

# Swath CLASS
# --> TRMM PR
# --> GPM DPR
# --> GPM CMB
# --> GPM PMW
# -> pyresample.SwathDef

# Grid CLASS
# --> GPM IMERG
# -> pyresample.AreaDef

# -----------------------------------------------------------------------------.