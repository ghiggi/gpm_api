#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 23:02:04 2022

@author: ghiggi
"""

### Refactor patches generator 

# -----------------------------------------------------------------------------.
# Implement valid geolocation checks
# --> In tests/test_dataset_valid_geolocation.py
# --> Use SSMIS, MHS and GMI for testing

# In checks
# - check_valid_geolocation
# - ensure_valid_geolocation (1 spurious pixel)
# - ds_gpm.gpm_api.has_valid_geolocation

# -----------------------------------------------------------------------------.
# Solve TODOs in dataset.py

### Orbit quality flags
# --> Add as coordinate?
# ScanStatus/dataQuality
# ScanStatus/geoError
# ScanStatus/modeStatus
# ScanStatus/dataWarning
# ScanStatus/operationalMode
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
# Refactor geospatial.py

###--------------------------------------------------------------------------.
# Add channels frequency coordinates to PMW 1B and 1C 
# - Derive YAML with frequencies from 1C files 
# --> test/test_pmw_channels_coords.py

# -----------------------------------------------------------------------------.
# pyresample accessor
# - pyresample.area
# - ds_gpm.pyresample.area.plot()  # property !!!

# -----------------------------------------------------------------------------.
# TODO: utils/archive: from corrupted fpath, extract product, start_time, end_time, version, and redownload

import numpy as np

granule_ids = [1, 2, 5, 6, 10, 11]

# check_not_duplicate_granules(filepaths)

# check_missing_granules

def check_consecutive_granules(filepaths, verbose=True):
    from gpm_api.io.info import get_granule_from_filepaths

    # Retrieve granule id from filename
    granule_ids = get_granule_from_filepaths(filepaths)
    # Compute difference in granule id
    diff_ids = np.diff(granule_ids)
    # Check no granule is duplicate
    is_duplicated = diff_ids == 0
    if np.any(is_duplicated):
        # TODO: CLARIFY
        raise ValueError("There are duplicated granules.")
    # Identify occurence of non-consecutive granules
    is_missing = diff_ids > 1
    # If non-consecutive granules occurs, reports the problems
    if np.any(is_missing):
        indices_missing = np.argwhere(is_missing).flatten()
        indices_missing
        list_non_consecutive = [granule_ids[i : i + 2] for i in indices_missing]
        first_non_consecutive = list_non_consecutive[0][0]
        # Display non-regular time interval
        if verbose:
            for start, stop in list_non_consecutive:
                print(f"- Missing data between granule_id {start} and {stop}")
        # Raise error and highligh first non-regular timestep
        raise ValueError(
            f"There are non-regular timesteps starting from granule_id {first_non_consecutive}"
        )


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

