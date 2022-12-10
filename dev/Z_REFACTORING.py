#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 23:02:04 2022

@author: ghiggi
"""
# - SwathDefinition(ds_template['lons'], ds_template['lats']).plot()
# gpm_api lon, lat --> longitude-latitude?
# ds_gpm.gpm_api.pyresample_area.plot()  # property !!!

# -----------------------------------------------------------------------------.
# list: start_time end_time per satellite

# download from filename


# pyresample accessor
# pyresample.area

# -----------------------------------------------------------------------------.
# TODO: download GPM V7 on servers
## Download GPM data after May 21 2018

# -----------------------------------------------------------------------------.
# clean out google colab tutorials

# -----------------------------------------------------------------------------.
#### Investigate chunking of a granule

# -----------------------------------------------------------------------------.
# Solves TODO for dataset.py


# -----------------------------------------------------------------------------.
# TODO: utils/archive: from corrupted fpath, extract product, start_time, end_time, version, and redownload

import numpy as np

granule_ids = [1, 2, 5, 6, 10, 11]

# check_not_duplicate_granules(filepaths)


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


###--------------------------------------------------------------------------.
# TODO for geospatial.py
# TODO: check_valid_geolocation
# TODO: ensure_valid_geolocation (1 spurious pixel)
# TODO: ds_gpm.gpm_api.has_valid_geolocation

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


###--------------------------------------------------------------------------.
### Orbit quality flags
# --> Add as coordinate?
# ScanStatus/dataQuality
# ScanStatus/geoError
# ScanStatus/modeStatus
# ScanStatus/dataWarning
# ScanStatus/operationalMode
# DataQualityFiltering = {'TotalQualityCode' : ['Good'],  # ”Fair” or ”EG”

#### Check non-missing scan
# - Footprint distance ...


##----------------------------------------------------------------------------.
#### Masking functions
# - Masking when NA on other variable
# - Drop scan with FLG/qualityFlag low or bad

# -----------------------------------------------------------------------------.
# dpr
# imerg
# pmw

# gpm = gpm_api.GPM(...)

# PR
# DPR
# PMW
# CMB
# IMERG

# gpm_api.bucket
# gpm_api.geometry (grid, swath)
# gpm_api.sensors

# -----------------------------------------------------------------------------.
#### GPM High-Level Classes (sensors)
# Swath CLASS
# --> TRMM PR
# --> GPM DPR
# --> GPM CMB
# -> pyresample.SwathDef

# Grid CLASS
# --> GPM IMERG
# -> pyresample.AreaDef

# -----------------------------------------------------------------------------.

# 2A-ENV-DPR --> 2A-DPR-ENV ?
# Ka--> KA , Ku --> KU ?

# -----------------------------------------------------------------------------.
