#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 11:22:04 2020

@author: ghiggi
"""
import os 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import gpm_api.accessor  # .methods
from gpm_api.io.download import download_data as download
from gpm_api.io.dataset import (
    open_granule,
    open_dataset,
)
from gpm_api.io.disk import find_filepaths as find_files
from gpm_api.utils.checks import (
    check_regular_time,
    check_contiguous_scans,
    check_valid_geolocation,
    check_missing_granules,
)
from gpm_api.io.products import available_products
from gpm_api.io.scan_modes import available_scan_modes

# Version of the GPM-API package
__version__ = "0.0.1"
