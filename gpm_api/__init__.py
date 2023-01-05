#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 11:22:04 2020

@author: ghiggi
"""
import gpm_api.accessor  # .methods
from gpm_api.io.download import download_data as download
from gpm_api.io.dataset import (
    open_granule,
    open_dataset,
)
from gpm_api.io.disk import find_filepaths as find_files
from gpm_api.utils.checks import check_regular_timesteps
from gpm_api.utils.checks import check_contiguous_scans
from gpm_api.utils.geospatial import check_valid_geolocation
from gpm_api.io.products import available_products
from gpm_api.io.scan_modes import available_scan_modes
