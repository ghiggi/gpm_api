#!/usr/bin/env python3
"""
Created on Mon Aug  3 11:22:04 2020

@author: ghiggi
"""
# import os
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # noqa
import gpm_api.accessor  # noqa
from importlib.metadata import PackageNotFoundError, version

from gpm_api.configs import define_gpm_api_configs as define_configs
from gpm_api.configs import read_gpm_api_configs as read_configs
from gpm_api.dataset.reader import (
    open_dataset,
    open_granule,
)
from gpm_api.io.disk import find_filepaths as find_files
from gpm_api.io.download import download_data as download
from gpm_api.io.products import available_products
from gpm_api.io.scan_modes import available_scan_modes
from gpm_api.utils.archive import download_daily_data, download_monthly_data
from gpm_api.utils.checks import (
    check_contiguous_scans,
    check_missing_granules,
    check_regular_time,
    check_valid_geolocation,
)

__all__ = [
    "define_configs",
    "read_configs",
    "available_products",
    "available_scan_modes",
    "download",
    "download_daily_data",
    "download_monthly_data",
    "find_files",
    "open_granule",
    "open_dataset",
    "check_regular_time",
    "check_contiguous_scans",
    "check_valid_geolocation",
    "check_missing_granules",
]

# Get version
try:
    __version__ = version("gpm_api")
except PackageNotFoundError:
    # package is not installed
    pass
