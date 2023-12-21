#!/usr/bin/env python3
"""
Created on Mon Aug  3 11:22:04 2020

@author: ghiggi
"""
import os
from importlib.metadata import PackageNotFoundError, version

# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # noqa
import gpm_api.accessor  # noqa
from gpm_api._config import config  # noqa
from gpm_api.configs import define_gpm_api_configs as define_configs  # noqa
from gpm_api.configs import read_gpm_api_configs as read_configs  # noqa
from gpm_api.dataset.dataset import open_dataset  # noqa
from gpm_api.dataset.datatree import open_datatree  # noqa
from gpm_api.dataset.granule import open_granule  # noqa
from gpm_api.io.download import download_archive as download  # noqa
from gpm_api.io.download import (  # noqa
    download_daily_data,
    download_files,
    download_monthly_data,
)
from gpm_api.io.find import find_filepaths as find_files  # noqa
from gpm_api.io.products import (  # noqa
    available_product_categories,
    available_product_levels,
    available_products,
    available_satellites,
    available_scan_modes,
    available_sensors,
    available_versions,
    get_product_end_time,
    get_product_start_time,
)
from gpm_api.utils.checks import (  # noqa
    check_contiguous_scans,
    check_missing_granules,
    check_regular_time,
    check_valid_geolocation,
)
from gpm_api.visualization.plot import plot_labels, plot_patches  # noqa

_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # noqa


# Get version
try:
    __version__ = version("gpm_api")
except PackageNotFoundError:
    # package is not installed
    pass
