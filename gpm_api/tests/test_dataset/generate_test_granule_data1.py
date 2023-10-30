#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:04:39 2023

@author: ghiggi
"""
import os
import gpm_api
import shutil

# from tqdm import tqdm
from gpm_api.io.products import (
    available_products,
    available_scan_modes,
)
from gpm_api.io.pps import find_first_pps_granule_filepath
from gpm_api.io.download import _download_files
from gpm_api.tests.utils.hdf5 import create_test_hdf5
from gpm_api import _root_path

LOCAL_DIR_PATH = "/home/ghiggi/GPM_TEST_DATA"
RAW_DIRNAME = "raw"
CUT_DIRNAME = "cut"
PROCESSED_DIRNAME = "processed"

VERSIONS = [7, 6, 5]
PRODUCT_TYPES = ["RS"]

FORCE_DOWNLOAD = False
FORCE_CUT = False
FORCE_PROCESSED = True


gpm_api.config.set(
    {
        "warn_non_contiguous_scans": False,
        "warn_non_regular_timesteps": False,
        "warn_invalid_spatial_coordinates": False,
    }
)

# Debug
VERSIONS = [7, 6]


## Test directory structure
# .../gpm_api/tests/data/granules/
# <CUT>/<product_type>/<version>/<product>/<filename.hdf5>)
# <PROCESSED>/<product_type>/<version>/<product>/<scan_mode>.nc)

###############################################################################
#### Create the test granules directory
local_granules_dir_path = os.path.join(LOCAL_DIR_PATH)
os.makedirs(local_granules_dir_path, exist_ok=True)

#### Prepare the test data
for product_type in PRODUCT_TYPES:
    for version in VERSIONS:
        version_str = "V" + str(version)
        for product in available_products(product_type="RS", version=version):
            product_info = f"{product_type} {product} {version_str} product"

            # Retrieve a PPS filepath
            try:
                pps_filepath = find_first_pps_granule_filepath(
                    product=product, product_type=product_type, version=version
                )
            except Exception as e:
                print(e)
                continue

            # Retrieve filename
            filename = os.path.basename(pps_filepath)
            # Define product dir
            product_pattern = os.path.join(product_type, version_str, product)
            # Define RAW filepath
            raw_dir = os.path.join(local_granules_dir_path, RAW_DIRNAME, product_pattern)
            raw_filepath = os.path.join(raw_dir, filename)

            # Download file
            if FORCE_DOWNLOAD or not os.path.exists(raw_filepath):
                print(f"Download {product_info}")
                # Download raw data
                _ = _download_files(
                    src_fpaths=[pps_filepath],
                    dst_fpaths=[raw_filepath],
                    storage="pps",
                    transfer_tool="wget",
                    verbose=True,
                )

            # Cut the granule
            print(f"Cutting {product_info}")
            cut_dir_path = os.path.join(local_granules_dir_path, CUT_DIRNAME, product_pattern)
            cut_filepath = os.path.join(cut_dir_path, filename)
            os.makedirs(cut_dir_path, exist_ok=True)
            if FORCE_CUT or not os.path.exists(cut_filepath):
                try:
                    create_test_hdf5(raw_filepath, cut_filepath)
                except Exception as e:
                    print(f"Failed to cut {product_info}: {e}")
                    continue

            # Create the processed netCDF
            print(f"Create {product_info} netCDF")
            scan_modes = available_scan_modes(product=product, version=version)
            processed_dir_path = os.path.join(
                local_granules_dir_path, PROCESSED_DIRNAME, product_pattern
            )
            os.makedirs(processed_dir_path, exist_ok=True)

            for scan_mode in scan_modes:
                processed_filepath = os.path.join(processed_dir_path, f"{scan_mode}.nc")
                if FORCE_PROCESSED or not os.path.exists(processed_filepath):
                    if os.path.exists(processed_filepath):
                        os.remove(processed_filepath)
                    try:
                        ds = gpm_api.open_granule(cut_filepath, scan_mode=scan_mode)
                        ds.to_netcdf(processed_filepath)
                    except Exception as e:
                        print(f"Failed to process {product_info} with scan mode {scan_mode}: {e}")


####---------------------------------------------------------------------------.
#### Move data from LOCAL directory to REPO directory
repo_granules_dir_path = os.path.join(_root_path, "gpm_api", "tests", "data", "granules")
repo_granules_dir_path = os.path.join("/home/ghiggi/GPM_TEST_DATA_DEMO")
os.makedirs(repo_granules_dir_path, exist_ok=True)

# Move CUT directory
local_cut_dir = os.path.join(local_granules_dir_path, CUT_DIRNAME)
repo_cut_dir = os.path.join(repo_granules_dir_path, CUT_DIRNAME)
shutil.copytree(local_cut_dir, repo_cut_dir)

# Move PROCESSED directory
local_cut_dir = os.path.join(local_granules_dir_path, PROCESSED_DIRNAME)
repo_cut_dir = os.path.join(repo_granules_dir_path, PROCESSED_DIRNAME)
shutil.copytree(local_cut_dir, repo_cut_dir)
