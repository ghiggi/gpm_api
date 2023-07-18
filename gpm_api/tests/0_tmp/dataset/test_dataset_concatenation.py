#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 18:48:09 2023

@author: ghiggi
"""
import os
import glob
import xarray as xr
from gpm_api.dataset.dataset import _open_valid_granules

path_dir = "/home/ghiggi/data/GPM/RS/V07/RADAR/2A-DPR/2021/07/05/"
scan_mode = "FS"
variables = None
groups = None
prefix_group = False
chunks = {}  # dask array

# chunks = None # very slow !

# Reading with dask array speed up concatenation !
# Otherwise with numpy array, they have to be stacked into memory, and it's really slow

filepaths = glob.glob(os.path.join(path_dir, "*"))

l_datasets = _open_valid_granules(
    filepaths=filepaths,
    scan_mode=scan_mode,
    variables=variables,
    groups=groups,
    decode_cf=False,
    prefix_group=prefix_group,
    chunks=chunks,
)

concat_dim = "along_track"

# Concatenate the datasets
ds = xr.concat(
    l_datasets,
    dim=concat_dim,
    coords="minimal",  # "all"
    compat="override",
    combine_attrs="override",
)

ds.nbytes / (1024**3)

ds1 = ds.compute()
