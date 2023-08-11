#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 19:35:24 2023

@author: ghiggi
"""
import numpy as np
import datetime
import gpm_api
import zarr
import xarray as xr
import time

start_time = datetime.datetime.strptime("2020-07-05 02:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-07-05 06:00:00", "%Y-%m-%d %H:%M:%S")
product = "2A-DPR"
product_type = "RS"
version = 7


####--------------------------------------------------------------------------.
#### Load GPM DPR 2A product dataset (with group prefix)

variables = [
    "precipRateAve24",
    "precipRateESurface",
    "precipRateESurface2",
    "precipRateNearSurface",
    "precipWaterIntegrated",  # precipWaterIntegrated_Liquid, precipWaterIntegrated_Solid during decoding
    "zFactorFinalESurface",
    "zFactorFinalNearSurface",
    "phaseNearSurface",
    "typePrecip",
    "flagBB",
    "flagShallowRain",
    "flagHeavyIcePrecip",
    "flagHail",
    "flagGraupelHail",
    "flagAnvil",
    "heightStormTop",
    "heightBB",
    "widthBB",
    "binClutterFreeBottom",
    "binRealSurface",
    "flagPrecip",  # 0 No precipitation, >1 precipitation
    "sunLocalTime",
    "localZenithAngle",
    "landSurfaceType",
    "elevation",
    "flagSurfaceSnowfall",
    "zFactorMeasured",  # just to get a 3D variable and do not discard "height"
    "zFactorFinal",
]
fpath = "/home/ghiggi/data/GPM/RS/V07/RADAR/2A-DPR/2022/07/08/2A.GPM.DPR.V9-20211125.20220708-S164934-E182207.047495.V07A.HDF5"
variables = None

ds = gpm_api.open_granule(fpath, variables=variables)

ds.nbytes / (1024**3)  # 5GB in memory, on disk 790 MB


# ds = gpm_api.open_dataset(
#     product=product,
#     product_type=product_type,
#     version=version,
#     variables=variables,
#     start_time=start_time,
#     end_time=end_time,
#     prefix_group=False,
# )

ds

ds["zFactorMeasured"].encoding


# ds.gpm_api.spatial_3d_variables
# ds.gpm_api.spatial_2d_variables

variable = "zFactorFinal"
variable = "zFactorMeasured"
variable = "height"


ds[variable].encoding

# Put in memory

t_i = time.time()
ds[variable] = ds[variable].compute()
t_f = time.time()
print(t_f - t_i)

### See unique values
np.set_printoptions(suppress=True)


values, counts = np.unique(ds[variable].data, return_counts=True)
values
values[0:100]
values
-100, 21966.098

# Define DataArray
# da = ds[variable]

# if variable in ds.coords:
#     da = da.reset_coords(ds.coords, drop=True)

# # Drop all coordinates

# ds = ds.drop_vars(ds.coords)

# if variable in ds.coords:
#     ds[variable] = da

# # Select a single variable
# ds = ds[[variable]]

# ds[variable].encoding


# da = ds[variable]
# da.nbytes/(1024**3) # 1.32 GB
# da.copy().astype('uint16').nbytes/(1024**3) # 0.66 GB


# counts[0:5]
# counts[:-1].sum() # 75'803'731 , 2'598'052

###---------------------------------------------------------------------------.
#### Write to Zarr

from xencoding.zarr.numcodecs import (
    get_valid_compressors,
    get_valid_blosc_algorithms,
    get_compressor,
)
from xencoding.checks.chunks import check_chunks
from xencoding.checks.zarr_compressor import check_compressor
from xencoding.zarr.writer import set_chunks, set_compressor


# Define chunks
chunks_dict = {"cross_track": -1, "along_track": 15, "range": -1, "radar_frequency": 1}

# Set xr.Dataset Chunks
chunks_dict = check_chunks(ds, chunks=chunks_dict)
ds = set_chunks(ds, chunks_dict=chunks_dict)

# Define Zarr compressor
compressor_name = "blosc"
kwargs = {"algorithm": "zstd", "clevel": 6}

# compressor_name = "b2"
# kwargs = {"clevel": 6}

compressor = get_compressor(compressor_name=compressor_name, **kwargs)

# Set Zarr compressor
compressor_dict = check_compressor(ds, compressor)
ds = set_compressor(ds, compressor_dict)

# Write Zip Zarr Store
store_path = "/tmp/example_new1_c6.zarr.zip"
zarr_store = zarr.ZipStore(store_path, mode="w")
ds.to_zarr(store=zarr_store)
zarr_store.close()

# Source netCDF: 790 MB
# 428 MB with c1 , 337 MB

# Look at ZarrZip sub directories for file size
# ZMeasured is the largest contribution !
# --> import numcodecs: ds["zFactorMeasured"].encoding['filters'] = numcodecs.Delta(?)
# --> zfpy ? `

# airTemperature
# sigmaZeroProfile
# AttenuationNP
# Followed by
# piaNPrainFree

###---------------------------------------------------------------------------.
#### Write to netCDF

# Float32, with compression 6 -->  # 442 MB
# ds.to_netcdf("/tmp/dummy_to_netcdf.nc")


# # Uint16
# ds[variable].encoding["dtype"] = "uint16"
# ds[variable].encoding["scale_factor"] = 0.01
# ds[variable].encoding["add_offset"] = -20
# ds[variable].encoding['chunksizes'] = (49, 15, 176, 2)
# ds[variable].encoding['_FillValue'] = 65535

# # Uint16, no compression --> 709 MB MB
# ds[variable].encoding['zlib'] = False
# ds[variable].encoding.pop('compression', None)
# ds[variable].encoding['complevel'] = 0
# ds.to_netcdf("/tmp/uint_no_comp_to_netcdf.nc")

# # Uint16, compression, no shuffle  --> 382.5 MB
# ds[variable].encoding['compression'] = "zlib"
# ds[variable].encoding['zlib'] = True
# ds[variable].encoding['complevel'] = 1
# ds[variable].encoding

# ds.to_netcdf("/tmp/uint1_comp1_to_netcdf.nc")

# # ds1 = xr.open_dataset("/tmp/uint_comp1_to_netcdf.nc")
# # ds1 = ds.compute()
# # values1, counts1 = np.unique(ds1[variable].data, return_counts=True)

# # Uint16, compression, shuffle=True --> 1 GB, 378 MB / 356 MB
# ds[variable].encoding["shuffle"] = True
# ds.to_netcdf("/tmp/uint_comp1_shuffle_to_netcdf.nc")

# ds[variable].encoding['complevel'] = 6
# ds.to_netcdf("/tmp/uint_comp6_shuffle_to_netcdf.nc")

# ------------------------------------------------------------------------------.
# # Changing chunk size does not change file volume !
# ds[variable].encoding['chunksizes'] = (49, 15*5, 176, 2)
# ds[[variable]].to_netcdf("/tmp/uint_comp6_shuffle_large_5_to_netcdf.nc")

# ds[variable].encoding['chunksizes'] = (49, 15*10, 176, 2)
# ds[[variable]].to_netcdf("/tmp/uint_comp6_shuffle_large_10_to_netcdf.nc")


# # Changing _FillValue to 0 slightly decrease file volume (378 --> 371 MB)
# ds[variable].encoding["dtype"] = "uint16"
# ds[variable].encoding["scale_factor"] = 0.01
# ds[variable].encoding["add_offset"] = -21
# ds[variable].encoding['chunksizes'] = (49, 15, 176, 2)
# ds[variable].encoding['_FillValue'] = 0
# ds[variable].encoding['complevel'] = 1
# ds.to_netcdf("/tmp/uint_comp1_fillvalue0_shuffle_to_netcdf.nc")

# ------------------------------------------------------------------------------.
# Uint16, compression, Zarr
ds[variable].encoding["dtype"] = "uint16"
ds[variable].encoding["scale_factor"] = 0.01
ds[variable].encoding["add_offset"] = -20
ds[variable].encoding["chunksizes"] = (49, 15, 176, 2)
ds[variable].encoding["_FillValue"] = 65535
ds[variable].encoding.pop("compression", None)
ds[variable].encoding.pop("zlib", None)
ds[variable].encoding.pop("complevel", None)
ds[variable].encoding


# lon/lat : blosc, zstd c1

# Set xr.Dataset Chunks
chunks_dict = {"cross_track": -1, "along_track": 15, "range": -1, "radar_frequency": -1}
chunks_dict = check_chunks(ds, chunks=chunks_dict)
ds = set_chunks(ds, chunks_dict=chunks_dict)


compressors_names = get_valid_compressors()
compressors_names = ["blosc", "gzip", "zstd", "zlib"]  # super slow ("b2")
clevels = [1, 6]

# compressors_names = ["b2"]
writing_time_dict = {}
reading_time_dict = {}

for compressor_name in compressors_names:
    for clevel in clevels:
        if compressor_name == "blosc":
            algorithms = get_valid_blosc_algorithms()
        else:
            algorithms = [""]

        for algorithm in algorithms:
            if compressor_name == "blosc":
                kwargs = {"clevel": clevel, "algorithm": algorithm}
            else:
                kwargs = {"clevel": clevel}
            compressor_acronym = f"{compressor_name}{algorithm}_c{clevel}"
            store_path = f"/tmp/example2_{compressor_acronym}.zarr.zip"
            print(compressor_acronym)

            t_i = time.time()

            zarr_store = zarr.ZipStore(store_path, mode="w")
            compressor = get_compressor(compressor_name=compressor_name, **kwargs)

            compressor_dict = check_compressor(ds, compressor)
            ds = set_compressor(ds, compressor_dict)

            ds.to_zarr(store=zarr_store)
            zarr_store.close()

            t_f = time.time()
            writing_time_dict[compressor_acronym] = round(t_f - t_i, 1)

            t_i = time.time()
            ds_read = xr.open_zarr(
                store_path, chunks="auto", decode_cf=True, mask_and_scale=True
            ).compute()
            # ds_read[variable]
            # ds_read[variable].encoding
            t_f = time.time()
            reading_time_dict[compressor_acronym] = round(t_f - t_i, 1)


def sort_dictionary_by_values(my_dict):
    return {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1])}


import pprint

pprint.pprint(sort_dictionary_by_values(writing_time_dict), sort_dicts=False)
pprint.pprint(sort_dictionary_by_values(reading_time_dict), sort_dicts=False)


import time

t_i = time.time()
fpath = "/tmp/uint_comp1_fillvalue0_shuffle_to_zarr.zarr"
ds = xr.open_zarr(fpath).compute()
t_f = time.time()
print(t_f - t_i)


import time

t_i = time.time()
fpath = "/tmp/uint_comp1_fillvalue0_shuffle_to_zarr.zarr.zip"
ds = xr.open_zarr(fpath).compute()
t_f = time.time()
print(t_f - t_i)


# Evaluate Read-time:
# Evaluate Compression of variables in encoding_dict
# Remove gpm_api_id, and other strings !


####--------------------------------------------------------------------------.
# TODO:
# compare compression_factor vs. writing_time and reading_time !
# independent on clevel because values can vary !

####--------------------------------------------------------------------------.
#### x-encoding

# If encoding is specified in ds.to_zarr, the ds.encoding are overwritten !

# xarray
# - If mask_and_scale=False: scale_factor, _FillValue, add_offset are in da.attrs
# - If mask_and_scale=True: scale_factor, _FillValue, add_offset are in da.encoding

# Compression Speed
# - (blosc, zstd) < gzib < zlib < b2

# Decompression speed
# - (blosc, zstd) < zlib < gzib < < b2


# Recommendations
# - blosc is fast at compression
# - bz2 is slower at compressing and decompressing, but achieve high-compression !
# - Increasing chunk size usually decrease compression ratio

# uint with similar values
# - blosc-lz4hc, blosc-lz and blosc-blosclz does not compress well
# - blosc-zstd compress well
# - b2 compress the most !
# - b2 < blosc-zstd < zstd <  blosczlib < (zlib, gzip) < blosc-lz4 < blosc-lz4hc < blosc-lz^

# Old codes to update
# - Zarr utils in xverif
#   --> https://github.com/ltelab/xverif/blob/main/xverif/utils/zarr.py#L195
# - Zarr encoding benchmarking in xforecasting/zarr
#   --> https://github.com/ghiggi/xforecasting/blob/main/xforecasting/utils/zarr.py
#
# --> I already put already everything in xencoding

# Encoding YAML:
# - Accept chunksize as dimension dictionary
# - Parse to appropriate format

# Check Encoding Schema
# --> DISDRODB Tests:
# --> https://github.com/ltelab/disdrodb/blob/main/disdrodb/l0/l0b_processing.py#L613
# --> https://github.com/ltelab/disdrodb/blob/a72dadd081ecb90a01510e4f77c21d392f6995f8/disdrodb/l0/check_configs.py#L120

# Benchmarking scripts
# --> https://github.com/deepsphere/deepsphere-weather/blob/main/scripts/03b_optimize_zarr_chunks.py

####--------------------------------------------------------------------------.
