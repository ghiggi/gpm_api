#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 20:45:17 2023

@author: ghiggi
"""
import os
import time
import dask
import gpm_api
from gpm_api.io.local import get_local_filepaths
from gpm_api.dataset.dataset import _open_valid_granules, _concat_datasets, _multi_file_closer
from dask.distributed import Client, LocalCluster

# client = Client(processes=True)

# Create dask.distributed local cluster
num_workers = os.cpu_count() - 2
cluster = LocalCluster(
    n_workers=num_workers,  # CORES
    threads_per_worker=2,
    processes=True,
    memory_limit="28GB",
)

Client(cluster)

# micromamba install mpich hdf5=*=mpi* enable to then install libnetcdf with MPI
# micromamba install libnetcdf=*=mpi*
# micromamba install netcdf4=*=mpi*

# autoclose=True is set automatically to True in xarray backends
# if chunks is not None and scheduler in ["distributed", "multiprocessing"]

# xarray.set_options(file_cache_maxsize=1024)
# - LRU (least-recently-used cache) is used to store open files; the default limit of 128

# we can sidestep the global HDF lock if we use multiprocessing
# (or the distributed scheduler) and the autoclose option
# --> https://github.com/pydata/xarray/pull/1983

# Implement datatree closer
# TODO: implement datatree.close() and datatree._close in datatree repository
# --> datatree._close as iterator ?
# --> https://github.com/xarray-contrib/datatree/issues/93
# --> https://github.com/xarray-contrib/datatree/pull/114/files


# Define GPM product
product = "2A-DPR"
version = 7
scan_mode = "FS"
product_type = "RS"

groups = None
variables = None
prefix_group = False
parallel = True

chunks = {}
decode_cf = False


fpaths = get_local_filepaths(product=product, product_type=product_type, version=version)

print(len(fpaths))
filepaths = fpaths
filepaths = fpaths[0:10]


t_i = time.time()

# with dask.config.set(scheduler="threading"):
# with dask.config.set(scheduler="threads"):
# with dask.config.set(scheduler="distributed"):
# with dask.config.set(pool=ThreadPoolExecutor(4)):
# with dask.config.set({"multiprocessing.context": "spawn"}):
# with dask.config.set({"multiprocessing.context": "forkserver"}):
# with dask.config.set(scheduler="multiprocessing"):
list_ds, list_closers = _open_valid_granules(
    filepaths,
    scan_mode=scan_mode,
    variables=variables,
    groups=groups,
    decode_cf=decode_cf,
    prefix_group=prefix_group,
    parallel=parallel,
    chunks=chunks,
)
t_f = time.time()
t_elapsed = round(t_f - t_i, 2)
print(t_elapsed)

print(list_closers)  # need to implement datatree closers

# parallel=False, no HDF warnings ...

# 5 files
# - parallel=False:
# - parallel=True, default dask :   s
# - parallel=True, dask distributed processes=True:  s
# - parallel=True, dask distributed processes=False:   s
# - parallel=True, dask threads:   s
# - parallel=True, dask threading:   s
# - parallel=True, dask distributed:   s
# - parallel=True, dask multiprocessing spawn:   s
# - parallel=True, dask multiprocessing forkserver:   s


##-------------------------------------------------------------------------.

try:
    ds = _concat_datasets(list_ds)
except ValueError:
    for ds in list_ds:
        ds.close()
    raise

##-------------------------------------------------------------------------.
# Set dataset closers to execute when ds is closed
ds.set_close(partial(_multi_file_closer, list_closers))

ds.close()

##-------------------------------------------------------------------------.
# To open a datatree
# filepath = "/home/ghiggi/data/GPM/RS/V07/RADAR/2A-DPR/2022/07/11/2A.GPM.DPR.V9-20211125.20220711-S093700-E110932.047537.V07A.HDF5"
