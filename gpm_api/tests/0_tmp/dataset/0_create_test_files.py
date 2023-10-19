#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:12:06 2023

@author: ghiggi
"""
import os
import h5py
import datatree
import gpm_api
from gpm_api.dataset.datatree import open_datatree


def _get_fixed_dimensions():
    """Dimensions over which to not subset the GPM HDF5 files."""
    fixed_dims = [
        # Elevations / Range
        "nBnPSD",
        "nBnPSDhi",
        "nBnEnv",
        "nbinMS",
        "nbinHS",
        "nbinFS",
        "nbin",
        # Radar frequency
        "nKuKa",
        "nfreq",
        # PMW frequency
        "nemiss",
        "nchan1",
        "nchan2",
        "nchannel1",
        "nchannel2",
        "nchannel3",
        "nchannel4",
        "nchannel5",
        "nchannel6",
    ]
    return fixed_dims


def _get_subset_shape_chunks(h5_obj, subset_size=5):
    """Return the shape and chunks of the subsetted HDF5 file."""
    dimnames = h5_obj.attrs.get("DimensionNames", None)
    fixed_dims = _get_fixed_dimensions()
    chunks = h5_obj.chunks
    if dimnames is not None:
        # Get dimension names list
        dimnames = dimnames.decode().split(",")
        # Get dimension shape
        shape = h5_obj.shape
        # Create dimension dictionary
        dict_dims = dict(zip(dimnames, shape))
        # Create chunks dictionary
        dict_chunks = dict(zip(dimnames, chunks))
        # Define subset shape and chunks
        subset_shape = []
        subset_chunks = []
        for dim, src_size in dict_dims.items():
            chunk = dict_chunks[dim]
            if dim in fixed_dims:
                subset_shape.append(src_size)
                subset_chunks.append(chunk)
            else:
                subset_size = min(subset_size, src_size)
                subset_chunk = min(chunk, subset_size)
                subset_shape.append(subset_size)
                subset_chunks.append(subset_chunk)

        # Determine subset shape
        subset_shape = tuple(subset_shape)
        subset_chunks = tuple(subset_chunks)
    else:
        subset_shape = h5_obj.shape
        subset_chunks = h5_obj.chunks
    return subset_shape, subset_chunks


def _copy_attrs(src_h5_obj, dst_h5_obj):
    """Copy attributes from the source file to the destination file."""
    for key, value in src_h5_obj.attrs.items():
        dst_h5_obj.attrs[key] = value


def _copy_datasets(src_group, dst_group, subset_size=5):
    for name, h5_obj in src_group.items():
        if isinstance(h5_obj, h5py.Dataset):
            # Determine the subset shape (2 indices per dimension)
            subset_shape, subset_chunks = _get_subset_shape_chunks(h5_obj, subset_size=subset_size)

            # Create a new dataset in the subset group with the subset shape
            subset_dataset = dst_group.create_dataset(
                name, subset_shape, dtype=h5_obj.dtype, chunks=subset_chunks
            )

            # Copy data from the src_h5_obj dataset to the subset dataset
            subset_dataset[:] = h5_obj[tuple(slice(0, size) for size in subset_shape)]

            # Copy attributes from the src_h5_obj dataset to the subset dataset
            _copy_attrs(h5_obj, subset_dataset)

            # Copy encoding information
            if h5_obj.compression is not None and "compression" in h5_obj.compression:
                subset_dataset.compression = h5_obj.compression
                subset_dataset.compression_opts = h5_obj.compression_opts

        elif isinstance(h5_obj, h5py.Group):
            # If the h5_obj is a group, create a corresponding group in the subset file and copy its datasets recursively
            subgroup = dst_group.create_group(name)
            # Copy group attributes
            _copy_attrs(h5_obj, subgroup)
            _copy_datasets(h5_obj, subgroup, subset_size=subset_size)


def create_test_hdf5(src_fpath, dst_fpath):
    # Open source HDF5 file
    src_file = h5py.File(src_fpath, "r")

    # Create empty HDF5 file
    dst_file = h5py.File(dst_fpath, "w")

    # Write a subset of the source HDF5 groups and leafs into the new HDF5 file
    _copy_datasets(src_file, dst_file, subset_size=10)

    # Write attributes from the source HDF5 root group to the new HDF5 file root group
    _copy_attrs(src_file, dst_file)

    # Close connection
    src_file.close()
    dst_file.close()


# ------------------------------------------------------------------------------.
## Create small HDF5 for testing !
dst_dir = "/tmp"
src_fpath = "/home/ghiggi/data/GPM/RS/V07/RADAR/2A-DPR/2022/07/06/2A.GPM.DPR.V9-20211125.20220706-S043937-E061210.047456.V07A.HDF5"
src_fpath = "/home/ghiggi/data/GPM/RS/V07/PMW/1A-GMI/2020/08/01/1A.GPM.GMI.COUNT2021.20200801-S105247-E122522.036508.V07A.HDF5"
src_fpath = "/home/ghiggi/data/GPM/RS/V07/PMW/1C-GMI/2020/08/01/1C.GPM.GMI.XCAL2016-C.20200801-S105247-E122522.036508.V07A.HDF5"
src_fpath = "/home/ghiggi/data/GPM/RS/V07/PMW/2A-GMI/2022/07/06/2A.GPM.GMI.GPROF2021v1.20220706-S183242-E200515.047465.V07A.HDF5"

# BUG with IMERG
# src_fpath = "/home/ghiggi/data/GPM/RS/V07/IMERG/IMERG-FR/2016/03/09/3B-HHR.MS.MRG.3IMERG.20160309-S100000-E102959.0600.V07A.HDF5"


dst_fpath = os.path.join(dst_dir, os.path.basename(src_fpath))
create_test_hdf5(
    src_fpath=src_fpath,
    dst_fpath=dst_fpath,
)

## Test it open correctly
ds = gpm_api.open_granule(dst_fpath)
ds

# ------------------------------------------------------------------------------.
# DEBUG
# Open source HDF5 file
# src_file = h5py.File(src_fpath, "r")

# # dt = datatree.open_datatree(src_fpath, engine="netcdf4")
# # name, h5_obj = list(src_file["/FS/PRE"].items())[0] # 2A-DPR
# # name, h5_obj = list(src_file["/Grid"].items())[0]   # IMERG-FR

# # Create empty HDF5 file
# dst_fpath = os.path.join(dst_dir, os.path.basename(src_fpath))
# dst_file = h5py.File(dst_fpath, "w")

# # Write a subset of the source HDF5 groups and leafs into the new HDF5 file
# _copy_datasets(src_file, dst_file, subset_size=10)

# # Write attributes from the source HDF5 root group to the new HDF5 file root group
# _copy_attrs(src_file, dst_file)

# # Close connection
# src_file.close()
# dst_file.close()

# ------------------------------------------------------------------------------.


# dt = open_datatree(src_fpath)

# dt = open_datatree(dst_fpath)
# dt.attrs
# dt["FS"].attrs
# dt["FS"]["SLV"].attrs
# dt["FS"]["scanStatus"]["SCorientation"]
