#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:12:06 2023

@author: ghiggi
"""
import os
import h5py
import gpm_api
from gpm_api.dataset.datatree import _open_datatree


def copy_attributes(original, copy):
    for key, value in original.attrs.items():
        copy.attrs[key] = value


def copy_datasets_recursive(original_group, subset_group, desired_size=10):
    for name, item in original_group.items():
        if isinstance(item, h5py.Dataset):

            # Determine the subset shape (2 indices per dimension)
            subset_shape = tuple(min(3, dim_size) for dim_size in item.shape)

            # Create a new dataset in the subset group with the subset shape
            subset_dataset = subset_group.create_dataset(name, subset_shape, dtype=item.dtype)

            # Copy data from the original dataset to the subset dataset
            subset_dataset[:] = item[tuple(slice(0, size) for size in subset_shape)]

            # Copy attributes from the original dataset to the subset dataset
            copy_attributes(item, subset_dataset)

            # Copy encoding information
            if item.compression is not None and "compression" in item.compression:
                subset_dataset.compression = item.compression
                subset_dataset.compression_opts = item.compression_opts

        elif isinstance(item, h5py.Group):

            # If the item is a group, create a corresponding group in the subset file and copy its datasets recursively
            subgroup = subset_group.create_group(name)
            # Copy group attributes
            copy_attributes(item, subgroup)
            copy_datasets_recursive(item, subgroup)


# ------------------------------------------------------------------------------.
## Create small HDF5 for testing !
dst_dir = "/tmp"
src_filepath = "/home/ghiggi/data/GPM/RS/V07/RADAR/2A-DPR/2022/07/06/2A.GPM.DPR.V9-20211125.20220706-S043937-E061210.047456.V07A.HDF5"

# Open source HDF5 file
src_file = h5py.File(src_filepath, "r")

# Create empty HDF5 file
dst_file_path = os.path.join(dst_dir, os.path.basename(src_filepath))
dst_file = h5py.File(dst_file_path, "w")

# Write a subset of the source HDF5 groups and leafs into the new HDF5 file
copy_datasets_recursive(src_file, dst_file, desired_size=10)

# Write attributes from the source HDF5 root group to the new HDF5 file root group
copy_attributes(src_file, dst_file)

# Close connection
src_file.close()
dst_file.close()

# --> File size is 178 KB (src file was 773 MB)

# ------------------------------------------------------------------------------.
## Test it open correctly

dt = _open_datatree(dst_file_path)
dt.attrs
dt["FS"].attrs
dt["FS"]["SLV"].attrs
dt["FS"]["scanStatus"]["SCorientation"]

ds = gpm_api.open_granule(dst_file_path)
