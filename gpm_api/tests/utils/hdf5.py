#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:44:10 2023

@author: ghiggi
"""
import os
import h5py


def _get_fixed_dimensions():
    """Dimensions over which to not subset the GPM HDF5 files."""
    fixed_dims = [
        # Elevations / Range
        "nBnPSD",
        "nBnPSDhi",
        "nbinMS",
        "nbinHS",
        "nbinFS",
        "nbin",
        "nBnEnv",  # CORRA
        "nlayer",  # SLH, CSH
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


def _get_subset_shape_chunks(h5_obj, subset_size):
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


def _copy_datasets(src_group, dst_group, subset_size):
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
    """Create test HDF5 file."""
    if os.path.exists(dst_fpath):
        os.remove(dst_fpath)

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
