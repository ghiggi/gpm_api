# -----------------------------------------------------------------------------.
# MIT License

# Copyright (c) 2024 GPM-API developers
#
# This file is part of GPM-API.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -----------------------------------------------------------------------------.
"""This module provide utilities to search GPM Geographic Buckets files."""
import importlib
import os

from gpm.utils.directories import get_filepaths_by_path, get_filepaths_within_paths
from gpm.utils.yaml import read_yaml, write_yaml


def read_bucket_info(bucket_dir):
    """Read the bucket metadata."""
    bucket_info_filepath = os.path.join(bucket_dir, "bucket_info.yaml")
    bucket_info = read_yaml(filepath=bucket_info_filepath)
    return bucket_info


def get_bucket_spatial_partitioning(bucket_dir):
    """Return the bucket spatial partitioning."""
    bucket_info = read_bucket_info(bucket_dir)
    partitioning_dict = bucket_info["spatial_partitioning"]
    class_name = partitioning_dict.pop("class")
    partitioning_class = getattr(importlib.import_module("gpm.bucket.partitioning"), class_name)
    spatial_partitioning = partitioning_class(**partitioning_dict)
    return spatial_partitioning


def get_bucket_temporal_partitioning(bucket_dir):
    """Return the bucket spatial partitioning."""
    bucket_info = read_bucket_info(bucket_dir)
    return bucket_info.get("temporal_partitioning", None)


def write_bucket_info(bucket_dir, spatial_partitioning, temporal_partitioning=None, start_time=None, end_time=None):
    """Write the bucket metadata."""
    # Define bucket metadata dictionary
    bucket_info = {}
    # - Add spatial and temporal partitioning information
    bucket_info["spatial_partitioning"] = spatial_partitioning.to_dict()
    bucket_info["temporal_partitioning"] = temporal_partitioning
    # - Update bucket temporal coverage information
    if start_time is not None:
        bucket_info["start_time"] = str(start_time)
    if end_time is not None:
        bucket_info["end_time"] = str(end_time)
    # Write bucket metadata
    os.makedirs(bucket_dir, exist_ok=True)
    bucket_info_filepath = os.path.join(bucket_dir, "bucket_info.yaml")
    write_yaml(bucket_info, filepath=bucket_info_filepath, sort_keys=False)


####---------------------------------------------------------------------------.
#### Bucket partitions utilities


def get_exisiting_partitions_paths(bucket_dir, dir_trees):
    """Get the path of existing bucket partitions on disk."""
    # Retrieve current partitions
    paths = [os.path.join(bucket_dir, dir_tree) for dir_tree in dir_trees]
    #  Select existing directories
    paths = [path for path in paths if os.path.exists(path)]
    return paths


def get_partitions_paths(bucket_dir):
    """Get the path of the bucket partitions."""
    spatial_partitioning = get_bucket_spatial_partitioning(bucket_dir=bucket_dir)
    dir_trees = spatial_partitioning.directories
    return get_exisiting_partitions_paths(bucket_dir, dir_trees)


def get_filepaths(bucket_dir, parallel=True, file_extension=None, glob_pattern=None, regex_pattern=None):
    """Return the filepaths matching the specified filename filtering criteria."""
    spatial_partitioning = get_bucket_spatial_partitioning(bucket_dir=bucket_dir)
    dir_trees = spatial_partitioning.directories
    partitions_paths = get_exisiting_partitions_paths(bucket_dir, dir_trees)
    filepaths = get_filepaths_within_paths(
        paths=partitions_paths,
        parallel=parallel,
        file_extension=file_extension,
        glob_pattern=glob_pattern,
        regex_pattern=regex_pattern,
    )
    return filepaths


def get_filepaths_by_partition(bucket_dir, parallel=True, file_extension=None, glob_pattern=None, regex_pattern=None):
    """Return a dictionary with the list of filepaths for each bucket partition."""
    spatial_partitioning = get_bucket_spatial_partitioning(bucket_dir=bucket_dir)
    n_levels = spatial_partitioning.n_levels
    dir_trees = spatial_partitioning.directories
    partitions_paths = get_exisiting_partitions_paths(bucket_dir, dir_trees)
    dict_filepaths = get_filepaths_by_path(
        paths=partitions_paths,
        parallel=parallel,
        file_extension=file_extension,
        glob_pattern=glob_pattern,
        regex_pattern=regex_pattern,
    )
    sep = os.path.sep
    dict_partition_files = {sep.join(k.strip(sep).split(sep)[-n_levels:]): v for k, v in dict_filepaths.items()}
    return dict_partition_files


####---------------------------------------------------------------------------.
