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
import concurrent
import fnmatch
import importlib
import os
import re

from gpm.utils.list import flatten_list
from gpm.utils.yaml import read_yaml, write_yaml


def read_bucket_info(bucket_dir):
    os.makedirs(bucket_dir, exist_ok=True)
    bucket_info_filepath = os.path.join(bucket_dir, "bucket_info.yaml")
    bucket_info = read_yaml(filepath=bucket_info_filepath)
    return bucket_info


def get_bucket_partitioning(bucket_dir):
    bucket_info = read_bucket_info(bucket_dir)
    class_name = bucket_info.pop("partitioning_class")
    partitioning_class = getattr(importlib.import_module("gpm.bucket.partitioning"), class_name)
    partitioning = partitioning_class(**bucket_info)
    return partitioning


def write_bucket_info(bucket_dir, partitioning):
    os.makedirs(bucket_dir, exist_ok=True)
    bucket_info = partitioning.to_dict()
    bucket_info_filepath = os.path.join(bucket_dir, "bucket_info.yaml")
    write_yaml(bucket_info, filepath=bucket_info_filepath, sort_keys=False)


####------------------------------------------------------------------------------------------------------------------.
###########################
#### Search and filter ####
###########################


def match_extension(filename, extension=None):
    if extension is None:
        return True
    return filename.endswith(extension)


def match_regex_pattern(filename, pattern=None):
    # assume regex pattern is re.compiled !
    if pattern is None:
        return True
    return re.match(pattern, filename) is not None


def match_glob_pattern(filename, pattern=None):
    # assume Unix shell-style wildcards
    if pattern is None:
        return True
    return fnmatch.fnmatch(filename, pattern)


def match_filters(filename, file_extension=None, glob_pattern=None, regex_pattern=None):
    return (
        match_extension(filename=filename, extension=file_extension)
        and match_regex_pattern(filename=filename, pattern=regex_pattern)
        and match_glob_pattern(filename=filename, pattern=glob_pattern)
    )


def list_and_filter_files(path, file_extension=None, glob_pattern=None, regex_pattern=None):
    """Retrieve list of files (filtered by extension and custom patterns."""
    with os.scandir(path) as file_it:
        filepaths = [
            file_entry.path
            for file_entry in file_it
            if (
                file_entry.is_file()
                and match_filters(
                    filename=file_entry.name,
                    file_extension=file_extension,
                    glob_pattern=glob_pattern,
                    regex_pattern=regex_pattern,
                )
            )
        ]
    return filepaths


def get_parallel_list_results(function, inputs, **kwargs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_dict = {executor.submit(function, i, **kwargs): i for i in inputs}
        results = [future.result() for future in concurrent.futures.as_completed(future_dict)]
    return results


def get_parallel_dict_results(function, inputs, **kwargs):
    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_dict = {executor.submit(function, i, **kwargs): i for i in inputs}
        for future in concurrent.futures.as_completed(future_dict):
            i = future_dict[future]
            try:
                result = future.result()
                results[i] = result
            except Exception as e:
                print(f"Error while processing {i}: {e}")
    return results


def get_filepaths_within_paths(paths, parallel=True, file_extension=None, glob_pattern=None, regex_pattern=None):
    """Return a list with all filepaths within a list of directories matching the filename filtering criteria."""
    if regex_pattern is not None:
        regex_pattern = re.compile(regex_pattern)
    if parallel:
        filepaths = get_parallel_list_results(
            function=list_and_filter_files,
            inputs=paths,
            file_extension=file_extension,
            glob_pattern=glob_pattern,
            regex_pattern=regex_pattern,
        )
    else:
        filepaths = [
            list_and_filter_files(
                path,
                file_extension=file_extension,
                glob_pattern=glob_pattern,
                regex_pattern=regex_pattern,
            )
            for path in paths
        ]
    # Unflatten filepaths
    return sorted(flatten_list(filepaths))


def get_filepaths_by_path(paths, parallel=True, file_extension=None, glob_pattern=None, regex_pattern=None):
    """Return a dictionary with the files within each directory path matching the filename filtering criteria."""
    if regex_pattern is not None:
        regex_pattern = re.compile(regex_pattern)
    if parallel:
        dict_partitions = get_parallel_dict_results(
            function=list_and_filter_files,
            inputs=paths,
            file_extension=file_extension,
            glob_pattern=glob_pattern,
            regex_pattern=regex_pattern,
        )
    else:
        dict_partitions = {
            path: list_and_filter_files(
                path,
                file_extension=file_extension,
                glob_pattern=glob_pattern,
                regex_pattern=regex_pattern,
            )
            for path in paths
        }
    return dict_partitions


def get_subdirectories(base_dir, path=True):
    """Return the name or path of the directories present in the input directory."""
    with os.scandir(base_dir) as base_it:
        if path:
            list_sub_dirs = [sub_entry.path for sub_entry in base_it if sub_entry.is_dir()]
        else:
            list_sub_dirs = [sub_entry.name for sub_entry in base_it if sub_entry.is_dir()]
    return list_sub_dirs


def _search_leaf_directories(base_dir):
    """Search leaf directories paths."""
    leaf_directories = []

    # Search for leaf directories
    def scan_directory(current_dir):
        is_leaf = True
        with os.scandir(current_dir) as it:
            for entry in it:
                if entry.is_dir():
                    is_leaf = False
                    scan_directory(os.path.join(current_dir, entry.name))
        if is_leaf:
            leaf_directories.append(current_dir)

    scan_directory(base_dir)
    return leaf_directories


def search_leaf_directories(base_dir, parallel=True, remove_base_path=True):
    """Search leaf directories."""
    if not parallel:
        leaf_directories = _search_leaf_directories(base_dir)
    else:
        # Find directories in the base_dir
        list_dirs = get_subdirectories(base_dir, path=True)
        # Search in parallel across subdirectories
        list_leaf_directories = get_parallel_list_results(function=_search_leaf_directories, inputs=list_dirs)
        leaf_directories = flatten_list(list_leaf_directories)
    # Remove base_dir path
    if remove_base_path:
        leaf_directories = [path.removeprefix(str(base_dir)).strip(os.path.sep) for path in leaf_directories]
    return leaf_directories


def search_leaf_files(base_dir, parallel=True, file_extension=None, glob_pattern=None, regex_pattern=None):
    """Search files in leaf directories."""
    paths = search_leaf_directories(base_dir, parallel=parallel, remove_base_path=False)
    filepaths = get_filepaths_within_paths(
        paths,
        parallel=parallel,
        file_extension=file_extension,
        glob_pattern=glob_pattern,
        regex_pattern=regex_pattern,
    )
    return filepaths


####------------------------------------------------------------------------------------------------------------------.
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
    partitioning = get_bucket_partitioning(bucket_dir=bucket_dir)
    dir_trees = partitioning.directories
    return get_exisiting_partitions_paths(bucket_dir, dir_trees)


def get_filepaths(bucket_dir, parallel=True, file_extension=None, glob_pattern=None, regex_pattern=None):
    """Return the filepaths matching the specified filename filtering criteria."""
    partitioning = get_bucket_partitioning(bucket_dir=bucket_dir)
    dir_trees = partitioning.directories
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
    partitioning = get_bucket_partitioning(bucket_dir=bucket_dir)
    n_levels = partitioning.n_levels
    dir_trees = partitioning.directories
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
