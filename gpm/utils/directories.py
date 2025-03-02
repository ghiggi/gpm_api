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
"""This module contains functions to search files and directories into the local machine."""
import concurrent
import fnmatch
import glob
import os
import pathlib
import re

from gpm.utils.list import flatten_list


def _recursive_glob(dir_path, glob_pattern):
    dir_path = pathlib.Path(dir_path)
    return [str(path) for path in dir_path.rglob(glob_pattern)]


def list_paths(dir_path, glob_pattern, recursive=False):
    """Return a list of filepaths and directory paths."""
    if not recursive:
        return glob.glob(os.path.join(dir_path, glob_pattern))
    return _recursive_glob(dir_path, glob_pattern)


def list_files(dir_path, glob_pattern, recursive=False):
    """Return a list of filepaths (exclude directory paths)."""
    paths = list_paths(dir_path, glob_pattern, recursive=recursive)
    return [f for f in paths if os.path.isfile(f)]


def list_directories(dir_path, glob_pattern, recursive=False):
    """Return a list of filepaths (exclude directory paths)."""
    paths = list_paths(dir_path, glob_pattern, recursive=recursive)
    return [f for f in paths if os.path.isdir(f)]


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


def list_and_filter_files(path, file_extension=None, glob_pattern=None, regex_pattern=None, sort=True):
    """Retrieve list of files (filtered by extension and custom patterns)."""
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
    if sort:
        filepaths = sorted(filepaths)
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
            sort=False,  # done at the end
        )
    else:
        filepaths = [
            list_and_filter_files(
                path,
                file_extension=file_extension,
                glob_pattern=glob_pattern,
                regex_pattern=regex_pattern,
                sort=False,
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
            sort=True,
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


def get_first_file(directory):
    """Retrieve filepath of first file inside a directory."""
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                return entry.path
    return None
