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
"""This module tests the bucket I/O utilities."""
import os

import pytest

from gpm.bucket.io import (
    get_filepaths_by_path,
    get_filepaths_within_paths,
    get_partitions_paths,
    get_subdirectories,
    search_leaf_directories,
    search_leaf_files,
    write_bucket_info,
)
from gpm.bucket.partitioning import LonLatPartitioning

# # TO DEBUG
# import pathlib
# tmp_path = pathlib.Path("/tmp/bucket14")


def create_test_bucket(bucket_dir):
    partitioning = LonLatPartitioning(size=(10, 10), partitioning_flavor="hive")
    write_bucket_info(bucket_dir=bucket_dir, partitioning=partitioning)
    # Define test paths
    paths_components = [
        ("lon_bin=-5.0", "lat_bin=5.0", "2A.GPM.DPR.V9-20211125.20230705-S013942-E031214.041760.V07A_0.parquet"),
        # Special case to test filtering
        ("lon_bin=-5.0", "lat_bin=5.0", "2A.GPM.DPR.V9-20211125.20210705-S013942-E031214.041760.V07A_0.bad_extension"),
        ("lon_bin=-5.0", "lat_bin=5.0", "2B.GPM.DPR.V9-20211125.20210805-S013942-E031214.041760.V07B_0.parquet"),
        # Other files
        ("lon_bin=-5.0", "lat_bin=-5.0", "2A.GPM.DPR.V10-20211125.20230705-S013942-E031214.041760.V07A_0.parquet"),
        ("lon_bin=-5.0", "lat_bin=-5.0", "2A.GPM.DPR.V9-20211125.20230705-S013942-E031214.041760.V07A_0.parquet"),
    ]
    for path_compoments in paths_components:
        path = os.path.join(bucket_dir, *path_compoments)
        # Extract the directory part of the path
        dir_path = os.path.dirname(path)
        # Create the directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)
        # Create an empty file at the final path
        with open(path, "w") as f:
            f.write("")  # Writing an empty string to create the file


def test_get_filepaths_within_paths(tmp_path):
    """Test get_filepaths_within_paths."""
    # Create the directory structure
    bucket_dir = tmp_path
    create_test_bucket(bucket_dir=bucket_dir)
    paths = [
        os.path.join(bucket_dir, "lon_bin=-5.0", "lat_bin=-5.0"),
        os.path.join(bucket_dir, "lon_bin=-5.0", "lat_bin=5.0"),
    ]

    # Test results without filtering
    filepaths_p = get_filepaths_within_paths(paths, parallel=True)
    filepaths = get_filepaths_within_paths(paths, parallel=False)
    assert len(filepaths) == 5
    assert filepaths == filepaths_p

    # Test results with extension filtering
    filepaths_p = get_filepaths_within_paths(paths, parallel=True, file_extension=".parquet")
    filepaths = get_filepaths_within_paths(paths, parallel=False, file_extension=".parquet")
    assert len(filepaths) == 4
    assert filepaths == filepaths_p

    # Test results with glob filtering
    filepaths_p = get_filepaths_within_paths(paths, parallel=True, glob_pattern="*.V07B_*")
    filepaths = get_filepaths_within_paths(paths, parallel=False, glob_pattern="*.V07B_*")
    assert len(filepaths) == 1
    assert filepaths == filepaths_p

    # Test results with regexp
    filepaths_p = get_filepaths_within_paths(paths, parallel=True, regex_pattern="2B\\.GPM.*\\.parquet$")
    filepaths = get_filepaths_within_paths(paths, parallel=False, regex_pattern="2B\\.GPM.*\\.parquet$")
    assert len(filepaths) == 1
    assert filepaths == filepaths_p


def test_get_filepaths_by_path(tmp_path):
    """Test get_filepaths_by_path."""
    # Create the directory structure
    bucket_dir = tmp_path
    create_test_bucket(bucket_dir=bucket_dir)
    path1 = os.path.join(bucket_dir, "lon_bin=-5.0", "lat_bin=-5.0")
    path2 = os.path.join(bucket_dir, "lon_bin=-5.0", "lat_bin=5.0")
    paths = [path1, path2]

    # Test results without filtering
    dict_filepaths_p = get_filepaths_by_path(paths, parallel=True)
    dict_filepaths = get_filepaths_by_path(paths, parallel=False)
    assert len(dict_filepaths) == 2
    assert dict_filepaths == dict_filepaths_p
    assert len(dict_filepaths[path1]) == 2
    assert len(dict_filepaths[path2]) == 3

    # Test results with filtering
    dict_filepaths_p = get_filepaths_by_path(paths, parallel=True, file_extension=".parquet", glob_pattern="*.V07B_*")
    dict_filepaths = get_filepaths_by_path(paths, parallel=False, file_extension=".parquet", glob_pattern="*.V07B_*")
    assert len(dict_filepaths) == 2
    assert dict_filepaths == dict_filepaths_p
    assert len(dict_filepaths[path1]) == 0
    assert len(dict_filepaths[path2]) == 1
    assert dict_filepaths[path1] == []

    # Test results filtering with regexp
    dict_filepaths_p = get_filepaths_by_path(paths, parallel=True, regex_pattern=r"2B\.GPM.*\.parquet$")
    dict_filepaths = get_filepaths_by_path(paths, parallel=False, regex_pattern=r"2B\.GPM.*\.parquet$")
    assert len(dict_filepaths) == 2
    assert dict_filepaths == dict_filepaths_p
    assert len(dict_filepaths[path1]) == 0
    assert len(dict_filepaths[path2]) == 1
    assert dict_filepaths[path1] == []


@pytest.mark.parametrize("remove_base_path", [True, False])
def test_search_leaf_directories(tmp_path, remove_base_path):
    # Create the directory structure
    bucket_dir = tmp_path
    create_test_bucket(bucket_dir=bucket_dir)
    leaf_path1 = os.path.join("lon_bin=-5.0", "lat_bin=-5.0")
    leaf_path2 = os.path.join("lon_bin=-5.0", "lat_bin=5.0")
    path1 = os.path.join(bucket_dir, leaf_path1)
    path2 = os.path.join(bucket_dir, leaf_path2)

    paths = [path1, path2]
    leaf_paths = [leaf_path1, leaf_path2]

    # Test results without filtering
    leaf_directories_p = search_leaf_directories(base_dir=bucket_dir, parallel=True, remove_base_path=remove_base_path)
    leaf_directories = search_leaf_directories(base_dir=bucket_dir, parallel=False, remove_base_path=remove_base_path)
    assert len(leaf_directories) == 2
    assert leaf_directories == leaf_directories_p
    if remove_base_path:
        assert sorted(leaf_directories) == sorted(leaf_paths)
    else:
        assert sorted(leaf_directories) == sorted(paths)


def test_get_subdirectories(tmp_path):
    # Create the directory structure
    bucket_dir = tmp_path
    create_test_bucket(bucket_dir=bucket_dir)

    # Test results
    results = get_subdirectories(base_dir=bucket_dir, path=False)
    assert results == ["lon_bin=-5.0"]
    results = get_subdirectories(base_dir=os.path.join(bucket_dir, "lon_bin=-5.0"), path=False)
    assert results == ["lat_bin=5.0", "lat_bin=-5.0"]


def test_get_partitions_paths(tmp_path):
    # Create the directory structure
    bucket_dir = tmp_path
    create_test_bucket(bucket_dir=bucket_dir)
    # Test results
    results = get_partitions_paths(bucket_dir=bucket_dir)
    assert results == [
        os.path.join(bucket_dir, "lon_bin=-5.0", "lat_bin=-5.0"),
        os.path.join(bucket_dir, "lon_bin=-5.0", "lat_bin=5.0"),
    ]


def test_search_leaf_files_in_parallel(tmp_path):
    # Create the directory structure
    bucket_dir = tmp_path
    create_test_bucket(bucket_dir=bucket_dir)

    # Test results without filtering
    filepaths_p = search_leaf_files(
        base_dir=bucket_dir,
        parallel=True,
    )
    filepaths = search_leaf_files(
        base_dir=bucket_dir,
        parallel=False,
    )
    assert len(filepaths) == 5
    assert sorted(filepaths) == sorted(filepaths_p)
