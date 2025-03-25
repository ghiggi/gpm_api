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

from gpm.bucket.io import (
    get_filepaths,
    get_filepaths_by_partition,
    get_partitions_paths,
    write_bucket_info,
)
from gpm.bucket.partitioning import LonLatPartitioning

# # TO DEBUG
# import pathlib
# tmp_path = pathlib.Path("/tmp/bucket14")


def create_test_bucket(bucket_dir):
    spatial_partitioning = LonLatPartitioning(size=(10, 10), flavor="hive")
    write_bucket_info(bucket_dir=bucket_dir, spatial_partitioning=spatial_partitioning)
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


def test_get_partitions_paths(tmp_path):
    # Create the directory structure
    bucket_dir = tmp_path
    create_test_bucket(bucket_dir=bucket_dir)
    # Test results
    results = get_partitions_paths(bucket_dir=bucket_dir)
    expected_results = [
        os.path.join(bucket_dir, "lon_bin=-5.0", "lat_bin=-5.0"),
        os.path.join(bucket_dir, "lon_bin=-5.0", "lat_bin=5.0"),
    ]
    assert sorted(results) == sorted(expected_results)


def test_get_filepaths(tmp_path):
    # Create the directory structure
    bucket_dir = tmp_path
    create_test_bucket(bucket_dir=bucket_dir)
    # Test results without filtering
    results = get_filepaths(
        bucket_dir=bucket_dir,
        parallel=True,
        file_extension=None,
        glob_pattern=None,
        regex_pattern=None,
    )
    assert len(results) == 5
    # Test results with filtering
    results = get_filepaths(
        bucket_dir=bucket_dir,
        parallel=True,
        file_extension="parquet",
        glob_pattern=None,
        regex_pattern=None,
    )
    assert len(results) == 4


def test_get_filepaths_by_partition(tmp_path):
    # Create the directory structure
    bucket_dir = tmp_path
    create_test_bucket(bucket_dir=bucket_dir)
    # Test results without filtering
    dict_results = get_filepaths_by_partition(
        bucket_dir=bucket_dir,
        parallel=True,
        file_extension=None,
        glob_pattern=None,
        regex_pattern=None,
    )
    assert isinstance(dict_results, dict)
    expected_keys = [f"lon_bin=-5.0{os.sep}lat_bin=-5.0", f"lon_bin=-5.0{os.sep}lat_bin=5.0"]
    assert sorted(dict_results) == sorted(expected_keys)
    assert len(dict_results[expected_keys[0]]) == 2
    assert len(dict_results[expected_keys[1]]) == 3
