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
"""This module test the directories/files utilities."""
import os

import pytest

from gpm.utils.directories import (
    get_filepaths_by_path,
    get_filepaths_within_paths,
    get_subdirectories,
    list_files,
    search_leaf_directories,
    search_leaf_files,
)


def test_list_files(tmp_path):
    # Set up test environment
    ext = "HDF5"
    dir1 = tmp_path / "2020"
    dir1.mkdir()

    dir2 = dir1 / "07"
    dir2.mkdir()

    dir1_dummy = tmp_path / "dir1_dummy"
    dir1_dummy.mkdir()

    dir2_dummy = dir1 / "dir2_dummy"
    dir2_dummy.mkdir()

    # Create files in the base directory (tmp_path)
    file1 = tmp_path / f"file1.{ext}"
    file2 = tmp_path / f"file2.{ext}"
    file3 = tmp_path / "file3.ANOTHER"

    # Create files in the sub directory
    file4 = dir1 / f"file4.{ext}"
    file5 = dir1 / "file5.ANOTHER"

    # Create files in the sub-sub directory
    file6 = dir2 / f"file6.{ext}"
    file7 = dir2 / "file7.ANOTHER"

    # Create files
    file1.touch()
    file2.touch()
    file3.touch()
    file4.touch()
    file5.touch()
    file6.touch()
    file7.touch()

    # Search for all files in the base directory
    glob_pattern = "*"
    expected_files = [file1, file2, file3]
    assert set(list_files(tmp_path, glob_pattern, recursive=False)) == set(map(str, expected_files))

    # Search for specific pattern (extension) in the base directory
    glob_pattern = f"*.{ext}"
    expected_files = [file1, file2]
    assert set(list_files(tmp_path, glob_pattern, recursive=False)) == set(map(str, expected_files))

    # Search for all files only in the sub directory
    glob_pattern = os.path.join("*", "*")
    expected_files = [file4, file5]
    assert set(list_files(tmp_path, glob_pattern, recursive=False)) == set(map(str, expected_files))

    # Search for specific pattern (extension) in the sub directory
    glob_pattern = os.path.join("*", f"*.{ext}")
    expected_files = [file4]
    assert set(list_files(tmp_path, glob_pattern, recursive=False)) == set(map(str, expected_files))

    # Search for all files (with specific pattern) in all the subdirectories (from the base directory)
    # --> file7 is not included because of different pattern
    glob_pattern = f"*.{ext}"
    expected_files = [file1, file2, file4, file6]
    assert set(list_files(tmp_path, glob_pattern, recursive=True)) == set(map(str, expected_files))

    # Search for all files (with specific pattern) in all the subdirectories (from the sub directory)
    # --> file7 is not included because of different pattern
    glob_pattern = os.path.join("*", f"*.{ext}")
    expected_files = [file4, file6]
    assert set(list_files(tmp_path, glob_pattern, recursive=True)) == set(map(str, expected_files))


def create_dir_tree(base_dir):
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
        path = os.path.join(base_dir, *path_compoments)
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
    base_dir = tmp_path
    create_dir_tree(base_dir=base_dir)
    paths = [
        os.path.join(base_dir, "lon_bin=-5.0", "lat_bin=-5.0"),
        os.path.join(base_dir, "lon_bin=-5.0", "lat_bin=5.0"),
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
    base_dir = tmp_path
    create_dir_tree(base_dir=base_dir)
    path1 = os.path.join(base_dir, "lon_bin=-5.0", "lat_bin=-5.0")
    path2 = os.path.join(base_dir, "lon_bin=-5.0", "lat_bin=5.0")
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
    base_dir = tmp_path
    create_dir_tree(base_dir=base_dir)
    leaf_path1 = os.path.join("lon_bin=-5.0", "lat_bin=-5.0")
    leaf_path2 = os.path.join("lon_bin=-5.0", "lat_bin=5.0")
    path1 = os.path.join(base_dir, leaf_path1)
    path2 = os.path.join(base_dir, leaf_path2)

    paths = [path1, path2]
    leaf_paths = [leaf_path1, leaf_path2]

    # Test results without filtering
    leaf_directories_p = search_leaf_directories(base_dir=base_dir, parallel=True, remove_base_path=remove_base_path)
    leaf_directories = search_leaf_directories(base_dir=base_dir, parallel=False, remove_base_path=remove_base_path)
    assert len(leaf_directories) == 2
    assert leaf_directories == leaf_directories_p
    if remove_base_path:
        assert sorted(leaf_directories) == sorted(leaf_paths)
    else:
        assert sorted(leaf_directories) == sorted(paths)


def test_get_subdirectories(tmp_path):
    # Create the directory structure
    base_dir = tmp_path
    create_dir_tree(base_dir=base_dir)

    # Test results
    results = get_subdirectories(base_dir=base_dir, path=False)
    assert results == ["lon_bin=-5.0"]
    results = get_subdirectories(base_dir=os.path.join(base_dir, "lon_bin=-5.0"), path=False)
    expected_results = ["lat_bin=5.0", "lat_bin=-5.0"]
    assert sorted(results) == sorted(expected_results)


def test_search_leaf_files_in_parallel(tmp_path):
    # Create the directory structure
    base_dir = tmp_path
    create_dir_tree(base_dir=base_dir)

    # Test results without filtering
    filepaths_p = search_leaf_files(
        base_dir=base_dir,
        parallel=True,
    )
    filepaths = search_leaf_files(
        base_dir=base_dir,
        parallel=False,
    )
    assert len(filepaths) == 5
    assert sorted(filepaths) == sorted(filepaths_p)
