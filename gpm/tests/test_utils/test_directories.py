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

from gpm.utils.directories import list_files


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
