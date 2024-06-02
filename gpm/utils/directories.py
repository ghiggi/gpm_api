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
import glob
import os
import pathlib


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


# def list_directories(dir_path, glob_pattern, recursive=False):
#     """Return a list of filepaths (exclude directory paths)."""
#     paths = list_paths(dir_path, glob_pattern, recursive=recursive)
#     return [f for f in paths if os.path.isdir(f)]
