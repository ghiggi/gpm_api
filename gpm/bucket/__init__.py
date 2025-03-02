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
"""This directory defines the GPM-API geographic binning toolbox."""
import importlib

if not importlib.util.find_spec("pyarrow"):
    raise ImportError(
        "The 'pyarrow' package is required but not found. "
        "Please install it using the following command: "
        "conda install -c conda-forge pyarrow",
    )
if not importlib.util.find_spec("polars"):
    raise ImportError(
        "The 'polars' package is required but not found. "
        "Please install it using the following command: "
        "conda install -c conda-forge polars",
    )
from gpm.bucket.partitioning import LonLatPartitioning, TilePartitioning
from gpm.bucket.readers import read_bucket as read
from gpm.bucket.routines import merge_granule_buckets, write_bucket, write_granules_bucket

__all__ = [
    "LonLatPartitioning",
    "TilePartitioning",
    "merge_granule_buckets",
    "read",
    "write_bucket",
    "write_granules_bucket",
]
