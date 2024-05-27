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
"""This module provide utilities to read GPM Geographic Buckets Apache Parquet files."""
import dask.dataframe as dd
import numpy as np
import pandas as pd


def _get_arrow_to_pandas_defaults():
    return {
        "zero_copy_only": False,  # Default is False. If True, raise error if doing copies
        "strings_to_categorical": False,
        "date_as_object": False,  # Default is True. If False convert to datetime64[ns]
        "timestamp_as_object": False,  # Default is True. If False convert to numpy.datetime64[ns]
        "use_threads": True,  #  parallelize the conversion using multiple threads.
        "safe": True,
        "split_blocks": False,
        "ignore_metadata": False,  # Default False. If False, use the pandas metadata to get the Index
        "types_mapper": pd.ArrowDtype,  # Ensure pandas is created with Arrow dtype
    }


def read_dask_partitioned_dataset(base_dir, columns=None):
    arrow_to_pandas = _get_arrow_to_pandas_defaults()
    return dd.read_parquet(
        base_dir,
        engine="pyarrow",
        dtype_backend="pyarrow",
        index=False,
        # Filtering
        columns=columns,  # Specify columns to load
        filters=None,  # Row-filtering at read-time
        # Metadata options
        calculate_divisions=False,  # Calculate divisions from metadata (set True if metadata available)
        ignore_metadata_file=True,  # True can slowdown a lot reading
        # Partitioning
        split_row_groups=False,  #   False --> Each file a partition
        # Arrow options
        arrow_to_pandas=arrow_to_pandas,
    )


def read_within_extent(bucket_dir, extent):
    partitioning = get_bucket_partitioning(bucket_dir)
    dir_trees = partitioning.directories_by_extent(extent)
    dir_paths = np.char.add(bucket_dir, dir_trees)
    # filter by existing

    # list_filepaths within directories

    # read in polars

    # convert to


# get glob pattern for read()

# backend="polars" ... dask, pandas, polars


# Readers
# - gpm.bucket.read_within_extent(bucket_dir, extent, **polars_kwargs)
# - gpm.bucket.read_within_country(bucket_dir, country, **polars_kwargs)
# - gpm.bucket.read_within_continent(bucket_dir, continent, **polars_kwargs)
# - gpm.bucket.read_around_point(bucket_dir, lon, lat, distance, size, **polars_kwargs)
# -->  compute distance on subset and select below threshold
# -->  https://stackoverflow.com/questions/76262681/i-need-to-create-a-column-with-the-distance-between-two-coordinates-in-polars


# Routines
# - Routine to repartition in smaller partitions (disaggregate bucket)
# - Routine to repartition in larger partitions (aggregate bucket)

# Analysis
# - Group by overpass
# - Reformat to dataset / Generator
