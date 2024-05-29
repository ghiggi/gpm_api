# -----------------------------------------------------------------------------.
# MIT License

# Copyright (c) 2024 GPM-API developers
#
# This file is part of GPM-API.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and_for sell
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
import os

import dask.dataframe as dd
import pandas as pd
import polars as pl

from gpm.bucket.io import (
    get_bucket_partitioning,
    get_filepaths,
    get_filepaths_within_paths,
)


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


def check_backend(backend):
    """Check backend type."""
    if backend == "dask":
        raise ValueError("Please use gpm.bucket.read_dask_partitioned_dataset instead.")
    supported_backend = ["pyarrow", "polars", "polars_lazy", "pandas"]
    if backend not in supported_backend:
        raise ValueError(f"Unsupported backend: {backend}. Supported backends are {supported_backend}.")


def _change_backend_from_polars(df, backend):
    """Change the backend of a Polars dataframe."""
    check_backend(backend)
    if backend == "pandas":
        return df.to_pandas(use_pyarrow_extension_array=True)
    if backend == "pyarrow":
        return df.to_arrow()
    return df


def _read_dataframe(source, backend, **polars_kwargs):
    """Read bucket with polars and convert to backend of choice."""
    if source is None or len(source) == 0:
        raise ValueError("No files available matching your request.")

    if backend == "polars_lazy":
        df = pl.scan_parquet(
            source=source,
            **polars_kwargs,
        )
    else:
        df = pl.read_parquet(
            source=source,
            **polars_kwargs,
        )

    # Filtering options (filters = dict)
    # bucket_filters
    # Partitions filters
    # - country/continent/point_distance, extent
    # Data filtering
    # - extent, point_distance (lon, lat, distance), start_time, end_time, month, season

    # Convert backend if necessary
    df = _change_backend_from_polars(df, backend=backend)
    return df


def _read_polars_subset(
    bucket_dir,
    dir_trees,
    backend,
    file_extension,
    glob_pattern,
    regex_pattern,
    **polars_kwargs,
):

    # Define partitions paths
    paths = [os.path.join(bucket_dir, dir_tree) for dir_tree in dir_trees]
    #  Select only existing directories
    paths = [path for path in paths if os.path.exists(path)]
    # List filepaths within directories
    filepaths = get_filepaths_within_paths(
        paths,
        parallel=True,
        file_extension=file_extension,
        glob_pattern=glob_pattern,
        regex_pattern=regex_pattern,
    )
    # Read the dataframe
    df = _read_dataframe(
        source=filepaths,
        backend=backend,
        **polars_kwargs,
    )
    return df


def read_bucket(
    bucket_dir,
    file_extension=None,
    glob_pattern=None,
    regex_pattern=None,
    backend="polars",
    **polars_kwargs,
):
    """
    Read bucket.

    https://docs.pola.rs/py-polars/html/reference/api/polars.read_parquet.html

    Parameters
    ----------
    bucket_dir : TYPE
        DESCRIPTION.
    **polars_kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    df_pl : TYPE
        DESCRIPTION.

    """
    if file_extension is None and glob_pattern is None and regex_pattern is None:
        partitioning = get_bucket_partitioning(bucket_dir)
        glob_pattern = ["*" for i in range(partitioning.n_levels + 1)]
        source = os.path.join(bucket_dir, *glob_pattern)
    else:
        source = get_filepaths(
            bucket_dir=bucket_dir,
            parallel=True,
            file_extension=file_extension,
            glob_pattern=glob_pattern,
            regex_pattern=regex_pattern,
        )
    # Read the dataframe
    df = _read_dataframe(
        source=source,
        backend=backend,
        **polars_kwargs,
    )
    return df


def read_bucket_within_extent(
    bucket_dir,
    extent,
    file_extension=None,
    glob_pattern=None,
    regex_pattern=None,
    backend="polars",
    **polars_kwargs,
):
    partitioning = get_bucket_partitioning(bucket_dir)
    dir_trees = partitioning.directories_by_extent(extent)
    return _read_polars_subset(
        bucket_dir=bucket_dir,
        dir_trees=dir_trees,
        file_extension=file_extension,
        glob_pattern=glob_pattern,
        regex_pattern=regex_pattern,
        backend=backend,
        **polars_kwargs,
    )


def read_bucket_within_country(
    bucket_dir,
    country,
    file_extension=None,
    glob_pattern=None,
    regex_pattern=None,
    backend="polars",
    **polars_kwargs,
):
    partitioning = get_bucket_partitioning(bucket_dir)
    dir_trees = partitioning.directories_by_country(country)
    return _read_polars_subset(
        bucket_dir=bucket_dir,
        dir_trees=dir_trees,
        file_extension=file_extension,
        glob_pattern=glob_pattern,
        regex_pattern=regex_pattern,
        backend=backend,
        **polars_kwargs,
    )


def read_bucket_within_continent(
    bucket_dir,
    continent,
    file_extension=None,
    glob_pattern=None,
    regex_pattern=None,
    backend="polars",
    **polars_kwargs,
):
    partitioning = get_bucket_partitioning(bucket_dir)
    dir_trees = partitioning.directories_by_continent(continent)
    return _read_polars_subset(
        bucket_dir=bucket_dir,
        dir_trees=dir_trees,
        file_extension=file_extension,
        glob_pattern=glob_pattern,
        regex_pattern=regex_pattern,
        backend=backend,
        **polars_kwargs,
    )


def read_bucket_around_point(
    bucket_dir,
    lon,
    lat,
    distance=None,
    size=None,
    file_extension=None,
    glob_pattern=None,
    regex_pattern=None,
    backend="polars",
    **polars_kwargs,
):
    partitioning = get_bucket_partitioning(bucket_dir)
    dir_trees = partitioning.directories_around_point(lon=lon, lat=lat, distance=distance, size=size)
    df = _read_polars_subset(
        bucket_dir=bucket_dir,
        dir_trees=dir_trees,
        file_extension=file_extension,
        glob_pattern=glob_pattern,
        regex_pattern=regex_pattern,
        backend=backend,
        **polars_kwargs,
    )
    return df
