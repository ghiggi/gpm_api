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

from gpm.bucket.filters import apply_spatial_filters
from gpm.bucket.io import (
    get_bucket_partitioning,
    get_filepaths,
)
from gpm.utils.directories import get_filepaths_within_paths
from gpm.utils.geospatial import (
    get_continent_extent,
    get_country_extent,
    get_geographic_extent_around_point,
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


def _read_dataframe(source, backend, filters=None, **polars_kwargs):
    """Read bucket with polars and convert to backend of choice."""
    if source is None or len(source) == 0:
        raise ValueError("No files available matching your request.")
    is_lazy = False
    # Preprocess polars kwargs
    if "hive_partitioning" not in polars_kwargs:
        polars_kwargs["hive_partitioning"] = False
    # Read dataframe with polars
    if "columns" not in polars_kwargs:  # backend == "polars_lazy":
        is_lazy = True
        df = pl.scan_parquet(
            source=source,
            **polars_kwargs,
        )
    else:
        df = pl.read_parquet(
            source=source,
            **polars_kwargs,
        )

    # Apply spatial filtering to remove unrelevant data within partitions
    df = apply_spatial_filters(df, filters=filters)

    # Put data into memory if not polars lazy
    if is_lazy and backend != "polars_lazy":
        df = df.collect()
        if df.shape[0] == 0:
            raise ValueError("No data match your request.")

    # Convert backend if necessary
    df = _change_backend_from_polars(df, backend=backend)

    return df


def read_bucket(
    bucket_dir,
    # Spatial filters
    extent=None,
    country=None,
    continent=None,
    point=None,
    distance=None,
    size=None,
    padding=0,
    # Filename filters
    file_extension=None,
    glob_pattern=None,
    regex_pattern=None,
    # Dataframe output
    backend="polars",
    # Reader arguments
    **polars_kwargs,
):
    """
    Read a geographic bucket.

    The ``extent``, ``country``, ``continent``, or ``point`` arguments allows to read only a spatial subset
    of the original bucket. Please specify only one of this arguments !

    The ``file_extension``, ``glob_pattern`` and ``regex_pattern`` arguments allows to further restrict the
    selection of files read from the partitioned dataset.

    Parameters
    ----------
    bucket_dir : str
        Base directory of the geographic bucket.
    extent: list, optional
        The extent specified as [xmin, xmax, ymin, ymax].
    country: str, optional
        The name of the country of interest.
    continent: str, optional
        The name of the continent of interest.
    point: list or tuple, optional
       The longitude and latitude coordinates of the point around which you are interested to get the data.
       To effectively subset data around this point, also specify ``size`` or ``distance`` arguments.
       Longitude of the point.
    distance: float, optional
        Distance (in meters) from the specified point in each direction.
    size: int, float, tuple, list, optional
        The size in degrees of the extent in each direction centered around the specified point.
    padding : int, float, tuple, list
        The number of degrees to extend the (country, continent) extent in each direction.
        If padding is a single number, the same padding is applied in all directions.
        If padding is a tuple or list, it must contain 2 or 4 elements.
        If two values are provided (x, y), they are interpreted as longitude and latitude padding, respectively.
        If four values are provided, they directly correspond to padding for each side (left, right, top, bottom).
        Default is 0.
    file_extension : str, optional
        Name of the file extension. The default is ``None``.
    glob_pattern : str, optional
        Unix shell-style wildcards to subset the files to read in. The default is ``None``.
    regex_pattern : str, optional
        Regex pattern to subset the files to read in. The default is ``None``.
    backend : str, optional
        The wished type of dataframe returned by the function.
        The default is a polars.DataFrame.
        Valid backends are ``pandas``, ``polars_lazy`` and ``pyarrow``.
    **polars_kwargs : dict
        Arguments to be passed to polars.read_parquet()
        ``columns`` allow to specify the subset of columns to read.
        ``n_rows`` allows to stop reading data from Parquet files after reading n_rows.
        For other arguments, please refer to:  https://docs.pola.rs/py-polars/html/reference/api/polars.read_parquet.html

    Returns
    -------
    df : pandas.DataFrame, polars.DataFrame, polars.LazyFrame or pyarrow.Table
        Bucket dataframe.

    """
    # Check if a spatial_filter option is specified
    spatial_filter_options = [extent, country, continent, point]
    specified_filters = [opt for opt in spatial_filter_options if opt is not None]
    if len(specified_filters) > 1:
        raise ValueError("Specify only one between extent, country, continent, and point arguments.")
    if specified_filters:
        # Read partitioning class
        partitioning = get_bucket_partitioning(bucket_dir)
        # Infer dir_tree based on the specified spatial filters
        if extent is not None:
            dir_trees = partitioning.directories_by_extent(extent)
            filters = {"extent": extent}
        elif country is not None:
            extent = get_country_extent(name=country, padding=padding)
            dir_trees = partitioning.directories_by_extent(extent)
            filters = {"extent": extent}
        elif continent is not None:
            extent = get_continent_extent(name=continent, padding=padding)
            dir_trees = partitioning.directories_by_extent(extent)
            filters = {"extent": extent}
        elif point is not None:
            lon, lat = point
            extent = get_geographic_extent_around_point(
                lon=lon,
                lat=lat,
                distance=distance,
                size=size,
            )
            dir_trees = partitioning.directories_by_extent(extent)
            filters = {"point_radius": (lon, lat, distance)} if distance else {"extent": extent}
        # Define partitions paths
        paths = [os.path.join(bucket_dir, dir_tree) for dir_tree in dir_trees]
        #  Select only existing directories
        paths = [path for path in paths if os.path.exists(path)]
        # List filepaths within directories
        source = get_filepaths_within_paths(
            paths,
            parallel=True,
            file_extension=file_extension,
            glob_pattern=glob_pattern,
            regex_pattern=regex_pattern,
        )
    else:
        filters = {}
        # If no filename filtering, specify a glob pattern across all the partitioned dataset
        if file_extension is None and glob_pattern is None and regex_pattern is None:
            partitioning = get_bucket_partitioning(bucket_dir)
            glob_pattern = ["*" for i in range(partitioning.n_levels + 1)]
            source = os.path.join(bucket_dir, *glob_pattern)
        # Alternatively search for files that match the desired criteria
        else:
            source = get_filepaths(
                bucket_dir=bucket_dir,
                parallel=True,
                file_extension=file_extension,
                glob_pattern=glob_pattern,
                regex_pattern=regex_pattern,
            )
    # Read the dataframe
    return _read_dataframe(source=source, backend=backend, filters=filters, **polars_kwargs)
