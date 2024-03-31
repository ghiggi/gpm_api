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
"""This module provide utilities for the creation of GPM Geographic Buckets."""
import math
import os
import time

import dask
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset
import xarray as xr
from tqdm import tqdm

from gpm.bucket.io import get_filepaths_by_bin
from gpm.dataset.granule import remove_unused_var_dims
from gpm.io.local import group_filepaths_by_time_group
from gpm.utils.timing import print_task_elapsed_time


def has_unique_chunking(ds):
    """Check if a dataset has unique chunking."""
    if not isinstance(ds, xr.Dataset):
        raise ValueError("Input must be an xarray Dataset.")

    # Create a dictionary to store unique chunk shapes for each dimension
    unique_chunks_per_dim = {}

    # Iterate through each variable's chunks
    for var_name in ds.variables:
        if hasattr(ds[var_name].data, "chunks"):  # is dask array
            var_chunks = ds[var_name].data.chunks
            for dim, chunks in zip(ds[var_name].dims, var_chunks):
                if dim not in unique_chunks_per_dim:
                    unique_chunks_per_dim[dim] = set()
                    unique_chunks_per_dim[dim].add(chunks)
                if chunks not in unique_chunks_per_dim[dim]:
                    return False

    # If all chunks are unique for each dimension, return True
    return True


def ensure_unique_chunking(ds):
    """Ensure the dataset has unique chunking.

    Conversion to dask.dataframe requires unique chunking.
    If the xr.Dataset does not have unique chunking, perform ds.unify_chunks.

    Variable chunks can be visualized with:

    for var in ds.data_vars:
        print(var, ds[var].chunks)

    """
    if not has_unique_chunking(ds):
        ds = ds.unify_chunks()

    return ds


def get_df_object_columns(df):
    """Get the dataframe columns which have 'object' type."""
    return list(df.select_dtypes(include="object").columns)


def ensure_pyarrow_string_columns(df):
    """Convert 'object' type columns to pyarrow strings."""
    for column in get_df_object_columns(df):
        df[column] = df[column].astype("string[pyarrow]")
    return df


def drop_undesired_columns(df):
    """Drop undesired columns like dataset dimensions without coordinates."""
    undesired_columns = ["cross_track", "along_track", "crsWGS84"]
    undesired_columns = [column for column in undesired_columns if column in df.columns]
    return df.drop(columns=undesired_columns)


def ds_to_pd_df_function(ds):
    """Convert an xr.Dataset to a pandas.Dataframe.

    This function expects a xr.Dataset with only 2D spatial DataArrays.
    """
    # Drop unrelevant coordinates
    ds = remove_unused_var_dims(ds)

    # Convert to pandas dataframe
    # - strings are converted to object !
    df = ds.to_dataframe(dim_order=None)

    # Convert object columns to pyarrow string
    df = ensure_pyarrow_string_columns(df)

    # Remove MultiIndex
    df = df.reset_index()

    # Drop unrequired columns (previous dataset dimensions)
    return drop_undesired_columns(df)


def ds_to_dask_df_function(ds):
    """Convert an xr.Dataset to a dask.Dataframe.

    This function expects a xr.Dataset with only 2D spatial DataArrays.
    """
    # Drop unrelevant coordinates
    ds = remove_unused_var_dims(ds)

    # Check dataset uniform chunking
    ds = ensure_unique_chunking(ds)

    # Convert to to dask dataframe
    # - strings are converted to object !
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        df = ds.to_dask_dataframe(dim_order=None, set_index=False)

    # Convert object columns to pyarrow string
    df = ensure_pyarrow_string_columns(df)

    # Drop unrequired columns (previous dataset dimensions)
    return drop_undesired_columns(df)


####--------------------------------------------------------------------------------------------------------------------.


def get_bin_partition(values, bin_size):
    """Compute the bins partitioning values.

    Parameters
    ----------
    values : float or array-like
        Values.
    bin_size : float
        Bin size.

    Returns
    -------
    Bin value : float or array-like
        DESCRIPTION.

    """
    return bin_size * np.floor(values / bin_size)


# bin_size = 10
# values = np.array([-180,-176,-175, -174, -171, 170, 166])
# get_bin_partition(values, bin_size)


def assign_spatial_partitions(
    df,
    xbin_name,
    ybin_name,
    xbin_size,
    ybin_size,
    x_column="lat",
    y_column="lon",
):
    """Add partitioning bin columns to dataframe.

    Works for both dask.dataframe and pandas.dataframe.
    """
    # Remove invalid coordinates
    df = df[~df[x_column].isna()]
    df = df[~df[y_column].isna()]

    # Add spatial partitions columns to dataframe
    partition_columns = {
        xbin_name: get_bin_partition(df[x_column], bin_size=xbin_size),
        ybin_name: get_bin_partition(df[y_column], bin_size=ybin_size),
    }
    return df.assign(**partition_columns)


def _convert_size_to_bytes(size_str):
    """Convert human filesizes to bytes.

    Special cases:
     - singular units, e.g., "1 byte"
     - byte vs b
     - yottabytes, zetabytes, etc.
     - with & without spaces between & around units.
     - floats ("5.2 mb")

    To reverse this, see hurry.filesize or the Django filesizeformat template
    filter.

    :param size_str: A human-readable string representing a file size, e.g.,
    "22 megabytes".
    :return: The number of bytes represented by the string.
    """
    multipliers = {
        "kilobyte": 1024,
        "megabyte": 1024**2,
        "gigabyte": 1024**3,
        "terabyte": 1024**4,
        "petabyte": 1024**5,
        "exabyte": 1024**6,
        "zetabyte": 1024**7,
        "yottabyte": 1024**8,
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
        "tb": 1024**4,
        "pb": 1024**5,
        "eb": 1024**6,
        "zb": 1024**7,
        "yb": 1024**8,
    }

    for suffix in multipliers:
        size_str = size_str.lower().strip().strip("s")
        if size_str.lower().endswith(suffix):
            return int(float(size_str[0 : -len(suffix)]) * multipliers[suffix])
    if size_str.endswith("b"):
        size_str = size_str[0:-1]
    elif size_str.endswith("byte"):
        size_str = size_str[0:-4]
    return int(size_str)


# def test_filesize_conversions(self):
#         """Can we convert human filesizes to bytes?"""
#         qa_pairs = [
#             ('58 kb', 59392),
#             ('117 kb', 119808),
#             ('117kb', 119808),
#             ('1 byte', 1),
#             ('1 b', 1),
#             ('117 bytes', 117),
#             ('117  bytes', 117),
#             ('  117 bytes  ', 117),
#             ('117b', 117),
#             ('117bytes', 117),
#             ('1 kilobyte', 1024),
#             ('117 kilobytes', 119808),
#             ('0.7 mb', 734003),
#             ('1mb', 1048576),
#             ('5.2 mb', 5452595),
#         ]
#         for qa in qa_pairs:
#             print("Converting '%s' to bytes..." % qa[0], end='')
#             self.assertEqual(convert_size_to_bytes(qa[0]), qa[1])
#             print('✓')


def convert_size_to_bytes(size):
    if not isinstance(size, (str, int)):
        raise TypeError("Expecting a string (i.e. 200MB) or the integer number of bytes.")
    if isinstance(size, int):
        return size
    try:
        size = _convert_size_to_bytes(size)
    except Exception:
        raise ValueError("Impossible to parse {size_str} to the number of bytes.")
    return size


def estimate_row_group_size(df, size="200MB"):
    """Estimate row_group_size parameter based on the desired row group memory size.

    row_group_size is a Parquet argument controlling the number of rows
    in each Apache Parquet File Row Group.
    """
    if isinstance(df, pa.Table):
        memory_used = df.nbytes
    elif isinstance(df, pd.DataFrame):
        memory_used = df.memory_usage().sum()
    else:
        raise NotImplementedError("Unrecognized dataframe type")
    size_bytes = convert_size_to_bytes(size)
    n_rows = len(df)
    memory_per_row = memory_used / n_rows
    return math.floor(size_bytes / memory_per_row)


@print_task_elapsed_time(prefix="Bucket Merging Terminated.")
def merge_granule_buckets(
    src_bucket_dir,
    dst_bucket_dir,
    row_group_size="400MB",
    max_file_size="2GB",
    compression="snappy",
    compression_level=None,
    # Computing options
    max_open_files=0,
    use_threads=True,
    # Scanner options
    batch_size=131_072,
    batch_readahead=16,
    fragment_readahead=4,
):
    """Merge the per-granule bucket archive in a single optimized archive.

    Parameters
    ----------
    src_bucket_dir : str
        Base directory of the per-granule bucket archive.
    dst_bucket_dir : str
        Directory path of the final bucket archive.
    row_group_size : (int, str), optional
        Maximum number of rows in each written Parquet row group.
        If specified as a string (i.e. "400 MB"), the equivalent row group size
        number is estimated. The default is "400MB".
    max_file_size: str, optional
        The maximum size of each Parquet File. Ideally a multiple of row_group_size.
        The default is "2GB".
    compression : str, optional
        Specify the compression codec, either on a general basis or per-column.
        Valid values: {"none", "snappy", "gzip", "brotli", "lz4", "zstd"}.
        The default is "snappy".
    compression : int or dict, optional
        Specify the compression level for a codec, either on a general basis or per-column.
        If ``None`` is passed, arrow selects the compression level for the compression codec in use.
        The compression level has a different meaning for each codec, so you have
        to read the pyArrow documentation of the codec you are using.
        The default is compression_level=None.
    max_open_files, int, optional
        If greater than 0 then this will limit the maximum number of files that can be left open.
        If an attempt is made to open too many files then the least recently used file will be closed.
        If this setting is set too low you may end up fragmenting your data into many small files.
        The default is 0.
        Note that Linux has a default limit of 1024. Before starting the python session,
        increase it with ulimit -n <new_much_higher_limit>.
    use_threads: bool, optional
        If enabled, then maximum parallelism will be used to read and write files (in multithreading).
        The number of threads is determined by the number of available CPU cores.
        The default is ``True``.
    batch_size, int, optional
        The maximum row count for scanned record batches.
        If scanned record batches are overflowing memory then this value can be reduced to reduce the memory usage.
        The default value is 131_072.
    batch_readahead
        The number of batches to read ahead in a file.
        Increasing this number will increase RAM usage but could also improve IO utilization.
        The default is 16.
    fragment_readahead
        The number of files to read ahead.
        Increasing this number will increase RAM usage but could also improve IO utilization.
        The default is 4.

    Returns
    -------
    None.

    """
    # Identify Parquet filepaths for each bin
    print("Searching of Parquet files has started.")
    t_i = time.time()
    bin_path_dict = get_filepaths_by_bin(src_bucket_dir)
    n_geographic_bins = len(bin_path_dict)
    t_f = time.time()
    t_elapsed = round((t_f - t_i) / 60, 1)
    print(f"Searching of Parquet files ended. Elapsed time: {t_elapsed} minutes.")
    print(f"{n_geographic_bins} geographic bins to process.")

    # Retrieve list of bins
    list_bin_names = list(bin_path_dict.keys())

    # Retrieve table schema
    template_filepath = bin_path_dict[list_bin_names[0]][0]
    table = pa.parquet.read_table(template_filepath)

    # Estimate row_group_size (in number of rows)
    if isinstance(row_group_size, str):  # "200 MB"
        row_group_size = estimate_row_group_size(table, size=row_group_size)
        max_rows_per_group = row_group_size
        min_rows_per_group = row_group_size

    # Estimate maximum number of file row (in number of rows)
    max_rows_per_file = estimate_row_group_size(table, size=max_file_size)

    # # Define file visitor for metadata collection
    # metadata_collector = []

    # def file_visitor(written_file):
    #     metadata_collector.append(written_file.metadata)

    print("Start concatenating the granules bucket archive")

    # Concatenate data within bins
    # - Cannot rewrite directly the full pyarrow.dataset because there is no way to specify when
    #    data from each partition have been scanned completely (and can be written to disk)

    # bin_id = "latbin=0/lonbin=10"
    # filepaths = bin_path_dict[bin_id]
    n_bins = len(bin_path_dict)
    for bin_id, filepaths in tqdm(bin_path_dict.items(), total=n_bins):
        partition_dir = os.path.join(dst_bucket_dir, bin_id)
        year_dict = group_filepaths_by_time_group(filepaths, group="year")
        for year, year_filepaths in year_dict.items():
            basename_template = f"{year}_" + "{i}.parquet"
            # Read Dataset
            dataset = pyarrow.dataset.dataset(year_filepaths, format="parquet")

            # Define scanner
            scanner = dataset.scanner(
                batch_size=batch_size,
                batch_readahead=batch_readahead,
                fragment_readahead=fragment_readahead,
                use_threads=use_threads,
            )

            # Define file options
            file_options = {}
            file_options["compression"] = compression
            file_options["compression_level"] = compression_level
            file_options["write_statistics"] = True

            parquet_format = pa.dataset.ParquetFileFormat()
            file_options = parquet_format.make_write_options(**file_options)

            # Rewrite dataset
            pa.dataset.write_dataset(
                scanner,
                base_dir=partition_dir,
                format="parquet",
                basename_template=basename_template,
                # Directory options
                create_dir=True,
                existing_data_behavior="overwrite_or_ignore",
                # Options
                use_threads=use_threads,
                file_options=file_options,
                # file_visitor=file_visitor,
                # Options for files size/rows
                max_rows_per_file=max_rows_per_file,
                min_rows_per_group=min_rows_per_group,
                max_rows_per_group=max_rows_per_group,
                # Options to control open connections
                max_open_files=max_open_files,
            )

    # # Write the metadata
    # print("Writing the metadata")
    # # Write the ``_common_metadata`` parquet file without row groups statistics
    # pq.write_metadata(table_schema, os.path.join(dst_bucket_dir, "_common_metadata"))

    # # Write the ``_metadata`` parquet file with row groups statistics of all files
    # pq.write_metadata(
    #     table_schema,
    #     os.path.join(dst_bucket_dir, "_metadata"),
    #     metadata_collector=metadata_collector,
    # )
