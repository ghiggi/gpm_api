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
"""This module provides the routines for the creation of GPM Geographic Buckets."""
import os
import time

import dask
import pyarrow as pa
import pyarrow.dataset
import pyarrow.parquet as pq
from tqdm import tqdm

from gpm.bucket.io import get_bucket_partitioning, get_filepaths_by_partition, write_bucket_info
from gpm.bucket.writers import preprocess_writer_kwargs, write_dataset_metadata, write_partitioned_dataset
from gpm.io.info import group_filepaths
from gpm.utils.dask import clean_memory, get_client
from gpm.utils.parallel import compute_list_delayed
from gpm.utils.timing import print_task_elapsed_time

####--------------------------------------------------------------------------------------------------.
#### Bucket Granules


def split_list_in_blocks(values, block_size):
    return [values[i : i + block_size] for i in range(0, len(values), block_size)]


def write_granule_bucket(
    src_filepath,
    bucket_dir,
    partitioning,
    granule_to_df_func,
    x="lon",
    y="lat",
    # Writer kwargs
    **writer_kwargs,
):
    """Write a geographically partitioned Parquet Dataset of a GPM granules.

    Parameters
    ----------
    src_filepath : str
        File path of the granule to store in the bucket archive.
    bucket_dir: str
        Base directory of the per-granule bucket archive.
    partitioning: `gpm.bucket.SpatialPartitioning`
        A spatial partitioning class.
    granule_to_df_func : Callable
        Function taking a granule filepath, opening it and returning a pandas or dask dataframe.
    x: str
        The name of the x column. The default is "lon".
    y: str
        The name of the y column. The default is "lat".
    **writer_kwargs: dict
        Optional arguments to be passed to the pyarrow Dataset Writer.
        Common arguments are 'format' and 'use_threads'.
        The default file ``format`` is ``'parquet'``.
        The default ``use_threads`` is ``True``, which enable multithreaded file writing.
        More information available at https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html

    """
    # Define unique prefix name so to add files to the bucket archive
    # - This prevent risk of overwriting
    # - If df is pandas.dataframe -->  f"{filename_prefix}_" + "{i}.parquet"
    # - if df is a dask.dataframe -->  f"{filename_prefix}_dask.partition_{part_index}"
    filename_prefix = os.path.splitext(os.path.basename(src_filepath))[0]

    # Retrieve dataframe
    df = granule_to_df_func(src_filepath)

    # Add partitioning columns
    df = partitioning.add_labels(df=df, x=x, y=y)

    # Write partitioned dataframe
    write_partitioned_dataset(
        df=df,
        base_dir=bucket_dir,
        filename_prefix=filename_prefix,
        partitions=partitioning.order,
        partitioning_flavor=partitioning.flavor,
        **writer_kwargs,
    )


def _try_write_granule_bucket(**kwargs):
    try:
        # synchronous
        with dask.config.set(scheduler="single-threaded"):
            write_granule_bucket(**kwargs)
            # If works, return None
            info = None
    except Exception as e:
        # Define tuple to return
        src_filepath = kwargs["src_filepath"]
        info = src_filepath, str(e)
    return info


@print_task_elapsed_time(prefix="Granules Bucketing Operation Terminated.")
def write_granules_bucket(
    filepaths,
    # Bucket configuration
    bucket_dir,
    partitioning,
    granule_to_df_func,
    # Processing options
    parallel=True,
    max_concurrent_tasks=None,
    max_dask_total_tasks=500,
    # Writer kwargs
    row_group_size="500MB",
    **writer_kwargs,
):
    """Write a geographically partitioned Parquet Dataset of GPM granules.

    Parameters
    ----------
    filepaths : str
        File paths of the GPM granules to store in the bucket archive.
    bucket_dir: str
        Base directory of the per-granule bucket archive.
    partitioning: `gpm.bucket.SpatialPartitioning`
        A spatial partitioning class.
        Carefully consider the size of the partitions.
        Earth partitioning by:
        - 1° degree corresponds to 64800 directories (360*180)
        - 5° degree corresponds to 2592 directories (72*36)
        - 10° degree corresponds to 648 directories (36*18)
        - 15° degree corresponds to 288 directories (24*12)
    granule_to_df_func : callable
        Function taking a granule filepath, opening it and returning a pandas or dask dataframe.
    parallel : bool
        Whether to bucket several granules in parallel.
        The default is ``True``.
    max_concurrent_tasks : int
        The maximum number of Dask tasks to be concurrently executed.
        If ``None``, let the Dask Scheduler to choose.
        The default is ``None``.
    max_dask_total_tasks : int
        The maximum number of Dask tasks to be scheduled.
        The default is 500.
    row_group_size : int or str, optional
        Maximum number of rows in each written Parquet row group.
        If specified as a string (i.e. "500 MB"), the equivalent row group size
        number is estimated. The default is "500MB".
    **writer_kwargs: dict
        Optional arguments to be passed to the pyarrow Dataset Writer.
        Common arguments are ``format`` and ``use_threads``.
        The default file ``format`` is ``parquet``.
        The default ``use_threads`` is ``True``, which enable multithreaded file writing.
        More information available at https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html

    """
    # Define flavor of directory partitioning
    writer_kwargs["row_group_size"] = row_group_size

    # Write down the information of the bucket
    write_bucket_info(bucket_dir=bucket_dir, partitioning=partitioning)

    # Split long list of files in blocks
    list_blocks = split_list_in_blocks(filepaths, block_size=max_dask_total_tasks)

    # Execute tasks by blocks to avoid dask overhead
    n_blocks = len(list_blocks)

    for i, block_filepaths in enumerate(list_blocks):
        print(f"Executing tasks block {i+1}/{n_blocks}")

        # Loop over granules
        func = dask.delayed(_try_write_granule_bucket) if parallel else _try_write_granule_bucket

        list_results = [
            func(
                src_filepath=src_filepath,
                bucket_dir=bucket_dir,
                partitioning=partitioning,
                granule_to_df_func=granule_to_df_func,
                # Writer kwargs
                **writer_kwargs,
            )
            for src_filepath in block_filepaths
        ]

        # If delayed, execute the tasks
        if parallel:
            list_results = compute_list_delayed(
                list_results,
                max_concurrent_tasks=max_concurrent_tasks,
            )

        # Process results to detect errors
        list_errors = [error_info for error_info in list_results if error_info is not None]
        for src_filepath, error_str in list_errors:
            print(f"An error occurred while processing {src_filepath}: {error_str}")

        # If parallel=True, retrieve client, clean the memory and restart
        if parallel:
            client = get_client()
            clean_memory(client)
            client.restart()


####--------------------------------------------------------------------------------------------------.
#### Bucket DataFrame
@print_task_elapsed_time(prefix="Dataset Bucket Operation Terminated.")
def write_bucket(
    df,
    bucket_dir,
    partitioning,
    x="lon",
    y="lat",
    # Writer arguments
    filename_prefix="part",
    row_group_size="500MB",
    **writer_kwargs,
):
    """
    Write a geographically partitioned Parquet Dataset.

    Parameters
    ----------
    ds : `pandas.DataFrame` or `dask.DataFrame`
        Pandas or Dask dataframe to be written into a geographic bucket.
    bucket_dir: str
        Base directory of the geographic bucket archive.
    partitioning: `gpm.bucket.SpatialPartitioning`
        A spatial partitioning class.
        Carefully consider the size of the partitions.
        Earth partitioning by:
        - 1° degree corresponds to 64800 directories (360*180)
        - 5° degree corresponds to 2592 directories (72*36)
        - 10° degree corresponds to 648 directories (36*18)
        - 15° degree corresponds to 288 directories (24*12)
    x: str
        The name of the x column. The default is "lon".
    y: str
        The name of the y column. The default is "lat".
    row_group_size : int or str, optional
        Maximum number of rows in each written Parquet row group.
        If specified as a string (i.e. "500 MB"), the equivalent row group size
        number is estimated. The default is "500MB".
    **writer_kwargs: dict
        Optional arguments to be passed to the pyarrow Dataset Writer.
        Common arguments are 'format' and 'use_threads'.
        The default file ``format`` is ``'parquet'``.
        The default ``use_threads`` is ``True``, which enable multithreaded file writing.
        More information available at https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html

    """
    # Write down the information of the bucket
    write_bucket_info(
        bucket_dir=bucket_dir,
        partitioning=partitioning,
    )

    # Add partitioning columns
    df = partitioning.add_labels(df=df, x=x, y=y)

    # Write bucket
    writer_kwargs["row_group_size"] = row_group_size
    write_partitioned_dataset(
        df=df,
        base_dir=bucket_dir,
        partitions=partitioning.order,
        partitioning_flavor=partitioning.flavor,
        filename_prefix=filename_prefix,
        **writer_kwargs,
    )


####--------------------------------------------------------------------------------------------------.
#### Merge Granules


@print_task_elapsed_time(prefix="Bucket Merging Terminated.")
def merge_granule_buckets(
    src_bucket_dir,
    dst_bucket_dir,
    row_group_size="400MB",
    max_file_size="1GB",
    compression="snappy",
    compression_level=None,
    write_metadata=False,
    write_statistics=False,
    # Computing options
    max_open_files=0,
    use_threads=True,
    # Scanner options
    batch_size=131_072,
    batch_readahead=16,
    fragment_readahead=4,
):
    """Merge the per-granule bucket archive in a single optimized archive.

    Set ulimit -n 999999 before running this routine !

    Parameters
    ----------
    src_bucket_dir : str
        Base directory of the per-granule bucket archive.
    dst_bucket_dir : str
        Directory path of the final bucket archive.
    row_group_size : int or str, optional
        Maximum number of rows to be written in each Parquet row group.
        If specified as a string (i.e. ``"400 MB"``), the equivalent number of rows is estimated.
        The default is ``"400MB"``.
    max_file_size: str, optional
        Maximum number of rows to be written in a Parquet file.
        If specified as a string, the equivalent number of rows is estimated.
        Ideally the value should be a multiple of ``row_group_size``.
        The default is ``"2GB"``.
    compression : str, optional
        Specify the compression codec, either on a general basis or per-column.
        Valid values: ``{"none", "snappy", "gzip", "brotli", "lz4", "zstd"}``.
        The default is ``snappy``.
    compression_level : int or dict, optional
        Specify the compression level for a codec, either on a general basis or per-column.
        If ``None`` is passed, arrow selects the compression level for the compression codec in use.
        The compression level has a different meaning for each codec, so you have
        to read the pyArrow documentation of the codec you are using at
        https://arrow.apache.org/docs/python/generated/pyarrow.Codec.html
        The default is ``None``.
    max_open_files, int, optional
        If greater than 0 then this will limit the maximum number of files that can be left open.
        If an attempt is made to open too many files then the least recently used file will be closed.
        If this setting is set too low you may end up fragmenting your data into many small files.
        The default is ``0``.
        Note that Linux has a default limit of ``1024``. Before starting the python session,
        increase it with ``ulimit -n <new_much_higher_limit>``.
    use_threads: bool, optional
        If enabled, then maximum parallelism will be used to read and write files (in multithreading).
        The number of threads is determined by the number of available CPU cores.
        The default is ``True``.
    batch_size, int, optional
        The maximum row count for scanned record batches.
        If scanned record batches are overflowing memory then this value can be reduced to reduce the memory usage.
        The default value is ``131_072``.
    batch_readahead
        The number of batches to read ahead in a file.
        Increasing this number will increase RAM usage but could also improve IO utilization.
        The default is ``16``.
    fragment_readahead
        The number of files to read ahead.
        Increasing this number will increase RAM usage but could also improve IO utilization.
        The default is ``4``.

    Returns
    -------
    None.

    """
    # Retrieve partitioning class
    partitioning = get_bucket_partitioning(bucket_dir=src_bucket_dir)

    # Identify Parquet filepaths for each bin
    print("Searching of Parquet files has started.")
    t_i = time.time()
    dict_partition_files = get_filepaths_by_partition(src_bucket_dir, parallel=True)
    n_geographic_bins = len(dict_partition_files)
    t_f = time.time()
    t_elapsed = round((t_f - t_i) / 60, 1)
    print(f"Searching of Parquet files ended. Elapsed time: {t_elapsed} minutes.")
    print(f"{n_geographic_bins} geographic partitions to process.")

    # Retrieve list of partitions
    list_partitions = list(dict_partition_files.keys())

    # Write the new partitioning class
    # TODO: add option maybe to provide new partitioning to this routine !
    # --> Will require to load data into memory inside a partition (instead of scanner) !
    # --> Check that new partitioning is aligned and subset of original partitioning?
    write_bucket_info(bucket_dir=dst_bucket_dir, partitioning=partitioning)

    # -----------------------------------------------------------------------------------------------.
    # Retrieve table schema
    template_filepath = dict_partition_files[list_partitions[0]][0]
    template_table = pq.read_table(template_filepath)
    schema = template_table.schema

    # Define writer_kwargs
    writer_kwargs = {}
    writer_kwargs["row_group_size"] = row_group_size
    writer_kwargs["max_file_size"] = max_file_size
    writer_kwargs["compression"] = compression
    writer_kwargs["compression_level"] = compression_level
    writer_kwargs["max_open_files"] = max_open_files
    writer_kwargs["use_threads"] = use_threads
    writer_kwargs["write_metadata"] = write_metadata
    writer_kwargs["write_statistics"] = write_statistics
    writer_kwargs, metadata_collector = preprocess_writer_kwargs(
        writer_kwargs=writer_kwargs,
        df=template_table,
    )

    # -----------------------------------------------------------------------------------------------.
    # Concatenate data within bins
    # - Cannot rewrite directly the full pyarrow.dataset because there is no way to specify when
    #    data from each partition have been scanned completely (and can be written to disk)
    print("Start concatenating the granules bucket archive")
    # partition_label = "latbin=0/lonbin=10"
    # filepaths = dict_partition_files[partition_label]
    n_partitions = len(dict_partition_files)
    for partition_label, filepaths in tqdm(dict_partition_files.items(), total=n_partitions):
        partition_dir = os.path.join(dst_bucket_dir, partition_label)
        year_dict = group_filepaths(filepaths, groups="year")
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

            # Rewrite dataset
            pa.dataset.write_dataset(
                scanner,
                base_dir=partition_dir,
                basename_template=basename_template,
                # Directory options
                create_dir=True,
                existing_data_behavior="overwrite_or_ignore",
                # Options
                **writer_kwargs,
            )

    if metadata_collector:
        write_dataset_metadata(base_dir=dst_bucket_dir, metadata_collector=metadata_collector, schema=schema)
