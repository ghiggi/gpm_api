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
"""This module provide utilities to write GPM Geographic Buckets Apache Parquet files."""
import os

import dask

from gpm.bucket.dataset import write_partitioned_dataset
from gpm.bucket.processing import assign_spatial_partitions

# from gpm.io.info import get_key_from_filepath
from gpm.utils.dask import clean_memory, get_client
from gpm.utils.parallel import compute_list_delayed
from gpm.utils.timing import print_task_elapsed_time

####--------------------------------------------------------------------------.
#### Single GPM Dataset Routines


@print_task_elapsed_time(prefix="Dataset Bucket Operation Terminated.")
def write_dataset_bucket(
    ds,
    bucket_filepath,
    ds_to_df_converter,
    # Partitioning arguments
    xbin_size=15,
    ybin_size=15,
    xbin_name="lonbin",
    ybin_name="latbin",
    # Writer arguments
    filename_prefix="part",
    row_group_size="500MB",
    **writer_kwargs,
):
    """Write a geographically partitioned Parquet Dataset of a GPM Dataset."""
    # Retrieve dataframe
    df = ds_to_df_converter(ds)

    # Define partitioning columns names
    partitioning = [xbin_name, ybin_name]

    # Add partitioning columns
    df = assign_spatial_partitions(
        df=df,
        x_column="lon",
        y_column="lat",
        xbin_name=xbin_name,
        ybin_name=ybin_name,
        xbin_size=xbin_size,
        ybin_size=ybin_size,
    )

    # Write bucket
    write_partitioned_dataset(
        df=df,
        base_dir=bucket_filepath,
        partitioning=partitioning,
        row_group_size=row_group_size,
        filename_prefix=filename_prefix**writer_kwargs,
    )


####--------------------------------------------------------------------------.
#### Single GPM Granule Routines


def write_granule_bucket(
    src_filepath,
    bucket_base_dir,
    ds_to_df_converter,
    # Partitioning arguments
    xbin_size=15,
    ybin_size=15,
    xbin_name="lonbin",
    ybin_name="latbin",
    # Writer kwargs
    row_group_size="500MB",
    **writer_kwargs,
):
    """Write a geographically partitioned Parquet Dataset of a GPM granules.

    Parameters
    ----------
    src_filepath : str
        File path of the granule to store in the bucket archive.
    bucket_base_dir: str
        Base directory of the per-granule bucket archive.
    ds_to_df_converter : callable,
        Function taking a granule filepath, opening it and returning a pandas or dask dataframe.
    xbin_name : str, optional
        Name of the binned column used to partition the data along the x dimension.
        The default is ``"lonbin"``.
    ybin_name : str, optional
        Name of the binned column used to partition the data along the y dimension.
        The default is ``"latbin"``.
    xbin_size : int
        Longitude bin size. The default is 15.
    xbin_size : int
        Latitude bin size. The default is 15.
    row_group_size : (int, str), optional
        Maximum number of rows in each written Parquet row group.
        If specified as a string (i.e. ``"500 MB"``), the equivalent row group size
        number is estimated. The default is ``"500MB"``.
    **writer_kwargs: dict
        Optional arguments to be passed to the pyarrow Dataset Writer.
        Common arguments are 'format' and 'use_threads'.
        The default file ``format`` is ``'parquet'``.
        The default ``use_threads`` is ``True``, which enable multithreaded file writing.
        More information available at https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html

    Notes
    -----
    Example of partitioning:

    - Partition by 1° degree pixels: 64800 directories (360*180)
    - Partition by 5° degree pixels: 2592 directories (72*36)
    - Partition by 10° degree pixels: 648 directories (36*18)
    - Partition by 15° degree pixels: 288 directories (24*12)

    """
    # Define unique prefix name so to add files to the bucket archive
    # - This prevent risk of overwriting
    # - If df is pandas.dataframe -->  f"{filename_prefix}_" + "{i}.parquet"
    # - if df is a dask.dataframe -->  f"{filename_prefix}_dask.partition_{part_index}"
    filename_prefix = os.path.splitext(os.path.basename(src_filepath))[0]

    # Retrieve dataframe
    df = ds_to_df_converter(src_filepath)

    # Define partitioning columns names
    partitioning = [xbin_name, ybin_name]

    # Add partitioning columns
    df = assign_spatial_partitions(
        df=df,
        x_column="lon",
        y_column="lat",
        xbin_name=xbin_name,
        ybin_name=ybin_name,
        xbin_size=xbin_size,
        ybin_size=ybin_size,
    )

    # Write partitioned dataframe
    write_partitioned_dataset(
        df=df,
        base_dir=bucket_base_dir,
        filename_prefix=filename_prefix,
        partitioning=partitioning,
        row_group_size=row_group_size,
        **writer_kwargs,
    )


####--------------------------------------------------------------------------.
#### GPM Granules Routines
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


def split_list_in_blocks(values, block_size):
    return [values[i : i + block_size] for i in range(0, len(values), block_size)]


@print_task_elapsed_time(prefix="Granules Bucketing Operation Terminated.")
def write_granules_bucket(
    filepaths,
    # Bucket configuration
    bucket_base_dir,
    ds_to_df_converter,
    # Partitioning arguments
    xbin_size=15,
    ybin_size=15,
    xbin_name="lonbin",
    ybin_name="latbin",
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
    bucket_base_dir: str
        Base directory of the per-granule bucket archive.
    ds_to_df_converter : callable,
        Function taking a granule filepath, opening it and returning a pandas or dask dataframe.
    xbin_name : str, optional
        Name of the binned column used to partition the data along the x dimension.
        The default is "lonbin".
    ybin_name : str, optional
        Name of the binned column used to partition the data along the y dimension.
        The default is "latbin".
    xbin_size : int
        Longitude bin size. The default is 15.
    xbin_size : int
        Latitude bin size. The default is 15.
    parallel : bool
        Whether to bucket several granules in parallel.
        The default is ``True``.
    max_concurrent_tasks : None
        The maximum number of Dask tasks to be concurrently executed.
        If ``None``, let the Dask Scheduler to choose.
        The default is ``None``.
    max_dask_total_tasks : None
        The maximum number of Dask tasks to be scheduled.
        The default is 500.
    row_group_size : (int, str), optional
        Maximum number of rows in each written Parquet row group.
        If specified as a string (i.e. "500 MB"), the equivalent row group size
        number is estimated. The default is "500MB".
    **writer_kwargs: dict
        Optional arguments to be passed to the pyarrow Dataset Writer.
        Common arguments are 'format' and 'use_threads'.
        The default file 'format' is 'parquet'.
        The default 'use_threads' is 'True', which enable multithreaded file writing.
        More information available at:
            - https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html

    Notes
    -----
    - Partition by 1° degree pixels: 64800 directories (360*180)
    - Partition by 5° degree pixels: 2592 directories (72*36)
    - Partition by 10° degree pixels: 648 directories (36*18)
    - Partition by 15° degree pixels: 288 directories (24*12)

    """
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
                bucket_base_dir=bucket_base_dir,
                ds_to_df_converter=ds_to_df_converter,
                # Partitioning arguments
                xbin_size=xbin_size,
                ybin_size=ybin_size,
                xbin_name=xbin_name,
                ybin_name=ybin_name,
                # Writer kwargs
                row_group_size=row_group_size,
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


####--------------------------------------------------------------------------.
