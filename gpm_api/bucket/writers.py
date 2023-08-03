#!/usr/bin/env python3
"""
Created on Wed Aug  2 12:11:11 2023

@author: ghiggi
"""
import os

import dask
import numpy as np

from gpm_api.bucket.processing import (
    convert_ds_to_df,
    ds_to_df_function,
    get_granule_dataframe,
)
from gpm_api.io.directories import get_time_tree
from gpm_api.io.info import get_key_from_filepath


def get_bin_partition(values, bin_size):
    """
    Compute the bins partitioning values.

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


def write_parquet_dataset(df, parquet_fpath, partition_on, name_function=None):
    # Define default naming scheme
    if name_function is None:

        def name_function(i):
            return f"part.{i}.parquet"

    # Write Parquet Dataset
    df.to_parquet(
        parquet_fpath,
        engine="pyarrow",
        # Index option
        write_index=False,
        # Metadata
        custom_metadata=None,
        write_metadata_file=True,  # enable writing the _metadata file
        # File structure
        name_function=name_function,
        partition_on=partition_on,
        # Encoding
        schema="infer",
        compression="snappy",
        # Writing options
        append=False,
        overwrite=False,
        ignore_divisions=False,
        compute=True,
    )


def write_partitioned_parquet(
    df, parquet_fpath, xbin_size, ybin_size, xbin_name, ybin_name, partition_size="100MB"
):
    """Write a geographically partitioned Parquet Dataset.

    Bin size info:
    - Partition by 1° degree pixels: 64800 directories (360*180)
    - Partition by 5° degree pixels: 2592 directories (72*36)
    - Partition by 10° degree pixels: 648 directories (36*18)
    - Partition by 15° degree pixels: 288 directories (24*12)
    """
    # Add spatial partitions columns to dataframe
    partition_columns = {
        xbin_name: get_bin_partition(df["lon"], bin_size=xbin_size),
        ybin_name: get_bin_partition(df["lat"], bin_size=ybin_size),
    }
    df = df.assign(**partition_columns)

    # Reorder DaskDataframe by partitioning columns
    df = df.sort_values([xbin_name, ybin_name])

    # Define partition sizes
    # - Control the number and size of parquet files in each disk partition
    df = df.repartition(partition_size=partition_size)

    # Write Parquet Dataset
    write_parquet_dataset(df=df, parquet_fpath=parquet_fpath, partition_on=[xbin_name, ybin_name])


def define_granule_bucket_fpath(bucket_base_dir, fpath):
    """Define GPM Granule Parquet Dataset filepath."""
    start_time = get_key_from_filepath(fpath, "start_time")
    time_tree = get_time_tree(start_time)
    time_dir = os.path.join(bucket_base_dir, time_tree)
    os.makedirs(time_dir, exist_ok=True)
    parquet_fpath = os.path.join(time_dir, os.path.basename(fpath) + ".parquet")
    return parquet_fpath


def write_granule_bucket(
    src_fpath,
    dst_fpath,
    open_granule_kwargs={},
    preprocessing_function=None,
    ds_to_df_function=ds_to_df_function,
    filtering_function=None,
    xbin_size=15,
    ybin_size=15,
    xbin_name="lonbin",
    ybin_name="latbin",
):
    # Retrieve dataframe
    df = get_granule_dataframe(
        src_fpath,
        open_granule_kwargs=open_granule_kwargs,
        preprocessing_function=preprocessing_function,
        ds_to_df_function=ds_to_df_function,
        filtering_function=filtering_function,
    )

    # Write Parquet Dataset
    write_partitioned_parquet(
        df=df,
        parquet_fpath=dst_fpath,
        xbin_size=xbin_size,
        ybin_size=ybin_size,
        xbin_name=xbin_name,
        ybin_name=ybin_name,
    )


def _try_write_granule_bucket(
    src_fpath,
    dst_fpath,
    open_granule_kwargs,
    preprocessing_function,
    filtering_function,
    xbin_size=15,
    ybin_size=15,
    force=False,
):
    try:
        # synchronous
        with dask.config.set(scheduler="single-threaded"):
            _ = write_granule_bucket(
                src_fpath=src_fpath,
                dst_fpath=dst_fpath,
                open_granule_kwargs=open_granule_kwargs,
                preprocessing_function=preprocessing_function,
                filtering_function=filtering_function,
                xbin_size=xbin_size,
                ybin_size=ybin_size,
            )
            info = None
    except Exception as e:
        info = src_fpath, str(e)
    return info


def write_granules_buckets(
    fpaths,
    bucket_base_dir,
    open_granule_kwargs,
    preprocessing_function,
    filtering_function,
    xbin_size=15,
    ybin_size=15,
    parallel=True,
    force=False,
    max_concurrent_tasks=None,
):
    import dask

    from gpm_api.utils.parallel import compute_list_delayed

    src_dst_dict = {
        src_fpath: define_granule_bucket_fpath(bucket_base_dir, src_fpath) for src_fpath in fpaths
    }

    if not force:
        src_dst_dict = {src: dst for src, dst in src_dst_dict.items() if not os.path.exists(dst)}

    if len(src_dst_dict) == 0:
        return None

    if parallel:
        func = dask.delayed(_try_write_granule_bucket)
    else:
        func = _try_write_granule_bucket

    list_results = [
        func(
            src_fpath=src_fpath,
            dst_fpath=dst_fpath,
            open_granule_kwargs=open_granule_kwargs,
            preprocessing_function=preprocessing_function,
            filtering_function=filtering_function,
            xbin_size=xbin_size,
            ybin_size=ybin_size,
        )
        for src_fpath, dst_fpath in src_dst_dict.items()
    ]
    if parallel:
        list_results = compute_list_delayed(list_results, max_concurrent_tasks=max_concurrent_tasks)

    # Process results to detect errors
    list_errors = [error_info for error_info in list_results if error_info is not None]
    for src_fpath, error_str in list_errors:
        print(f"An error occurred while processing {src_fpath}: {error_str}")

    return None


def write_dataset_bucket(
    ds,
    bucket_fpath,
    open_granule_kwargs={},
    preprocessing_function=None,
    ds_to_df_function=ds_to_df_function,
    filtering_function=None,
    xbin_size=15,
    ybin_size=15,
    xbin_name="lonbin",
    ybin_name="latbin",
):

    df = convert_ds_to_df(
        ds=ds,
        preprocessing_function=preprocessing_function,
        ds_to_df_function=ds_to_df_function,
        filtering_function=filtering_function,
    )

    # Write Parquet Dataset
    write_partitioned_parquet(
        df=df,
        parquet_fpath=bucket_fpath,
        xbin_size=xbin_size,
        ybin_size=ybin_size,
        xbin_name=xbin_name,
        ybin_name=ybin_name,
    )
    return None
