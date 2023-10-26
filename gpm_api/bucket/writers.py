#!/usr/bin/env python3
"""
Created on Wed Aug  2 12:11:11 2023

@author: ghiggi
"""
import os

import dask

from gpm_api.bucket.dataset import write_partitioned_dataset
from gpm_api.bucket.processing import (
    assign_spatial_partitions,
    convert_ds_to_df,
    ds_to_dask_df_function,
    ds_to_pd_df_function,
    get_granule_dataframe,
)

# from gpm_api.io.info import get_key_from_filepath
from gpm_api.utils.dask import clean_memory, get_client
from gpm_api.utils.timing import print_task_elapsed_time

####--------------------------------------------------------------------------.
#### Single GPM Granule Routines


def write_granule_bucket(
    src_fpath,
    bucket_base_dir,
    open_granule_kwargs={},
    preprocessing_function=None,
    ds_to_df_function=ds_to_pd_df_function,
    filtering_function=None,
    precompute_granule=True,
    # Partitioning arguments
    xbin_size=15,
    ybin_size=15,
    xbin_name="lonbin",
    ybin_name="latbin",
    # Writer kwargs
    format="parquet",
    use_threads=True,
    **writer_kwargs,
):
    """Write a geographically partitioned Parquet Dataset.

    Bin size info:
    - Partition by 1° degree pixels: 64800 directories (360*180)
    - Partition by 5° degree pixels: 2592 directories (72*36)
    - Partition by 10° degree pixels: 648 directories (36*18)
    - Partition by 15° degree pixels: 288 directories (24*12)

    """
    # Retrieve dataframe
    df = get_granule_dataframe(
        src_fpath,
        open_granule_kwargs=open_granule_kwargs,
        preprocessing_function=preprocessing_function,
        ds_to_df_function=ds_to_df_function,
        filtering_function=filtering_function,
        precompute_granule=precompute_granule,
    )

    # Define partitioning columns names
    partitioning = [xbin_name, ybin_name]

    # Add partitioning columns
    df = assign_spatial_partitions(
        df=df,
        x_column="lat",
        y_column="lon",
        xbin_name=xbin_name,
        ybin_name=ybin_name,
        xbin_size=xbin_size,
        ybin_size=ybin_size,
    )

    # Define unique prefix name so to add files to the bucket archive
    # - This prevent risk of overwriting
    # - If df is pandas.dataframe -->  f"{fname_prefix}_" + "{i}.parquet"
    # - if df is a dask.dataframe -->  f"{fname_prefix}_dask.partition_{part_index}"
    fname_prefix = os.path.splitext(os.path.basename(src_fpath))[0]

    write_partitioned_dataset(
        df=df,
        base_dir=bucket_base_dir,
        fname_prefix=fname_prefix,
        partitioning=partitioning,
        format=format,
        use_threads=use_threads,
        **writer_kwargs,
    )


####--------------------------------------------------------------------------.
#### GPM Granules Routines
def _try_write_granule_bucket(
    src_fpath,
    # Bucket configuration
    bucket_base_dir,
    open_granule_kwargs,
    preprocessing_function,
    filtering_function,
    ds_to_df_function,
    # Partitioning arguments
    xbin_size=15,
    ybin_size=15,
    xbin_name="lonbin",
    ybin_name="latbin",
    # Writer kwargs
    format="parquet",
    use_threads=True,
    **writer_kwargs,
):
    try:
        # synchronous
        with dask.config.set(scheduler="single-threaded"):
            _ = write_granule_bucket(
                src_fpath=src_fpath,
                bucket_base_dir=bucket_base_dir,
                open_granule_kwargs=open_granule_kwargs,
                ds_to_df_function=ds_to_df_function,
                preprocessing_function=preprocessing_function,
                filtering_function=filtering_function,
                # Partitioning arguments
                xbin_size=xbin_size,
                ybin_size=ybin_size,
                xbin_name=xbin_name,
                ybin_name=ybin_name,
                # Writer kwargs
                format=format,
                use_threads=use_threads,
                **writer_kwargs,
            )
            # If works, return None
            info = None
    except Exception as e:
        # Define tuple to return
        info = src_fpath, str(e)
    return info


def split_list_in_blocks(values, block_size):
    list_blocks = [values[i : i + block_size] for i in range(0, len(values), block_size)]
    return list_blocks


@print_task_elapsed_time(prefix="Granules Bucketing Operation Terminated.")
def write_granules_bucket(
    fpaths,
    # Bucket configuration
    bucket_base_dir,
    open_granule_kwargs,
    preprocessing_function,
    filtering_function,
    ds_to_df_function=ds_to_pd_df_function,
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
    format="parquet",
    use_threads=True,
    **writer_kwargs,
):
    # TODO: force=True/False log which were processed so to skip reprocessing

    import dask

    from gpm_api.utils.parallel import compute_list_delayed

    # Split long list of files in blocks
    list_blocks = split_list_in_blocks(fpaths, block_size=max_dask_total_tasks)

    # Execute tasks by blocks to avoid dask overhead
    n_blocks = len(list_blocks)

    for i, block_fpaths in enumerate(list_blocks):
        print(f"Executing tasks block {i}/{n_blocks}")

        # Loop over granules
        if parallel:
            func = dask.delayed(_try_write_granule_bucket)
        else:
            func = _try_write_granule_bucket

        list_results = [
            func(
                src_fpath=src_fpath,
                bucket_base_dir=bucket_base_dir,
                open_granule_kwargs=open_granule_kwargs,
                ds_to_df_function=ds_to_df_function,
                preprocessing_function=preprocessing_function,
                filtering_function=filtering_function,
                # Partitioning arguments
                xbin_size=xbin_size,
                ybin_size=ybin_size,
                xbin_name=xbin_name,
                ybin_name=ybin_name,
                # Writer kwargs
                format=format,
                use_threads=use_threads,
                **writer_kwargs,
            )
            for src_fpath in block_fpaths
        ]

        # If delayed, execute the tasks
        if parallel:
            list_results = compute_list_delayed(
                list_results, max_concurrent_tasks=max_concurrent_tasks
            )

        # Process results to detect errors
        list_errors = [error_info for error_info in list_results if error_info is not None]
        for src_fpath, error_str in list_errors:
            print(f"An error occurred while processing {src_fpath}: {error_str}")

        # If parallel=True, retrieve client, clean the memory and restart
        if parallel:
            client = get_client()
            clean_memory(client)
            client.restart()

    return None


####--------------------------------------------------------------------------.
#### Single GPM Granule Routines OLD


# TODO: Currently used for final merging
# --> Adapt code to use write_partitioned_dataset
def write_parquet_dataset(
    df,
    parquet_fpath,
    partition_on,
    name_function=None,
    schema="infer",
    compression="snappy",
    write_index=False,
    custom_metadata=None,
    write_metadata_file=True,  # create _metadata file
    append=False,
    overwrite=False,
    ignore_divisions=False,
    compute=True,
    **writer_kwargs,
):
    # Note: Append currently works only when using fastparquet

    # Define default naming scheme
    if name_function is None:

        def name_function(i):
            return f"part.{i}.parquet"

    # Write Parquet Dataset
    df.to_parquet(
        parquet_fpath,
        engine="pyarrow",
        # Index option
        write_index=write_index,
        # Metadata
        custom_metadata=custom_metadata,
        write_metadata_file=write_metadata_file,  # enable writing the _metadata file
        # File structure
        name_function=name_function,
        partition_on=partition_on,
        # Encoding
        schema=schema,
        compression=compression,
        # Writing options
        append=append,
        overwrite=overwrite,
        ignore_divisions=ignore_divisions,
        compute=compute,
        **writer_kwargs,
    )


# def write_partitioned_parquet(
#     df, parquet_fpath, xbin_size, ybin_size, xbin_name, ybin_name, partition_size="100MB"
# ):
#     """Write a geographically partitioned Parquet Dataset.

#     Bin size info:
#     - Partition by 1° degree pixels: 64800 directories (360*180)
#     - Partition by 5° degree pixels: 2592 directories (72*36)
#     - Partition by 10° degree pixels: 648 directories (36*18)
#     - Partition by 15° degree pixels: 288 directories (24*12)
#     """
#     # Add spatial partitions columns to dataframe
#     partition_columns = {
#         xbin_name: get_bin_partition(df["lon"], bin_size=xbin_size),
#         ybin_name: get_bin_partition(df["lat"], bin_size=ybin_size),
#     }
#     df = df.assign(**partition_columns)

#     ## HERE IS DELICATE CODES ... NOT SURE IS THE OPTIMAL WAY YET !
#     # Reorder DaskDataframe by partitioning columns
#     # - The goal is to have each partitioning column(s) values in a single partition
#     df = df.sort_values([xbin_name, ybin_name], npartitions="auto")

#     # Define partition sizes
#     # - Control the number and size of parquet files in each disk partition
#     df = df.repartition(partition_size=partition_size) # maybe not needed

#     # Write Parquet Dataset
#     write_parquet_dataset(df=df, parquet_fpath=parquet_fpath, partition_on=[xbin_name, ybin_name])


# def write_granule_bucket(
#     src_fpath,
#     dst_fpath,
#     open_granule_kwargs={},
#     preprocessing_function=None,
#     ds_to_df_function=ds_to_df_function,
#     filtering_function=None,
#     xbin_size=15,
#     ybin_size=15,
#     xbin_name="lonbin",
#     ybin_name="latbin",
# ):
#     # Retrieve dataframe
#     df = get_granule_dataframe(
#         src_fpath,
#         open_granule_kwargs=open_granule_kwargs,
#         preprocessing_function=preprocessing_function,
#         ds_to_df_function=ds_to_df_function,
#         filtering_function=filtering_function,
#     )

#     # Write Parquet Dataset
#     write_partitioned_parquet(
#         df=df,
#         parquet_fpath=dst_fpath,
#         xbin_size=xbin_size,
#         ybin_size=ybin_size,
#         xbin_name=xbin_name,
#         ybin_name=ybin_name,
#     )

####--------------------------------------------------------------------------.
#### GPM Granules Routines OLD
# def _try_write_granule_bucket(
#     src_fpath,
#     dst_fpath,
#     open_granule_kwargs,
#     preprocessing_function,
#     filtering_function,
#     ds_to_df_function=ds_to_dask_df_function,
#     xbin_size=15,
#     ybin_size=15,
#     force=False,
# ):
#     try:
#         # synchronous
#         with dask.config.set(scheduler="single-threaded"):
#             _ = write_granule_bucket(
#                 src_fpath=src_fpath,
#                 dst_fpath=dst_fpath,
#                 open_granule_kwargs=open_granule_kwargs,
#                 preprocessing_function=preprocessing_function,
#                 filtering_function=filtering_function,
#                 xbin_size=xbin_size,
#                 ybin_size=ybin_size,
#             )
#             # If works, return None
#             info = None
#     except Exception as e:
#         # Define tuple to return
#         info = src_fpath, str(e)
#     return info


# def split_dict_in_blocks(dictionary, block_size):
#     dict_list = []
#     keys = list(dictionary.keys())
#     total_keys = len(keys)
#     for i in range(0, total_keys, block_size):
#         block_keys = keys[i : i + block_size]
#         block_dict = {key: dictionary[key] for key in block_keys}
#         dict_list.append(block_dict)
#     return dict_list


# def define_granule_bucket_fpath(bucket_base_dir, fpath):
#     """Define GPM Granule Parquet Dataset filepath."""
#     start_time = get_key_from_filepath(fpath, "start_time")
#     time_tree = get_time_tree(start_time)
#     time_dir = os.path.join(bucket_base_dir, time_tree)
#     os.makedirs(time_dir, exist_ok=True)
#     parquet_fpath = os.path.join(time_dir, os.path.basename(fpath) + ".parquet")
#     return parquet_fpath


# def write_granules_buckets(
#     fpaths,
#     bucket_base_dir,
#     open_granule_kwargs,
#     preprocessing_function,
#     filtering_function,
#     ds_to_df_function=ds_to_dask_df_function,
#     # Partitioning arguments
#     xbin_size=15,
#     ybin_size=15,
#     parallel=True,
#     force=False,
#     max_concurrent_tasks=None,
#     max_dask_total_tasks=500,
# ):
#     import dask

#     from gpm_api.utils.parallel import compute_list_delayed

#     src_dst_dict = {
#         src_fpath: define_granule_bucket_fpath(bucket_base_dir, src_fpath) for src_fpath in fpaths
#     }

#     if not force:
#         src_dst_dict = {src: dst for src, dst in src_dst_dict.items() if not os.path.exists(dst)}

#     if len(src_dst_dict) == 0:
#         return None

#     # Split long list of files in blocks
#     list_src_dst_dict = split_dict_in_blocks(src_dst_dict, block_size=max_dask_total_tasks)

#     # Execute tasks by blocks to avoid dask overhead
#     n_blocks = len(list_src_dst_dict)

#     for i, src_dst_dict in enumerate(list_src_dst_dict):
#         print(f"Executing tasks block {i}/{n_blocks}")

#         # Loop over granules
#         if parallel:
#             func = dask.delayed(_try_write_granule_bucket)
#         else:
#             func = _try_write_granule_bucket

#         list_results = [
#             func(
#                 src_fpath=src_fpath,
#                 dst_fpath=dst_fpath,
#                 open_granule_kwargs=open_granule_kwargs,
#                 preprocessing_function=preprocessing_function,
#                 ds_to_df_function=ds_to_df_function,
#                 filtering_function=filtering_function,
#                 xbin_size=xbin_size,
#                 ybin_size=ybin_size,
#             )
#             for src_fpath, dst_fpath in src_dst_dict.items()
#         ]
#         # If delayed, execute the tasks
#         if parallel:
#             list_results = compute_list_delayed(
#                 list_results, max_concurrent_tasks=max_concurrent_tasks
#             )

#         # Process results to detect errors
#         list_errors = [error_info for error_info in list_results if error_info is not None]
#         for src_fpath, error_str in list_errors:
#             print(f"An error occurred while processing {src_fpath}: {error_str}")

#         # If parallel=True, retrieve client, clean the memory and restart
#         if parallel:
#             client = get_client()
#             clean_memory(client)
#             client.restart()

#     return None


####--------------------------------------------------------------------------.
#### GPM Datasets Routines
@print_task_elapsed_time(prefix="Dataset Bucket Operation Terminated.")
def write_dataset_bucket(
    ds,
    bucket_fpath,
    open_granule_kwargs={},
    preprocessing_function=None,
    ds_to_df_function=ds_to_dask_df_function,
    filtering_function=None,
    # Partitioning arguments
    xbin_size=15,
    ybin_size=15,
    xbin_name="lonbin",
    ybin_name="latbin",
    # Writer arguments
    fname_prefix="part",
    format="parquet",
    use_threads=True,
    **writer_kwargs,
):
    df = convert_ds_to_df(
        ds=ds,
        preprocessing_function=preprocessing_function,
        ds_to_df_function=ds_to_df_function,
        filtering_function=filtering_function,
    )

    # Define partitioning columns names
    partitioning = [xbin_name, ybin_name]

    # Add partitioning columns
    df = assign_spatial_partitions(
        df=df,
        x_column="lat",
        y_column="lon",
        xbin_name=xbin_name,
        ybin_name=ybin_name,
        xbin_size=xbin_size,
        ybin_size=ybin_size,
    )

    # Write bucket
    write_partitioned_dataset(
        df=df,
        base_dir=bucket_fpath,
        partitioning=partitioning,
        format=format,
        use_threads=use_threads,
        **writer_kwargs,
    )

    return None
