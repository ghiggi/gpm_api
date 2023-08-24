#!/usr/bin/env python3
"""
Created on Wed Aug  2 12:11:44 2023

@author: ghiggi
"""
import os

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _read_parquet_bin_files(filepaths, bin_name):
    # Read the list of Parquet files
    datasets = [pq.ParquetDataset(filepath, split_row_groups=False) for filepath in filepaths]
    # Concatenate the datasets
    table = pa.concat_tables([dataset.read() for dataset in datasets])
    # Add partitioning columns
    partition_key_value_list = bin_name.split(os.path.sep)
    for partition_str in partition_key_value_list:
        partition_column, value = partition_str.split("=")
        table = table.add_column(0, partition_column, pa.array([value] * table.num_rows))
    # Conversion to Pandas
    df = table.to_pandas(
        types_mapper=pd.ArrowDtype, zero_copy_only=False
    )  # TODO: set zero_copy_only=True one day
    return df


#### Unused and to move away ... maybe in dataset.py ...
def _get_arrow_to_pandas_defaults():
    arrow_to_pandas = {
        "zero_copy_only": False,  # Default is False. If True, raise error if doing copies
        "strings_to_categorical": False,
        "date_as_object": False,  # Default is True. If False convert to datetime64[ns]
        "timestamp_as_object": False,  # Default is True. If False convert to np.datetime64[ns]
        "use_threads": True,  #  parallelize the conversion using multiple threads.
        "safe": True,
        "split_blocks": False,
        "ignore_metadata": False,  # Default False. If False, use the ‘pandas’ metadata to get the Index
        "types_mapper": pd.ArrowDtype,  # Ensure pandas is created with Arrow dtype
    }
    return arrow_to_pandas


def read_partitioned_dataset(fpath, columns=None):
    arrow_to_pandas = _get_arrow_to_pandas_defaults()
    df = dd.read_parquet(
        fpath,
        engine="pyarrow",
        dtype_backend="pyarrow",
        index=False,
        # Filtering
        columns=columns,  # Specify columns to load
        filters=None,  # Row-filtering at read-time
        # Metadata options
        calculate_divisions=True,  # Calculate divisions from metadata
        ignore_metadata_file=False,  # True can slowdown a lot reading
        # Partitioning
        split_row_groups=False,  #   False --> Each file a partition
        # Arrow options
        arrow_to_pandas=arrow_to_pandas,
    )
    return df


#### Deprecated
# def read_bin_buckets_files(bin_fpaths, columns=None, partition_size=None, split_row_group=False):
#     arrow_to_pandas = _get_arrow_to_pandas_defaults()
#     df = dd.read_parquet(
#         bin_fpaths,
#         engine="pyarrow",
#         dtype_backend="pyarrow",
#         index=False,
#         infer_divisions=False,
#         # Filtering
#         columns=columns,  # Specify columns to load
#         filters=None,  # Row-filtering at read-time
#         # Metadata options
#         calculate_divisions=False,  # Calculate divisions from metadata
#         ignore_metadata_file=False,  # True can slowdown a lot reading
#         # Partitioning
#         split_row_groups=split_row_group,
#         # split_row_groups=False,  # False --> Each file a partition.
#         # Arrow options
#         arrow_to_pandas=arrow_to_pandas,
#     )

#     # Define partition sizes
#     if partition_size is not None:
#         df = df.repartition(partition_size=partition_size)

#     return df
