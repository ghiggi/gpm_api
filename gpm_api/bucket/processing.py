#!/usr/bin/env python3
"""
Created on Wed Aug  2 12:14:51 2023

@author: ghiggi
"""
import math
import time

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

import gpm_api
from gpm_api.bucket.io import get_fpaths_by_bin
from gpm_api.dataset.granule import remove_unused_var_dims
from gpm_api.utils.timing import print_task_elapsed_time


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
    df = df.drop(columns=undesired_columns)
    return df


def ds_to_pd_df_function(ds):
    """Default function to convert an xr.Dataset to a pandas.Dataframe."""
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
    df = drop_undesired_columns(df)
    return df


def ds_to_dask_df_function(ds):
    """Default function to convert an xr.Dataset to a dask.Dataframe."""
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
    df = drop_undesired_columns(df)
    return df


def _check_is_callable_or_none(argument, argument_name):
    if not (callable(argument) or argument is None):
        raise TypeError(f"{argument_name} must be a function (or None).")


def convert_ds_to_df(
    ds, preprocessing_function, ds_to_df_function, filtering_function, precompute_granule=False
):
    # Check inputs
    _check_is_callable_or_none(preprocessing_function, argument_name="preprocessing_function")
    _check_is_callable_or_none(ds_to_df_function, argument_name="ds_to_df_function")
    _check_is_callable_or_none(filtering_function, argument_name="filtering_function")

    if precompute_granule:
        ds = ds.compute()

    # Preprocess xarray Dataset
    if callable(preprocessing_function):
        ds = preprocessing_function(ds)

    # Convert xarray Dataset to dask.Dataframe
    df = ds_to_df_function(ds)

    # Filter the dataset
    if callable(filtering_function):
        df = filtering_function(df)
    return df


def get_granule_dataframe(
    fpath,
    open_granule_kwargs={},
    preprocessing_function=None,
    ds_to_df_function=ds_to_dask_df_function,
    filtering_function=None,
    precompute_granule=False,
):
    # Open granule
    ds = gpm_api.open_granule(fpath, **open_granule_kwargs)

    # Convert to dataframe
    df = convert_ds_to_df(
        ds=ds,
        preprocessing_function=preprocessing_function,
        ds_to_df_function=ds_to_df_function,
        filtering_function=filtering_function,
        precompute_granule=precompute_granule,
    )

    return df


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


def assign_spatial_partitions(
    df, xbin_name, ybin_name, xbin_size, ybin_size, x_column="lat", y_column="lon"
):
    """Add partitioning bin columns to dataframe.

    Works for both dask.dataframe and pandas.dataframe.
    """
    # Add spatial partitions columns to dataframe
    partition_columns = {
        xbin_name: get_bin_partition(df[x_column], bin_size=xbin_size),
        ybin_name: get_bin_partition(df[y_column], bin_size=ybin_size),
    }
    df = df.assign(**partition_columns)
    return df


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
    else:
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
    row_group_size = math.floor(size_bytes / memory_per_row)
    return row_group_size


# def _get_bin_meta_template(filepath, bin_name):
#     from dask.dataframe.utils import make_meta

#     from gpm_api.bucket.readers import _read_parquet_bin_files

#     template_df = _read_parquet_bin_files([filepath], bin_name=bin_name)
#     meta = make_meta(template_df)
#     return meta


@print_task_elapsed_time(prefix="Bucket Merging Terminated.")
def merge_granule_buckets(
    bucket_base_dir,
    bucket_fpath,
    xbin_name="lonbin",
    ybin_name="latbin",
    row_group_size="500MB",
    compression="snappy",
    compression_level=None,
    **writer_kwargs,
):
    """
     Merge the per-granule bucket archive in a single  optimized archive !

     Parameters
     ----------
     bucket_base_dir : str
         Base directory of the per-granule bucket archive.
     bucket_fpath : str
         File path of the final bucket archive.
     xbin_name : str, optional
         Name of the binned column used to partition the data along the x dimension.
         The default is "lonbin".
     ybin_name : str, optional
         Name of the binned column used to partition the data along the y dimension.
         The default is "latbin".
     row_group_size : TYPE, optional
         Maximum number of rows in each written Parquet row group.
         If specified as a string (i.e. "500 MB"), the equivalent row group size
         number is estimated. The default is "500MB".
    compression : str, optional
         Specify the compression codec, either on a general basis or per-column.
         Valid values: {"none", "snappy", "gzip", "brotli", "lz4", "zstd"}.
         The default is "snappy".
     compression : int or dict, optional
         Specify the compression level for a codec, either on a general basis or per-column.
         If None is passed, arrow selects the compression level for the compression codec in use.
         The compression level has a different meaning for each codec, so you have
         to read the pyArrow documentation of the codec you are using.
         The default is compression_level=None.
     **writer_kwargs: dict
         Other writer options passed to dask.Dataframe.to_parquet, pyarrow.parquet.write_table
         and pyarrow.parquet.ParquetWriter.
         More information available at:
          - https://docs.dask.org/en/stable/generated/dask.dataframe.to_parquet.html
          - https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_table.html
          - https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html

     Returns
     -------
     None.

    """
    from dask.dataframe.utils import make_meta

    from gpm_api.bucket.readers import _read_parquet_bin_files
    from gpm_api.bucket.writers import write_parquet_dataset

    # TODO: remove need of xbin_name and ybin_name  !
    # --> Must be derived from source bucket !

    # Identify Parquet filepaths for each bin
    print("Searching of Parquet files has started.")
    t_i = time.time()
    bin_path_dict = get_fpaths_by_bin(bucket_base_dir)
    n_geographic_bins = len(bin_path_dict)
    t_f = time.time()
    t_elapsed = round((t_f - t_i) / 60, 1)
    print(f"Searching of Parquet files ended. Elapsed time: {t_elapsed} minutes.")
    print(f"{n_geographic_bins} geographic bins to process.")

    # Retrieve list of bins and associated filepaths
    list_bin_names = list(bin_path_dict.keys())
    list_bin_fpaths = list(bin_path_dict.values())

    # Define meta and row_group_size
    template_fpath = list_bin_fpaths[0][0]
    template_bin_name = list_bin_names[0]
    template_df_pd = _read_parquet_bin_files([template_fpath], bin_name=template_bin_name)
    meta = make_meta(template_df_pd)
    row_group_size = estimate_row_group_size(template_df_pd, size=row_group_size)

    # TODO: debug
    # list_bin_names = list_bin_names[0:10]
    # list_bin_fpaths = list_bin_fpaths[0:10]

    # Read dataframes for each geographic bin
    print("Lazy reading of dataframe has started")
    df = dd.from_map(_read_parquet_bin_files, list_bin_fpaths, list_bin_names, meta=meta)

    # Write Parquet Dataset
    print("Parquet Dataset writing has started")
    partitioning = [xbin_name, ybin_name]
    write_parquet_dataset(
        df=df,
        parquet_fpath=bucket_fpath,
        partition_on=partitioning,
        row_group_size=row_group_size,
        compression=compression,
        compression_level=compression_level,
        **writer_kwargs,
    )

    # TODO: Use write_partitioned_dataset instead !
    # --> https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html
    # --> https://arrow.apache.org/docs/python/generated/pyarrow.parquet.write_metadata.html
    # write_partitioned_dataset(
    #     df=df,
    #     base_dir,
    #     partitioning,
    #     fname_prefix="part",
    #     format="parquet",
    #     use_threads=True,
    #     **writer_kwargs,
    # )
    print("Parquet Dataset writing has completed")
