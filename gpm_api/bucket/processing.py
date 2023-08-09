#!/usr/bin/env python3
"""
Created on Wed Aug  2 12:14:51 2023

@author: ghiggi
"""
import dask
import dask.dataframe as dd
import xarray as xr

import gpm_api
from gpm_api.bucket.io import get_parquet_fpaths, group_fpaths_by_bin
from gpm_api.dataset.granule import remove_unused_var_dims


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


def ds_to_df_function(ds):
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


def convert_ds_to_df(ds, preprocessing_function, ds_to_df_function, filtering_function):
    # Check inputs
    _check_is_callable_or_none(preprocessing_function, argument_name="preprocessing_function")
    _check_is_callable_or_none(ds_to_df_function, argument_name="ds_to_df_function")
    _check_is_callable_or_none(filtering_function, argument_name="filtering_function")

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
    ds_to_df_function=ds_to_df_function,
    filtering_function=None,
):
    # Open granule
    ds = gpm_api.open_granule(fpath, **open_granule_kwargs)

    df = convert_ds_to_df(
        ds=ds,
        preprocessing_function=preprocessing_function,
        ds_to_df_function=ds_to_df_function,
        filtering_function=filtering_function,
    )

    return df


def _get_bin_meta_template(filepath, bin_name): 
    from dask.dataframe.utils import make_meta
    from gpm_api.bucket.readers import _read_parquet_bin_files
    template_df = _read_parquet_bin_files([filepath], bin_name=bin_name)
    meta = make_meta(template_df)
    return meta 


def merge_granule_buckets(
    bucket_base_dir,
    bucket_fpath,
    xbin_name="lonbin",
    ybin_name="latbin",
):
    from gpm_api.bucket.readers import _read_parquet_bin_files
    from gpm_api.bucket.writers import write_parquet_dataset

    # Identify all Parquet filepaths
    fpaths = get_parquet_fpaths(bucket_base_dir)
    n_fpaths = len(fpaths)
    print(f"{n_fpaths} Parquet files to merge.")

    # Group filepaths by geographic bin
    bin_path_dict = group_fpaths_by_bin(fpaths)
    n_geographic_bins = len(bin_path_dict)
    print(f"{n_geographic_bins} geographic bins to process.")

    # Retrieve list of bins and associated filepaths 
    list_bin_name = list(bin_path_dict.keys())
    list_bin_fpaths = list(bin_path_dict.values())
   
    # Read dataframes for each geographic bin
    print("Lazy reading of dataframe has started")
    meta = _get_bin_meta_template(list_bin_fpaths[0][0], bin_name=list_bin_name[0])
    df = dd.from_map(_read_parquet_bin_files, list_bin_fpaths, list_bin_name, meta=meta)
    
    # Write Parquet Dataset
    # --> TODO add row_group_size
    print("Parquet Dataset writing has started")
    xbin_name = "lonbin"
    ybin_name = "latbin"
    write_parquet_dataset(df=df,
                          parquet_fpath=bucket_fpath, 
                          partition_on=[xbin_name, ybin_name])
    
    print("Parquet Dataset writing has completed")
