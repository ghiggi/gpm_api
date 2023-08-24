#!/usr/bin/env python3
"""
Created on Fri Aug 11 13:27:35 2023

@author: ghiggi
"""
import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.dataset


#### Dataset Writers
def _write_pd_partitioned_dataset(df, base_dir, fname_prefix, partitioning, **writer_kwargs):
    # Sanitize writer_kwargs
    _ = writer_kwargs.pop("create_dir", None)
    _ = writer_kwargs.pop("existing_data_behavior", None)
    _ = writer_kwargs.pop("partitioning_flavor", None)

    # Define basename template
    basename_template = f"{fname_prefix}_" + "{i}.parquet"

    # Conversion to pyarrow table
    table = pa.Table.from_pandas(df, nthreads=None, preserve_index=False)

    # Write files
    # -  https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html
    pyarrow.dataset.write_dataset(
        table,
        base_dir=base_dir,
        basename_template=basename_template,
        partitioning=partitioning,
        partitioning_flavor="hive",
        create_dir=True,
        existing_data_behavior="overwrite_or_ignore",
        **writer_kwargs,
    )


def _write_dask_partition(part, part_index, base_dir, fname_prefix, partitioning, **writer_kwargs):
    # Convert to pandas
    part = part.compute()
    # Define actual fname_prefix
    part_fname_prefix = f"{fname_prefix}_dask.partition_{part_index}"
    # Write dask partition into various directories
    _write_pd_partitioned_dataset(
        part,
        base_dir=base_dir,
        fname_prefix=part_fname_prefix,
        partitioning=partitioning,
        **writer_kwargs,
    )


def _write_dask_partitioned_dataset(df, base_dir, fname_prefix, partitioning, **writer_kwargs):
    # TODO: map_partitions could be used to write each partition in parallel
    # TODO: or pass list partitions to pyarrow directly
    for part_index, part in enumerate(df.partitions):
        _write_dask_partition(
            part=part,
            part_index=part_index,
            base_dir=base_dir,
            fname_prefix=fname_prefix,
            partitioning=partitioning,
            **writer_kwargs,
        )


def write_partitioned_dataset(
    df,
    base_dir,
    partitioning,
    fname_prefix="part",
    format="parquet",
    use_threads=True,
    **writer_kwargs,
):
    writer_kwargs["format"] = format
    writer_kwargs["use_threads"] = use_threads
    if isinstance(df, dd.DataFrame):
        _write_dask_partitioned_dataset(
            df=df,
            base_dir=base_dir,
            fname_prefix=fname_prefix,
            partitioning=partitioning,
            **writer_kwargs,
        )
    elif isinstance(df, pd.DataFrame):
        _write_pd_partitioned_dataset(
            df=df,
            base_dir=base_dir,
            fname_prefix=fname_prefix,
            partitioning=partitioning,
            **writer_kwargs,
        )
    else:
        raise TypeError("Expecting a pands.DataFrame or dask.DataFrame")
