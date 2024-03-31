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
"""This module provide to write a GPM Geographic Bucket Apache Parquet Dataset."""
import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.dataset

from gpm.bucket.processing import estimate_row_group_size


#### Dataset Writers
def _preprocess_writer_kwargs(writer_kwargs, df):
    # Set default format to Parquet
    if "format" not in writer_kwargs:
        writer_kwargs["format"] = "parquet"
    # Set default multithreaded parquet writing
    if "use_threads" not in writer_kwargs:
        writer_kwargs["use_threads"] = True

    # Sanitize writer_kwargs
    _ = writer_kwargs.pop("create_dir", None)
    _ = writer_kwargs.pop("existing_data_behavior", None)
    _ = writer_kwargs.pop("partitioning_flavor", None)

    # Define row_group_size
    row_group_size = writer_kwargs.pop("row_group_size", None)
    min_rows_per_group = writer_kwargs.get("min_rows_per_group", None)
    max_rows_per_group = writer_kwargs.get("max_rows_per_group", 1024 * 1024)  # pyarrow default
    if "min_rows_per_group" not in writer_kwargs and row_group_size is not None:
        if isinstance(row_group_size, str):
            min_rows_per_group = estimate_row_group_size(df, size=row_group_size)
        else:
            min_rows_per_group = row_group_size
        writer_kwargs["min_rows_per_group"] = min_rows_per_group
        if min_rows_per_group > max_rows_per_group:
            writer_kwargs["max_rows_per_group"] = min_rows_per_group
    return writer_kwargs


def _write_pd_partitioned_dataset(df, base_dir, filename_prefix, partitioning, **writer_kwargs):
    # Preprocess writer kwargs
    writer_kwargs = _preprocess_writer_kwargs(writer_kwargs=writer_kwargs, df=df)

    # Define basename template
    basename_template = f"{filename_prefix}_" + "{i}.parquet"

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


def _write_dask_partition(
    part,
    part_index,
    base_dir,
    filename_prefix,
    partitioning,
    **writer_kwargs,
):
    # Convert to pandas
    part = part.compute()
    # Define actual filename_prefix
    part_filename_prefix = f"{filename_prefix}_dask.partition_{part_index}"
    # Write dask partition into various directories
    _write_pd_partitioned_dataset(
        part,
        base_dir=base_dir,
        filename_prefix=part_filename_prefix,
        partitioning=partitioning,
        **writer_kwargs,
    )


def _write_dask_partitioned_dataset(df, base_dir, filename_prefix, partitioning, **writer_kwargs):
    # TODO: map_partitions could be used to write each partition in parallel
    # TODO: or pass list partitions to pyarrow directly
    for part_index, part in enumerate(df.partitions):
        _write_dask_partition(
            part=part,
            part_index=part_index,
            base_dir=base_dir,
            filename_prefix=filename_prefix,
            partitioning=partitioning,
            **writer_kwargs,
        )


def write_partitioned_dataset(
    df,
    base_dir,
    partitioning,
    filename_prefix="part",
    **writer_kwargs,
):
    # Do not write if empty dataframe
    if df.size == 0:
        return

    if isinstance(df, dd.DataFrame):
        _write_dask_partitioned_dataset(
            df=df,
            base_dir=base_dir,
            filename_prefix=filename_prefix,
            partitioning=partitioning,
            **writer_kwargs,
        )
    elif isinstance(df, pd.DataFrame):
        _write_pd_partitioned_dataset(
            df=df,
            base_dir=base_dir,
            filename_prefix=filename_prefix,
            partitioning=partitioning,
            **writer_kwargs,
        )
    else:
        raise TypeError("Expecting a pands.DataFrame or dask.DataFrame")
