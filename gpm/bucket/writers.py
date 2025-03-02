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
import math
import os

import dask.dataframe as dd
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.dataset
import pyarrow.parquet as pq


def _convert_size_to_bytes(size_str):
    """Convert human filesizes to bytes.

    Special cases:
     - singular units, e.g., "1 byte"
     - byte vs b
     - yottabytes, zetabytes, etc.
     - with & without spaces between & around units.
     - floats ("5.2 mb")

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

    for suffix, multiplier in multipliers.items():
        size_str = size_str.lower().strip().strip("s")
        if size_str.lower().endswith(suffix):
            return int(float(size_str[0 : -len(suffix)]) * multiplier)
    if size_str.endswith("b"):
        size_str = size_str[0:-1]
    elif size_str.endswith("byte"):
        size_str = size_str[0:-4]
    return int(size_str)


def convert_size_to_bytes(size):
    if not isinstance(size, (str, int)):
        raise TypeError("Expecting a string (i.e. 200MB) or the integer number of bytes.")
    if isinstance(size, int):
        return size
    try:
        size = _convert_size_to_bytes(size)
    except Exception:
        raise ValueError(f"Impossible to parse '{size}' to the number of bytes.")
    return size


def estimate_row_group_size(df, size="200MB"):
    """Estimate ``row_group_size`` parameter based on the desired row group memory size.

    ``row_group_size`` is a Parquet argument controlling the number of rows
    in each Apache Parquet File Row Group.
    """
    if isinstance(df, pa.Table):
        memory_used = df.nbytes
    elif isinstance(df, pd.DataFrame):
        memory_used = df.memory_usage(index=False).sum()
    elif isinstance(df, pl.DataFrame):
        memory_used = df.estimated_size()
    else:
        raise TypeError("Expecting a pandas, polars or pyarrow DataFrame.")
    size_bytes = convert_size_to_bytes(size)
    n_rows = len(df)
    memory_per_row = memory_used / n_rows
    return math.floor(size_bytes / memory_per_row)


####--------------------------------------------------------------------------------------------------------
#### Dataset Writers


def get_table_schema_without_partitions(table, partitions=None):
    if partitions is None:
        return table.schema
    return table.drop_columns(columns=partitions).schema


def get_table_from_dask_dataframe_partition(df):
    df_pd = df.get_partition(0).compute()
    table = pa.Table.from_pandas(df_pd, nthreads=None, preserve_index=False)
    return table


def write_dataset_metadata(base_dir, metadata_collector, schema):
    # Write the metadata
    print("Writing the metadata")
    # Write the ``_common_metadata`` parquet file without row groups statistics
    pq.write_metadata(schema=schema, where=os.path.join(base_dir, "_common_metadata"))

    # Write the ``_metadata`` parquet file with row groups statistics of all files
    pq.write_metadata(
        schema=schema,
        where=os.path.join(base_dir, "_metadata"),
        metadata_collector=metadata_collector,
    )


def preprocess_writer_kwargs(writer_kwargs, df):
    # https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html

    # Set default format to Parquet
    if "format" not in writer_kwargs:
        writer_kwargs["format"] = "parquet"
    # Set default multithreaded parquet writing
    if "use_threads" not in writer_kwargs:
        writer_kwargs["use_threads"] = True
    if writer_kwargs.get("partitioning_flavor", "") == "directory":
        writer_kwargs["partitioning_flavor"] = None
    # Sanitize writer_kwargs
    _ = writer_kwargs.pop("create_dir", None)
    _ = writer_kwargs.pop("existing_data_behavior", None)

    # Get writer kwargs (with pyarrow defaults)
    max_file_size = writer_kwargs.pop("max_file_size", None)
    row_group_size = writer_kwargs.pop("row_group_size", None)

    max_rows_per_file = writer_kwargs.get("max_rows_per_file", None)
    min_rows_per_group = writer_kwargs.get("min_rows_per_group", None)  # 0
    max_rows_per_group = writer_kwargs.get("max_rows_per_group", None)  # 1024 * 1024

    # Define row_group_size and max_file_size
    # - If string, estimate the number of corresponding rows
    # - If integer, assumes it is the wished number of rows
    # --> Here we estimate the number of rows
    # --> If input is dask dataframe, compute first partition to estimate row numbers
    if (isinstance(row_group_size, str) or isinstance(max_file_size, str)) and isinstance(df, dd.DataFrame):
        df = get_table_from_dask_dataframe_partition(df)
    if isinstance(row_group_size, str):  # "200 MB"
        row_group_size = estimate_row_group_size(df=df, size=row_group_size)
    if isinstance(max_file_size, str):
        max_file_size = estimate_row_group_size(df=df, size=max_file_size)

    # If row_group_size is not None --> Override min_rows_per_group, max_rows_per_group
    if row_group_size is not None:
        max_rows_per_group = row_group_size
        min_rows_per_group = row_group_size

    # If max_file_size is not None --> Override max_rows_per_file
    if max_file_size is not None:
        max_rows_per_file = max_file_size

    # Define file options if None
    file_options = writer_kwargs.get("file_options", None)
    if file_options is None:
        compression = writer_kwargs.pop("compression", None)
        compression_level = writer_kwargs.pop("compression_level", None)
        write_statistics = writer_kwargs.pop("write_statistics", False)
        file_options = {}
        file_options["compression"] = compression
        file_options["compression_level"] = compression_level
        file_options["write_statistics"] = write_statistics
        parquet_format = pa.dataset.ParquetFileFormat()
        file_options = parquet_format.make_write_options(**file_options)

    # Define row_group_size
    writer_kwargs["min_rows_per_group"] = min_rows_per_group
    writer_kwargs["max_rows_per_group"] = max_rows_per_group
    writer_kwargs["max_rows_per_file"] = max_rows_per_file
    writer_kwargs["file_options"] = file_options

    # Define metadata objects
    write_metadata = writer_kwargs.pop("write_metadata", False)
    metadata_collector = []
    if write_metadata:
        # Define file visitor for metadata collection
        def file_visitor(written_file):
            metadata_collector.append(written_file.metadata)

        writer_kwargs["file_visitor"] = file_visitor
    return writer_kwargs, metadata_collector


def write_arrow_partitioned_dataset(table, base_dir, filename_prefix, partitions, **writer_kwargs):
    # Do not write if empty dataframe
    if table.num_rows == 0:
        return None

    # Preprocess writer kwargs
    writer_kwargs, metadata_collector = preprocess_writer_kwargs(writer_kwargs=writer_kwargs, df=table)

    # Define basename template
    basename_template = f"{filename_prefix}_" + "{i}.parquet"

    # Write files
    # -  https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html
    pyarrow.dataset.write_dataset(
        table,
        base_dir=base_dir,
        basename_template=basename_template,
        partitioning=partitions,
        create_dir=True,
        existing_data_behavior="overwrite_or_ignore",
        **writer_kwargs,
    )

    # Define schema
    schema = get_table_schema_without_partitions(table, partitions)

    # Define metadata
    if metadata_collector:
        write_dataset_metadata(base_dir=base_dir, metadata_collector=metadata_collector, schema=schema)
    return schema


def write_pandas_partitioned_dataset(df, base_dir, filename_prefix, partitions, **writer_kwargs):
    # Do not write if empty dataframe
    if df.size == 0:
        return None

    table = pa.Table.from_pandas(df, nthreads=None, preserve_index=False)
    schema = write_arrow_partitioned_dataset(
        table=table,
        base_dir=base_dir,
        filename_prefix=filename_prefix,
        partitions=partitions,
        **writer_kwargs,
    )
    return schema


def write_polars_partitioned_dataset(df, base_dir, filename_prefix, partitions, **writer_kwargs):
    # Do not write if empty dataframe
    if df.shape[0] == 0:
        return None
    schema = write_arrow_partitioned_dataset(
        table=df.to_arrow(),
        base_dir=base_dir,
        filename_prefix=filename_prefix,
        partitions=partitions,
        **writer_kwargs,
    )
    return schema


def write_dask_partitioned_dataset(df, base_dir, filename_prefix, partitions, **writer_kwargs):
    """Write a Dask DataFrame to a partitioned dataset.

    It loops over the dataframe partitions and write them to disk.
    If ``row_group_size`` or ``max_file_size`` are specified as string, it loads the first dataframe partition
    to estimate the row numbers.
    """
    writer_kwargs, metadata_collector = preprocess_writer_kwargs(writer_kwargs=writer_kwargs, df=df)
    # get_table_schema_without_partitions
    for partition_index, df_partition in enumerate(df.partitions):
        schema = _write_dask_partition(
            df_partition=df_partition,
            partition_index=partition_index,
            base_dir=base_dir,
            filename_prefix=filename_prefix,
            partitions=partitions,
            **writer_kwargs,
        )
    if metadata_collector:
        write_dataset_metadata(base_dir=base_dir, metadata_collector=metadata_collector, schema=schema)


def _write_dask_partition(
    df_partition,
    partition_index,
    base_dir,
    filename_prefix,
    partitions,
    **writer_kwargs,
):
    # Convert to pandas
    df_partition = df_partition.compute()
    # Define actual filename_prefix
    part_filename_prefix = f"{filename_prefix}_dask_partition_{partition_index}"
    # Write dask partition into various directories
    table_schema = write_pandas_partitioned_dataset(
        df=df_partition,
        base_dir=base_dir,
        filename_prefix=part_filename_prefix,
        partitions=partitions,
        **writer_kwargs,
    )
    return table_schema


def write_partitioned_dataset(
    df,
    base_dir,
    partitions=None,
    filename_prefix="part",
    **writer_kwargs,
):
    if isinstance(partitions, str):
        partitions = [partitions]
    if isinstance(df, dd.DataFrame):
        write_dask_partitioned_dataset(
            df=df,
            base_dir=base_dir,
            filename_prefix=filename_prefix,
            partitions=partitions,
            **writer_kwargs,
        )
    elif isinstance(df, pd.DataFrame):
        _ = write_pandas_partitioned_dataset(
            df=df,
            base_dir=base_dir,
            filename_prefix=filename_prefix,
            partitions=partitions,
            **writer_kwargs,
        )
    elif isinstance(df, pl.DataFrame):
        _ = write_polars_partitioned_dataset(
            df=df,
            base_dir=base_dir,
            filename_prefix=filename_prefix,
            partitions=partitions,
            **writer_kwargs,
        )
    elif isinstance(df, pa.Table):
        _ = write_arrow_partitioned_dataset(
            table=df,
            base_dir=base_dir,
            filename_prefix=filename_prefix,
            partitions=partitions,
            **writer_kwargs,
        )
    else:
        raise TypeError("Expecting a pandas, dask, polars or pyarrow DataFrame.")
