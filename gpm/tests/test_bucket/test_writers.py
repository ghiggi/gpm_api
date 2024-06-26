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
"""This module tests the Apache Arrow Partitioned Dataset Writers."""
import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gpm.bucket.readers import read_dask_partitioned_dataset
from gpm.bucket.writers import convert_size_to_bytes, estimate_row_group_size, write_partitioned_dataset
from gpm.tests.utils.fake_datasets import get_orbit_dataarray


def test_filesize_conversions():
    """Test conversion filesize strings to bytes."""
    # Test string conversion
    qa_pairs = [
        ("58 kb", 59392),
        ("117 kb", 119808),
        ("117kb", 119808),
        ("1 byte", 1),
        ("1 b", 1),
        ("117 bytes", 117),
        ("117  bytes", 117),
        ("  117 bytes  ", 117),
        ("117b", 117),
        ("117bytes", 117),
        ("1 kilobyte", 1024),
        ("117 kilobytes", 119808),
        ("0.7 mb", 734003),
        ("1mb", 1048576),
        ("5.2 mb", 5452595),
    ]
    for qa in qa_pairs:
        assert convert_size_to_bytes(qa[0]) == qa[1]

    # Test value conversion
    assert convert_size_to_bytes(1) == 1

    # Test bad input
    with pytest.raises(ValueError):
        convert_size_to_bytes("dummy")
    with pytest.raises(TypeError):
        convert_size_to_bytes(2.2)
    with pytest.raises(TypeError):
        convert_size_to_bytes(None)


def test_estimate_row_group_size():
    df_pd = pd.DataFrame(
        {
            "x": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
            "y": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
        },
    )
    df_dask = dd.from_pandas(df_pd, npartitions=1)
    df_pyarrow = pa.Table.from_pandas(df_pd, nthreads=None, preserve_index=False)
    df_pl = pl.DataFrame(df_pd)

    # df_pl.estimated_size()
    # df_pyarrow.nbytes
    # df_pd.memory_usage(index=False).sum()

    assert estimate_row_group_size(df_pd, size="1MB") == 65536  # strange pandas behaviour
    assert estimate_row_group_size(df_pyarrow, size="1MB") == 63764
    assert estimate_row_group_size(df_pl, size="1MB") == 63764
    with pytest.raises(TypeError):
        estimate_row_group_size(df_dask, size="1MB")


class TestWritePandasPartitionedDataset:

    def test_row_group_and_file_size(self, tmp_path):
        # Create pandas dataframe
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=10,
            end_lat=20,
            width=1e6,
            n_along_track=10,
            n_cross_track=5,
        )
        ds = da.to_dataset(name="dummy_var")
        df = ds.gpm.to_pandas_dataframe()

        # Check row_group_size and max_file_size has effect
        row_group_size = 3
        max_file_size = 10
        write_partitioned_dataset(
            df,
            base_dir=tmp_path,
            filename_prefix="prefix",
            partitions=None,
            row_group_size=row_group_size,
            max_file_size=max_file_size,
        )
        assert len(os.listdir(tmp_path)) == 5
        parquet_file = pq.ParquetFile(os.path.join(tmp_path, "prefix_0.parquet"))
        assert parquet_file.metadata.row_group(0).num_rows == row_group_size
        assert parquet_file.metadata.row_group(0).column(0).compression == "UNCOMPRESSED"
        assert parquet_file.metadata.num_rows == max_file_size

    def test_defaults(self, tmp_path):
        # Create pandas dataframe
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=10,
            end_lat=20,
            width=1e6,
            n_along_track=10,
            n_cross_track=5,
        )
        ds = da.to_dataset(name="dummy_var")
        df = ds.gpm.to_pandas_dataframe()

        # Check row_group_size and max_file_size defaults
        write_partitioned_dataset(
            df,
            base_dir=tmp_path,
            filename_prefix="prefix",
            partitions=None,
        )
        assert len(os.listdir(tmp_path)) == 1
        parquet_file = pq.ParquetFile(os.path.join(tmp_path, "prefix_0.parquet"))
        assert parquet_file.metadata.row_group(0).num_rows == df.shape[0]
        assert parquet_file.metadata.num_rows == df.shape[0]
        assert parquet_file.metadata.row_group(0).column(0).compression == "UNCOMPRESSED"
        assert parquet_file.metadata.row_group(0).column(0).statistics is None  # no statistics

    def test_custom_compression(self, tmp_path):
        # Create pandas dataframe
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=10,
            end_lat=20,
            width=1e6,
            n_along_track=10,
            n_cross_track=5,
        )
        ds = da.to_dataset(name="dummy_var")
        df = ds.gpm.to_pandas_dataframe()

        # Test custom compression
        write_partitioned_dataset(
            df,
            base_dir=tmp_path,
            filename_prefix="prefix",
            partitions=None,
            compression="lz4",
            compression_level=2,
        )
        parquet_file = pq.ParquetFile(os.path.join(tmp_path, "prefix_0.parquet"))
        assert parquet_file.metadata.row_group(0).column(0).compression == "LZ4"
        assert parquet_file.metadata.row_group(0).column(0).statistics is None

    def test_write_statistics(self, tmp_path):
        # Create pandas dataframe
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=10,
            end_lat=20,
            width=1e6,
            n_along_track=10,
            n_cross_track=5,
        )
        ds = da.to_dataset(name="dummy_var")
        df = ds.gpm.to_pandas_dataframe()

        # Write statistics
        write_partitioned_dataset(
            df,
            base_dir=tmp_path,
            filename_prefix="prefix",
            partitions=None,
            write_statistics=True,
        )
        parquet_file = pq.ParquetFile(os.path.join(tmp_path, "prefix_0.parquet"))
        assert parquet_file.metadata.row_group(0).column(0).statistics is not None

    def test_write_metadata_with_partitions(self, tmp_path):
        # Create dask dataframe
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=10,
            end_lat=20,
            width=1e6,
            n_along_track=10,
            n_cross_track=5,
        )
        ds = da.to_dataset(name="dummy_var")
        df = ds.gpm.to_pandas_dataframe()

        # Write
        write_partitioned_dataset(
            df,
            base_dir=tmp_path,
            filename_prefix="prefix",
            partitions="gpm_granule_id",
            write_metadata=True,
        )
        # Test files in the base directory
        expected_files = [
            "0",
            "_common_metadata",
            "_metadata",
        ]
        assert sorted(os.listdir(tmp_path)) == sorted(expected_files)

        # Assert can be read with dask using metadata
        df = read_dask_partitioned_dataset(base_dir=tmp_path)
        assert isinstance(df.compute(), pd.DataFrame)


class TestWriteDaskPartitionedDataset:

    def test_without_partitions(self, tmp_path):
        # Create dask dataframe
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=10,
            end_lat=20,
            width=1e6,
            n_along_track=10,
            n_cross_track=5,
        )
        ds = da.to_dataset(name="dummy_var")
        df = ds.gpm.to_dask_dataframe()
        df = df.repartition(npartitions=2)

        # Write
        write_partitioned_dataset(
            df,
            base_dir=tmp_path,
            filename_prefix="prefix",
            partitions=None,
        )
        # Assert structure
        assert sorted(os.listdir(tmp_path)) == sorted(
            ["prefix_dask_partition_0_0.parquet", "prefix_dask_partition_1_0.parquet"],
        )
        parquet_file = pq.ParquetFile(os.path.join(tmp_path, "prefix_dask_partition_1_0.parquet"))
        assert parquet_file.metadata.row_group(0).num_rows == 26
        assert parquet_file.metadata.num_rows == 26
        assert parquet_file.metadata.row_group(0).column(0).compression == "UNCOMPRESSED"
        assert parquet_file.metadata.row_group(0).column(0).statistics is None  # no statistics

        # Assert can be read with dask using metadata
        df = read_dask_partitioned_dataset(base_dir=tmp_path)
        assert isinstance(df.compute(), pd.DataFrame)

    def test_without_partitions_with_metadata(self, tmp_path):
        # Create dask dataframe
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=10,
            end_lat=20,
            width=1e6,
            n_along_track=10,
            n_cross_track=5,
        )
        ds = da.to_dataset(name="dummy_var")
        df = ds.gpm.to_dask_dataframe()
        df = df.repartition(npartitions=2)

        # Write
        write_partitioned_dataset(
            df,
            base_dir=tmp_path,
            filename_prefix="prefix",
            partitions=None,
            write_metadata=True,
        )
        # Assert structure
        expected_files = [
            "prefix_dask_partition_0_0.parquet",
            "prefix_dask_partition_1_0.parquet",
            "_common_metadata",
            "_metadata",
        ]
        assert sorted(os.listdir(tmp_path)) == sorted(expected_files)
        parquet_file = pq.ParquetFile(os.path.join(tmp_path, "prefix_dask_partition_1_0.parquet"))
        assert parquet_file.metadata.row_group(0).num_rows == 26
        assert parquet_file.metadata.num_rows == 26
        assert parquet_file.metadata.row_group(0).column(0).compression == "UNCOMPRESSED"
        assert parquet_file.metadata.row_group(0).column(0).statistics is None  # no statistics

        # Assert can be read with dask using metadata
        df = read_dask_partitioned_dataset(base_dir=tmp_path)
        assert isinstance(df.compute(), pd.DataFrame)

    def test_with_partitions(self, tmp_path):
        # Create dask dataframe
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=10,
            end_lat=20,
            width=1e6,
            n_along_track=10,
            n_cross_track=5,
        )
        ds = da.to_dataset(name="dummy_var")
        df = ds.gpm.to_dask_dataframe()
        df = df.repartition(npartitions=2)
        # Write
        write_partitioned_dataset(
            df,
            base_dir=tmp_path,
            filename_prefix="prefix",
            partitions="gpm_granule_id",
            write_metadata=True,
            row_group_size="100MB",  # enforce computation of first dask partition
        )
        # Assert generated files
        assert os.listdir(os.path.join(tmp_path, "0")) == sorted(
            [
                "prefix_dask_partition_0_0.parquet",
                "prefix_dask_partition_1_0.parquet",
            ],
        )
        # Assert can be read with dask using metadata
        df = read_dask_partitioned_dataset(base_dir=tmp_path)
        assert isinstance(df.compute(), pd.DataFrame)


def test_write_polars_partitioned_dataset(tmp_path):
    # Create pandas dataframe
    da = get_orbit_dataarray(
        start_lon=0,
        start_lat=0,
        end_lon=10,
        end_lat=20,
        width=1e6,
        n_along_track=10,
        n_cross_track=5,
    )
    ds = da.to_dataset(name="dummy_var")
    df = ds.gpm.to_pandas_dataframe()
    df_pl = pl.DataFrame(df)
    # Check row_group_size and max_file_size has effect
    row_group_size = 3
    max_file_size = 10
    write_partitioned_dataset(
        df_pl,
        base_dir=tmp_path,
        filename_prefix="prefix",
        partitions=None,
        row_group_size=row_group_size,
        max_file_size=max_file_size,
    )
    assert len(os.listdir(tmp_path)) == 5
    parquet_file = pq.ParquetFile(os.path.join(tmp_path, "prefix_0.parquet"))
    assert parquet_file.metadata.row_group(0).num_rows == row_group_size
    assert parquet_file.metadata.row_group(0).column(0).compression == "UNCOMPRESSED"
    assert parquet_file.metadata.num_rows == max_file_size


def test_write_empty_dataset(tmp_path):
    # Test pandas
    df_pandas_empty = pd.DataFrame()
    assert write_partitioned_dataset(df=df_pandas_empty, base_dir=tmp_path, write_metadata=True) is None
    assert os.listdir(tmp_path) == []
    # Test polars
    assert write_partitioned_dataset(df=pl.DataFrame(), base_dir=tmp_path, write_metadata=True) is None
    assert os.listdir(tmp_path) == []
    # Test dask
    df_dask_empty = dd.from_pandas(df_pandas_empty, npartitions=1)
    assert write_partitioned_dataset(df=df_dask_empty, base_dir=tmp_path, write_metadata=True) is None
    assert os.listdir(tmp_path) == []
    # Test pyarrow
    df_arrow_empty = pa.Table.from_pandas(df_pandas_empty, nthreads=None, preserve_index=False)
    assert write_partitioned_dataset(df=df_arrow_empty, base_dir=tmp_path, write_metadata=True) is None
    assert os.listdir(tmp_path) == []

    # Test invalid type
    with pytest.raises(TypeError):
        write_partitioned_dataset("dummy", base_dir=tmp_path, partitions=None)
