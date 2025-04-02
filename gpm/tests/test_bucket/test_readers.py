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
"""This module tests the bucket readers."""

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from gpm.bucket import LonLatPartitioning
from gpm.bucket.readers import read_bucket
from gpm.bucket.routines import write_granules_bucket
from gpm.tests.utils.fake_datasets import get_orbit_dataarray


def create_granule_dataframe():
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
    ds = ds.drop_vars("spatial_ref")
    df = ds.gpm.to_pandas_dataframe()
    return df


def granule_to_df_toy_func(filepath):
    return create_granule_dataframe()


def create_bucket_archive(bucket_dir):
    # Define filepaths
    filepaths = [
        "2A.GPM.DPR.V9-20211125.20210705-S013942-E031214.041760.V07A.HDF5",
        "2A.GPM.DPR.V9-20211125.20210805-S013942-E031214.041760.V07B.HDF5",
        "2A.GPM.DPR.V9-20211125.20230705-S013942-E031214.041760.V07A.HDF5",
    ]
    # Define partitioning
    spatial_partitioning = LonLatPartitioning(size=(10, 10))

    # Run processing
    write_granules_bucket(
        # Bucket Input/Output configuration
        filepaths=filepaths,
        bucket_dir=bucket_dir,
        spatial_partitioning=spatial_partitioning,
        granule_to_df_func=granule_to_df_toy_func,
        # Processing options
        parallel=False,
    )


NUM_COLUMNS = 8

# import pathlib
# tmp_path = pathlib.Path("/tmp/bucket14")


class TestReadBucket:

    def test_read_full_bucket(self, tmp_path):
        # Define bucket dir
        bucket_dir = tmp_path
        create_bucket_archive(bucket_dir)

        # Test read full database
        df_pl = read_bucket(bucket_dir)
        assert isinstance(df_pl, pl.DataFrame)
        assert df_pl.shape == (150, NUM_COLUMNS)

    def test_rows_columns_subsets(self, tmp_path):
        # Define bucket dir
        bucket_dir = tmp_path
        create_bucket_archive(bucket_dir)

        # Test read row subset
        df_pl = read_bucket(bucket_dir, n_rows=2)
        assert isinstance(df_pl, pl.DataFrame)
        assert df_pl.shape == (2, NUM_COLUMNS)

        # Test read row, columns subset
        df_pl = read_bucket(bucket_dir, n_rows=3, columns=["lon", "lat"])
        assert df_pl.shape == (3, 2)
        assert "lon" in df_pl
        assert "lat" in df_pl

    def test_files_filters(self, tmp_path):
        # Define bucket dir
        bucket_dir = tmp_path
        create_bucket_archive(bucket_dir)

        # Test filtering by file_extension
        df_pl = read_bucket(bucket_dir, file_extension=".parquet")
        assert df_pl.shape == (150, NUM_COLUMNS)

        df_pl = read_bucket(bucket_dir, glob_pattern="*V07B*")
        assert df_pl.shape == (50, NUM_COLUMNS)

        # Test raise error if no files (after i.e. filtering criteria)
        with pytest.raises(ValueError):
            df_pl = read_bucket(bucket_dir, file_extension=".csv")
        with pytest.raises(ValueError):
            df_pl = read_bucket(bucket_dir, glob_pattern="dummy")

    def test_backends(self, tmp_path):
        # Define bucket dir
        bucket_dir = tmp_path
        create_bucket_archive(bucket_dir)

        # Test backends
        assert isinstance(read_bucket(bucket_dir), pl.DataFrame)
        assert isinstance(read_bucket(bucket_dir, backend="polars"), pl.DataFrame)
        assert isinstance(read_bucket(bucket_dir, backend="polars_lazy"), pl.LazyFrame)
        assert isinstance(read_bucket(bucket_dir, backend="pandas"), pd.DataFrame)
        assert isinstance(read_bucket(bucket_dir, backend="pyarrow"), pa.Table)

        with pytest.raises(ValueError):
            read_bucket(bucket_dir, backend="dask")
        with pytest.raises(ValueError):
            read_bucket(bucket_dir, backend="whatever_other")

    def test_invalid_arguments(self, tmp_path):
        # Define bucket dir
        bucket_dir = tmp_path
        create_bucket_archive(bucket_dir)
        # Test multiple spatial filters
        with pytest.raises(ValueError):
            read_bucket(bucket_dir, extent="dummy", country="dummy")


def test_read_bucket_within_extent(tmp_path):
    """Test read_bucket_within_extent."""
    # Define bucket dir
    bucket_dir = tmp_path
    create_bucket_archive(bucket_dir)

    # Test read full database (extent larger than database extent)
    extent = [-30, 30, -30, 30]
    df_pl = read_bucket(bucket_dir, extent=extent)
    assert df_pl.shape == (150, NUM_COLUMNS)

    # Test read inner portion of database
    extent = [5, 8, 0, 20]
    df_pl = read_bucket(bucket_dir, extent=extent)
    assert df_pl.shape == (33, NUM_COLUMNS)

    # Test with partial extent outside database extent
    extent = [-10, 1, -10, 1]
    df_pl = read_bucket(bucket_dir, extent=extent)
    assert df_pl.shape == (6, NUM_COLUMNS)

    # Test extent outside database extent (no intersecting partitions)
    extent = [-50, -30, -50, -30]
    with pytest.raises(ValueError):
        df_pl = read_bucket(bucket_dir, extent=extent)

    # Test extent outside database extent (with intersecting partitions)
    extent = [-10, -5, -10, -5]
    with pytest.raises(ValueError):
        read_bucket(bucket_dir, extent=extent)

    # Test polars kwargs subsetting
    extent = [-30, 30, -30, 30]
    df_pl = read_bucket(bucket_dir, extent=extent, n_rows=3, columns=["lon", "lat"])
    assert df_pl.shape == (3, 2)
    assert "lon" in df_pl
    assert "lat" in df_pl

    # Test filtering
    extent = [-30, 30, -30, 30]
    df_pl = read_bucket(bucket_dir, extent=extent, glob_pattern="*V07B*")
    assert df_pl.shape == (50, NUM_COLUMNS)


def test_read_bucket_within_country(tmp_path):
    """Test read_bucket_within_country."""
    # Define bucket dir
    bucket_dir = tmp_path
    create_bucket_archive(bucket_dir)

    # Test with country contained in bucket
    df_pl = read_bucket(bucket_dir, country="Nigeria")
    assert df_pl.shape == (42, NUM_COLUMNS)

    # Test with country not contained in bucket
    with pytest.raises(ValueError):
        read_bucket(bucket_dir, country="Switzerland")


def test_read_bucket_within_continent(tmp_path):
    """Test read_bucket_within_continent."""
    # Define bucket dir
    bucket_dir = tmp_path
    create_bucket_archive(bucket_dir)

    # Test with continent contained in bucket
    df_pl = read_bucket(bucket_dir, continent="Africa")
    assert df_pl.shape == (150, NUM_COLUMNS)

    # Test with continent not contained in bucket
    with pytest.raises(ValueError):
        read_bucket(bucket_dir, continent="Europe")


def test_read_bucket_around_point(tmp_path):
    """Test read_bucket_around_point."""
    # Define bucket dir
    bucket_dir = tmp_path
    create_bucket_archive(bucket_dir)

    # Test with point contained in bucket (with distance)
    point = (3, 3)
    distance = 200_000
    df_pl = read_bucket(
        bucket_dir,
        point=point,
        distance=distance,
    )
    assert df_pl.shape == (9, NUM_COLUMNS + 1)
    assert "distance" in df_pl

    # Test with point contained in bucket (with size)
    df_pl = read_bucket(bucket_dir, point=point, size=20)
    assert df_pl.shape == (93, NUM_COLUMNS)

    # Test with point outside bucket  (but intersecting area outside intersecting partitions
    df_pl = read_bucket(bucket_dir, point=(-10, -10), size=25)
    assert df_pl.shape == (15, NUM_COLUMNS)
