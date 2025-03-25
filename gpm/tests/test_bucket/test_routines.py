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
"""This module tests the bucket routines."""
import os

import pandas as pd
import pytest

from gpm.bucket import LonLatPartitioning
from gpm.bucket.readers import read_dask_partitioned_dataset
from gpm.bucket.routines import (
    check_temporal_partitioning,
    get_partitioning_boundaries,
    get_time_prefix,
    merge_granule_buckets,
    write_bucket,
    write_granules_bucket,
)
from gpm.io.info import get_key_from_filepath
from gpm.tests.utils.fake_datasets import get_orbit_dataarray


def create_granule_dataframe(df_type="pandas"):
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
    df = ds.gpm.to_pandas_dataframe() if df_type == "pandas" else ds.gpm.to_dask_dataframe()
    return df


def granule_to_df_toy_func(filepath):
    df = create_granule_dataframe()
    # Add correct time to simulate filtering of files when updating !
    start_time = get_key_from_filepath(filepath, key="start_time")
    df["time"] = start_time
    return df


####----------------------------------------------------------------------.
#### Test public routines
# # TO DEBUG
# import pathlib
# tmp_path = pathlib.Path("/tmp/bucket01")

DF_TYPES = ["pandas", "dask"]


@pytest.mark.parametrize("df_type", DF_TYPES)
def test_write_bucket(tmp_path, df_type):
    # Define bucket dir
    bucket_dir = tmp_path
    # Create dataframe
    df = create_granule_dataframe(df_type=df_type)
    spatial_partitioning = LonLatPartitioning(size=(10, 10))
    write_bucket(
        df=df,
        bucket_dir=bucket_dir,
        spatial_partitioning=spatial_partitioning,
        # Writer arguments
        filename_prefix="filename_prefix",
        row_group_size="500MB",
    )
    # Check file created with correct prefix
    if df_type == "pandas":
        assert os.path.exists(os.path.join(bucket_dir, "lon_bin=5.0", "lat_bin=5.0", "filename_prefix_0.parquet"))
    else:  # dask
        assert os.path.exists(
            os.path.join(bucket_dir, "lon_bin=5.0", "lat_bin=5.0", "filename_prefix_dask_partition_0_0.parquet"),
        )


@pytest.mark.parametrize("order", [["lon_bin", "lat_bin"], ["lat_bin", "lon_bin"]])
@pytest.mark.parametrize("flavor", ["hive", None])
def test_write_granules_bucket(tmp_path, order, flavor):
    """Test write_granules_bucket routine with parallel=False."""
    # Define bucket dir
    bucket_dir = tmp_path

    # Define filepaths
    filepaths = [
        "2A.GPM.DPR.V9-20211125.20210705-S013942-E031214.041760.V07A.HDF5",  # year 2021
        "2A.GPM.DPR.V9-20211125.20210805-S013942-E031214.041760.V07A.HDF5",  # year 2021
        "2A.GPM.DPR.V9-20211125.20230705-S013942-E031214.041760.V07A.HDF5",
    ]  # year 2023

    # Define spatial partitioning
    # order = ["lat_bin", "lon_bin"]
    # flavor = "hive" # None
    spatial_partitioning = LonLatPartitioning(
        size=(10, 10),
        flavor=flavor,
        order=order,
    )

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

    # Check directories with wished partitioning format created
    if flavor == "hive":
        if order == ["lon_bin", "lat_bin"]:
            expected_directories = [
                "bucket_info.yaml",  # always there
                "lon_bin=-5.0",
                "lon_bin=15.0",
                "lon_bin=5.0",
            ]
        else:
            expected_directories = [
                "bucket_info.yaml",
                "lat_bin=-5.0",
                "lat_bin=15.0",
                "lat_bin=25.0",
                "lat_bin=5.0",
            ]
    elif order == ["lon_bin", "lat_bin"]:
        expected_directories = [
            "-5.0",
            "15.0",
            "5.0",
            "bucket_info.yaml",
        ]
    else:
        expected_directories = [
            "-5.0",
            "5.0",
            "15.0",
            "25.0",
            "bucket_info.yaml",
        ]
    assert sorted(expected_directories) == sorted(os.listdir(bucket_dir))

    # Check parquet files named by granule
    if flavor == "hive":
        if order == ["lon_bin", "lat_bin"]:
            partition_dir = os.path.join(bucket_dir, "lon_bin=-5.0", "lat_bin=5.0")
        else:
            partition_dir = os.path.join(bucket_dir, "lat_bin=5.0", "lon_bin=-5.0")
    elif order == ["lon_bin", "lat_bin"]:
        partition_dir = os.path.join(bucket_dir, "-5.0", "5.0")
    else:
        partition_dir = os.path.join(bucket_dir, "5.0", "-5.0")

    expected_filenames = [os.path.splitext(f)[0] + "_0.parquet" for f in filepaths]
    assert sorted(os.listdir(partition_dir)) == sorted(expected_filenames)


def test_write_granules_bucket_capture_error(tmp_path, capsys):
    bucket_dir = tmp_path

    # Define filepaths
    filepaths = [
        "2A.GPM.DPR.V9-20211125.20210705-S013942-E031214.041760.V07A.HDF5",
        "2A.GPM.DPR.V9-20211125.20230705-S013942-E031214.041760.V07A.HDF5",
    ]

    # Define spatial partitioning
    spatial_partitioning = LonLatPartitioning(size=(10, 10))

    # Define bad granule_to_df_func
    def granule_to_df_func(filepath):
        raise ValueError("check_this_error_captured")

    # Run processing
    write_granules_bucket(
        # Bucket Input/Output configuration
        filepaths=filepaths,
        bucket_dir=bucket_dir,
        spatial_partitioning=spatial_partitioning,
        granule_to_df_func=granule_to_df_func,
        # Processing options
        parallel=False,
    )
    captured = capsys.readouterr()
    assert "check_this_error_captured" in captured.out, "Expected error message not printed"


def test_write_granules_bucket_parallel(tmp_path):
    """Test write_granules_bucket routine with dask distributed client."""
    from dask.distributed import Client, LocalCluster

    # Define bucket dir
    bucket_dir = tmp_path

    # Define filepaths
    filepaths = [
        "2A.GPM.DPR.V9-20211125.20210705-S013942-E031214.041760.V07A.HDF5",  # year 2021
        "2A.GPM.DPR.V9-20211125.20210805-S013942-E031214.041760.V07A.HDF5",  # year 2021
        "2A.GPM.DPR.V9-20211125.20230705-S013942-E031214.041760.V07A.HDF5",
    ]  # year 2023

    # Define parallel options
    parallel = True
    max_concurrent_tasks = None
    max_dask_total_tasks = 2

    # Define spatial partitioning
    spatial_partitioning = LonLatPartitioning(size=(10, 10))

    # Create Dask Distributed LocalCluster
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=True,
    )
    client = Client(cluster)

    # Run processing
    write_granules_bucket(
        # Bucket Input/Output configuration
        filepaths=filepaths,
        bucket_dir=bucket_dir,
        spatial_partitioning=spatial_partitioning,
        granule_to_df_func=granule_to_df_toy_func,
        # Processing options
        parallel=parallel,
        max_concurrent_tasks=max_concurrent_tasks,
        max_dask_total_tasks=max_dask_total_tasks,
    )

    # Close Dask Distributed client
    client.close()

    # Check directories with wished partitioning format created
    expected_directories = [
        "bucket_info.yaml",  # always there
        "lon_bin=-5.0",
        "lon_bin=15.0",
        "lon_bin=5.0",
    ]
    assert expected_directories == sorted(os.listdir(bucket_dir))


def test_merge_granule_buckets(tmp_path):
    """Test merge_granule_buckets routine."""
    # Define bucket dir
    src_bucket_dir = tmp_path / "src"
    dst_bucket_dir = tmp_path / "dst"

    # Define filepaths
    filepaths = [
        "2A.GPM.DPR.V9-20211125.20210705-S013942-E031214.041760.V07A.HDF5",  # year 2021
        "2A.GPM.DPR.V9-20211125.20210805-S013942-E031214.041760.V07A.HDF5",  # year 2021
        "2A.GPM.DPR.V9-20211125.20230705-S013942-E031214.041760.V07A.HDF5",  # year 2023
    ]

    # Define spatial partitioning
    spatial_partitioning = LonLatPartitioning(size=(10, 10))

    # Run processing
    write_granules_bucket(
        # Bucket Input/Output configuration
        filepaths=filepaths,
        bucket_dir=src_bucket_dir,
        spatial_partitioning=spatial_partitioning,
        granule_to_df_func=granule_to_df_toy_func,
        # Processing options
        parallel=False,
    )

    # Merge granules
    merge_granule_buckets(
        src_bucket_dir=src_bucket_dir,
        dst_bucket_dir=dst_bucket_dir,
        write_metadata=True,
    )

    # Check file naming
    partition_dir = os.path.join(dst_bucket_dir, "lon_bin=-5.0", "lat_bin=5.0")
    expected_filenames = ["2021_0.parquet", "2023_0.parquet"]
    assert sorted(os.listdir(partition_dir)) == sorted(expected_filenames)
    assert os.path.exists(os.path.join(dst_bucket_dir, "bucket_info.yaml"))
    assert os.path.exists(os.path.join(dst_bucket_dir, "_common_metadata"))
    assert os.path.exists(os.path.join(dst_bucket_dir, "_metadata"))

    # Assert can be read with Dask too without errors
    df = read_dask_partitioned_dataset(base_dir=dst_bucket_dir)
    assert isinstance(df.compute(), pd.DataFrame)


def test_merge_granule_buckets_update(tmp_path):
    """Test merge_granule_buckets routine."""
    # Define bucket dir
    src_bucket_dir = tmp_path / "src"
    dst_bucket_dir = tmp_path / "dst"

    # Define filepaths
    filepaths = [
        "2A.GPM.DPR.V9-20211125.20210705-S013942-E031214.041760.V07A.HDF5",  # year 2021
        "2A.GPM.DPR.V9-20211125.20210805-S013942-E031214.041760.V07A.HDF5",  # year 2021
    ]

    # Define spatial partitioning
    spatial_partitioning = LonLatPartitioning(size=(10, 10))

    # Run processing
    write_granules_bucket(
        # Bucket Input/Output configuration
        filepaths=filepaths,
        bucket_dir=src_bucket_dir,
        spatial_partitioning=spatial_partitioning,
        granule_to_df_func=granule_to_df_toy_func,
        # Processing options
        parallel=False,
    )

    # Merge granules
    merge_granule_buckets(
        src_bucket_dir=src_bucket_dir,
        dst_bucket_dir=dst_bucket_dir,
        write_metadata=False,
    )

    # Check file naming
    partition_dir = os.path.join(dst_bucket_dir, "lon_bin=-5.0", "lat_bin=5.0")
    expected_filenames = ["2021_0.parquet"]
    assert sorted(os.listdir(partition_dir)) == sorted(expected_filenames)
    assert os.path.exists(os.path.join(dst_bucket_dir, "bucket_info.yaml"))

    # Update granules buckets
    updated_filepaths = ["2A.GPM.DPR.V9-20211125.20230705-S013942-E031214.041760.V07A.HDF5"]
    write_granules_bucket(
        # Bucket Input/Output configuration
        filepaths=updated_filepaths,
        bucket_dir=src_bucket_dir,
        spatial_partitioning=spatial_partitioning,
        granule_to_df_func=granule_to_df_toy_func,
        # Processing options
        parallel=False,
    )

    # Update bucket archive
    merge_granule_buckets(
        src_bucket_dir=src_bucket_dir,
        dst_bucket_dir=dst_bucket_dir,
        # Update archive
        update=True,
        start_time="2023-01-01",
        end_time="2024-01-01",
        write_metadata=False,
    )

    partition_dir = os.path.join(dst_bucket_dir, "lon_bin=-5.0", "lat_bin=5.0")
    expected_filenames = ["2021_0.parquet", "2023_0.parquet"]
    assert sorted(os.listdir(partition_dir)) == sorted(expected_filenames)


class TestMergeGranuleBucketsErrors:
    def test_dst_bucket_does_not_exist(self, tmp_path):
        """Test that OSError is raised when destination bucket directory does not exist in update mode."""
        # Create a temporary source directory.
        src_bucket = tmp_path / "src_bucket"
        src_bucket.mkdir()

        # Use a non-existent destination directory.
        dst_bucket = tmp_path / "nonexistent_bucket"
        with pytest.raises(OSError, match="does not exists"):
            merge_granule_buckets(
                src_bucket_dir=str(src_bucket),
                dst_bucket_dir=str(dst_bucket),
                update=True,
                start_time=pd.Timestamp("2021-01-01"),
                end_time=pd.Timestamp("2022-01-01"),
            )

    def test_write_metadata_error(self, tmp_path):
        """Test that NotImplementedError is raised when write_metadata is True and update is True."""
        # Create both source and destination directories.
        src_bucket = tmp_path / "src_bucket"
        dst_bucket = tmp_path / "dst_bucket"
        src_bucket.mkdir()
        dst_bucket.mkdir()
        with pytest.raises(NotImplementedError, match="update=True.*metadata"):
            merge_granule_buckets(
                src_bucket_dir=str(src_bucket),
                dst_bucket_dir=str(dst_bucket),
                update=True,
                write_metadata=True,
                start_time=pd.Timestamp("2021-01-01"),
                end_time=pd.Timestamp("2022-01-01"),
            )

    def test_missing_start_end_time(self, tmp_path):
        """Test that ValueError is raised when start_time or end_time is missing in update mode."""
        # Create both source and destination directories.
        src_bucket = tmp_path / "src_bucket"
        dst_bucket = tmp_path / "dst_bucket"
        src_bucket.mkdir()
        dst_bucket.mkdir()
        # Test with start_time missing.
        with pytest.raises(ValueError, match="Define both 'start_time' and 'end_time'"):
            merge_granule_buckets(
                src_bucket_dir=str(src_bucket),
                dst_bucket_dir=str(dst_bucket),
                update=True,
                start_time=None,
                end_time=pd.Timestamp("2022-01-01"),
            )
        # Test with end_time missing.
        with pytest.raises(ValueError, match="Define both 'start_time' and 'end_time'"):
            merge_granule_buckets(
                src_bucket_dir=str(src_bucket),
                dst_bucket_dir=str(dst_bucket),
                update=True,
                start_time=pd.Timestamp("2021-01-01"),
                end_time=None,
            )


####----------------------------------------------------------------------.
#### Test routines inner function
class TestGetPartitioningBoundaries:
    def test_year_partitioning(self):
        """Test yearly boundaries are aligned to January 1 when end time is not aligned."""
        start_time = pd.Timestamp("2021-07-05 01:39:42")
        end_time = pd.Timestamp("2021-10-15 12:00:00")
        result = get_partitioning_boundaries(start_time, end_time, "year")
        expected = pd.date_range(
            start=pd.Timestamp("2021-01-01"),
            end=pd.Timestamp("2022-01-01"),
            freq="YS",
        )
        pd.testing.assert_index_equal(result, expected)

    def test_year_partitioning_aligned(self):
        """Test yearly boundaries do not add extra year when end time is aligned."""
        start_time = pd.Timestamp("2021-01-01 00:00:00")
        end_time = pd.Timestamp("2022-01-01 00:00:00")
        result = get_partitioning_boundaries(start_time, end_time, "year")
        expected = pd.date_range(
            start=pd.Timestamp("2021-01-01"),
            end=pd.Timestamp("2022-01-01"),
            freq="YS",
        )
        pd.testing.assert_index_equal(result, expected)

    def test_month_partitioning(self):
        """Test monthly boundaries are aligned to the first day of the month when end time is not aligned."""
        start_time = pd.Timestamp("2021-07-05 01:39:42")
        end_time = pd.Timestamp("2021-10-15 12:00:00")
        result = get_partitioning_boundaries(start_time, end_time, "month")
        expected = pd.date_range(
            start=pd.Timestamp("2021-07-01 00:00:00"),
            end=pd.Timestamp("2021-11-01 00:00:00"),
            freq="MS",
        )
        pd.testing.assert_index_equal(result, expected)

    def test_month_partitioning_aligned(self):
        """Test monthly boundaries do not add extra month when end time is aligned."""
        start_time = pd.Timestamp("2021-07-01 00:00:00")
        end_time = pd.Timestamp("2021-10-01 00:00:00")
        result = get_partitioning_boundaries(start_time, end_time, "month")
        expected = pd.date_range(
            start=pd.Timestamp("2021-07-01 00:00:00"),
            end=pd.Timestamp("2021-10-01 00:00:00"),
            freq="MS",
        )
        pd.testing.assert_index_equal(result, expected)

    def test_quarter_partitioning(self):
        """Test quarterly boundaries are aligned to quarter starts when end time is not aligned."""
        start_time = pd.Timestamp("2021-08-05 01:39:42")  # Q3 start should be July 1, 2021.
        end_time = pd.Timestamp("2021-11-15 12:00:00")  # Not aligned, so adjusted forward.
        result = get_partitioning_boundaries(start_time, end_time, "quarter")
        # For this test, end_time quarter start at Oct 1, 2021 for ends on Jan 1, 2022.
        # - Since not aligned we select the quarter ends: Jan 1, 2022.
        expected = pd.date_range(
            start=pd.Timestamp("2021-07-01 00:00:00"),
            end=pd.Timestamp("2022-01-01 00:00:00"),
            freq="QS",
        )
        pd.testing.assert_index_equal(result, expected)

    def test_quarter_partitioning_aligned(self):
        """Test quarterly boundaries do not add extra quarter when end time is aligned."""
        start_time = pd.Timestamp("2021-07-01 00:00:00")  # Q3 start
        end_time = pd.Timestamp("2021-10-01 00:00:00")  # Q4 start, already aligned
        result = get_partitioning_boundaries(start_time, end_time, "quarter")
        expected = pd.date_range(
            start=pd.Timestamp("2021-07-01 00:00:00"),
            end=pd.Timestamp("2021-10-01 00:00:00"),
            freq="QS",
        )
        pd.testing.assert_index_equal(result, expected)

    def test_day_partitioning(self):
        """Test daily boundaries are aligned to midnight when end time is not aligned."""
        start_time = pd.Timestamp("2021-07-05 01:39:42")
        end_time = pd.Timestamp("2021-07-07 15:30:00")
        result = get_partitioning_boundaries(start_time, end_time, "day")
        expected = pd.date_range(
            start=pd.Timestamp("2021-07-05 00:00:00"),
            end=pd.Timestamp("2021-07-08 00:00:00"),
            freq="D",
        )
        pd.testing.assert_index_equal(result, expected)

    def test_day_partitioning_aligned(self):
        """Test daily boundaries do not add extra day when end time is aligned."""
        start_time = pd.Timestamp("2021-07-05 00:00:00")
        end_time = pd.Timestamp("2021-07-07 00:00:00")
        result = get_partitioning_boundaries(start_time, end_time, "day")
        expected = pd.date_range(
            start=pd.Timestamp("2021-07-05 00:00:00"),
            end=pd.Timestamp("2021-07-07 00:00:00"),
            freq="D",
        )
        pd.testing.assert_index_equal(result, expected)

    def test_invalid_partitioning(self):
        """Test that an invalid temporal partitioning raises NotImplementedError."""
        start_time = pd.Timestamp("2021-07-05 01:39:42")
        end_time = pd.Timestamp("2021-07-07 15:30:00")
        with pytest.raises(NotImplementedError):
            get_partitioning_boundaries(start_time, end_time, "invalid")


class TestGetTimePrefix:
    def test_year_prefix(self):
        """Test year prefix returns correct year."""
        timestep = pd.Timestamp("2021-05-06 13:20:00")
        assert get_time_prefix(timestep, "year") == "2021"

    def test_month_prefix(self):
        """Test month prefix returns correct year and month."""
        timestep = pd.Timestamp("2021-03-15 09:10:00")
        assert get_time_prefix(timestep, "month") == "2021_3"

    def test_quarter_prefix_q1(self):
        """Test quarter prefix returns correct quarter for Q1."""
        timestep = pd.Timestamp("2021-02-20 08:00:00")
        assert get_time_prefix(timestep, "quarter") == "2021_1"

    def test_quarter_prefix_q2(self):
        """Test quarter prefix returns correct quarter for Q2."""
        timestep = pd.Timestamp("2021-05-10 14:30:00")
        assert get_time_prefix(timestep, "quarter") == "2021_2"

    def test_quarter_prefix_q3(self):
        """Test quarter prefix returns correct quarter for Q3."""
        timestep = pd.Timestamp("2021-08-01 00:00:00")
        assert get_time_prefix(timestep, "quarter") == "2021_3"

    def test_quarter_prefix_q4(self):
        """Test quarter prefix returns correct quarter for Q4."""
        timestep = pd.Timestamp("2021-11-11 23:59:59")
        assert get_time_prefix(timestep, "quarter") == "2021_4"

    def test_day_prefix(self):
        """Test day prefix returns correct year, month and day."""
        timestep = pd.Timestamp("2021-07-05 01:39:42")
        assert get_time_prefix(timestep, "day") == "2021_7_5"

    def test_invalid_partitioning(self):
        """Test invalid partitioning raises NotImplementedError."""
        timestep = pd.Timestamp("2021-07-05 01:39:42")
        with pytest.raises(NotImplementedError):
            get_time_prefix(timestep, "hour")


class TestCheckTemporalPartitioning:
    def test_valid_year(self):
        """Test valid 'year' partitioning returns the same value."""
        assert check_temporal_partitioning("year") == "year"

    def test_valid_month(self):
        """Test valid 'month' partitioning returns the same value."""
        assert check_temporal_partitioning("month") == "month"

    def test_valid_season(self):
        """Test valid 'season' partitioning returns the same value."""
        assert check_temporal_partitioning("season") == "season"

    def test_valid_quarter(self):
        """Test valid 'quarter' partitioning returns the same value."""
        assert check_temporal_partitioning("quarter") == "quarter"

    def test_invalid_value(self):
        """Test that an invalid string raises a ValueError."""
        with pytest.raises(ValueError):
            check_temporal_partitioning("hour")

    def test_invalid_type(self):
        """Test that a non-string input raises a TypeError."""
        with pytest.raises(TypeError):
            check_temporal_partitioning(2021)
