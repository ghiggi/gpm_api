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
from gpm.tests.utils.fake_datasets import get_orbit_dataarray
from gpm.bucket import GeographicPartitioning
from gpm.bucket.routines import write_granules_bucket, merge_granule_buckets, write_bucket
from gpm.bucket.readers import read_dask_partitioned_dataset


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
   if df_type == "pandas":
       df = ds.gpm.to_pandas_dataframe()
   else: 
       df = ds.gpm.to_dask_dataframe()
   return df 


def granule_to_df_toy_func(filepath): 
   return create_granule_dataframe()


@pytest.mark.parametrize("df_type", ["pandas", "dask"])
def test_write_bucket(tmp_path, df_type): 
    # Define bucket dir 
    import pathlib
    tmp_path = pathlib.Path("/tmp/bucket13")
    bucket_dir = tmp_path
    # Create dataframe 
    df = create_granule_dataframe(df_type=df_type)
    partitioning = GeographicPartitioning(size=(10, 10))
    write_bucket(
        df=df,
        bucket_dir=bucket_dir,
        partitioning=partitioning,
        # Writer arguments
        filename_prefix="filename_prefix",
        row_group_size="500MB",
    )
    # Check file created with correct prefix 
    assert os.path.exists(os.path.join(bucket_dir, "lon_bin=5.0","lat_bin=5.0", "filename_prefix_0.parquet"))
    
    
@pytest.mark.parametrize("partitions", (["lon_bin", "lat_bin"], ["lat_bin", "lon_bin"]))
@pytest.mark.parametrize("partitioning_flavor", ["hive", None])                                        
def test_write_granules_bucket(tmp_path, partitions, partitioning_flavor):    
    """Test write_granules_bucket routine with parallel=False."""
    # Define bucket dir 
    # import pathlib
    # tmp_path = pathlib.Path("/tmp/bucket12")
    bucket_dir = tmp_path
    
    # Define filepaths 
    filepaths = ["2A.GPM.DPR.V9-20211125.20210705-S013942-E031214.041760.V07A.HDF5", # year 2021
                 "2A.GPM.DPR.V9-20211125.20210805-S013942-E031214.041760.V07A.HDF5", # year 2021
                 "2A.GPM.DPR.V9-20211125.20230705-S013942-E031214.041760.V07A.HDF5"] # year 2023 
    
    # Define partitioning
    # partitions = ["lat_bin", "lon_bin"]
    # partitioning_flavor = "hive" # None     
    partitioning = GeographicPartitioning(size=(10, 10), 
                                          partitioning_flavor=partitioning_flavor,
                                          partitions=partitions)
     
    # Run processing
    write_granules_bucket(
        # Bucket Input/Output configuration
        filepaths=filepaths,
        bucket_dir=bucket_dir,
        partitioning = partitioning, 
        granule_to_df_func=granule_to_df_toy_func,
        # Processing options
        parallel=False,
    )
        
    # Check directories with wished partitioning format created 
    if partitioning_flavor == "hive":
        if partitions == ["lon_bin", "lat_bin"]:
            expected_directories = [
             'bucket_info.yaml', # always there
             'lon_bin=-5.0',
             'lon_bin=15.0',
             'lon_bin=5.0',
            ]
        else: 
            expected_directories = [
             'bucket_info.yaml',
             'lat_bin=-5.0',
             'lat_bin=15.0',
             'lat_bin=25.0',
             'lat_bin=5.0',
            ]
    else: 
        if partitions == ["lon_bin", "lat_bin"]:
            expected_directories = [
            '-5.0', '15.0', '5.0', 'bucket_info.yaml',
            ]
        else: 
            expected_directories = [
            '-5.0', '5.0', '15.0', '25.0', 'bucket_info.yaml',
            ]
    assert sorted(expected_directories) == sorted(os.listdir(bucket_dir))
        
    # Check parquet files named by granule 
    if partitioning_flavor == "hive": 
        if partitions == ["lon_bin", "lat_bin"]:
            partition_dir = os.path.join(bucket_dir, "lon_bin=-5.0", "lat_bin=5.0")
        else:
            partition_dir = os.path.join(bucket_dir, "lat_bin=5.0", "lon_bin=-5.0") 
    else: 
        if partitions == ["lon_bin", "lat_bin"]:
            partition_dir = os.path.join(bucket_dir, "-5.0", "5.0")
        else: 
            partition_dir = os.path.join(bucket_dir, "5.0",  "-5.0")
            
    expected_filenames = [os.path.splitext(f)[0] + "_0.parquet"  for f in filepaths]
    assert sorted(os.listdir(partition_dir)) == sorted(expected_filenames)
    

def test_write_granules_bucket_capture_error(tmp_path, capsys):
    bucket_dir = tmp_path
    
    # Define filepaths 
    filepaths = ["2A.GPM.DPR.V9-20211125.20210705-S013942-E031214.041760.V07A.HDF5", 
                 "2A.GPM.DPR.V9-20211125.20230705-S013942-E031214.041760.V07A.HDF5"]
    
    # Define partitioning
    partitioning = GeographicPartitioning(size=(10, 10))
    
    # Define bad granule_to_df_func 
    def granule_to_df_func(filepath): 
        raise ValueError("check_this_error_captured")
        
    # Run processing
    write_granules_bucket(
        # Bucket Input/Output configuration
        filepaths=filepaths,
        bucket_dir=bucket_dir,
        partitioning = partitioning, 
        granule_to_df_func=granule_to_df_func,
        # Processing options
        parallel=False,
    )
    captured = capsys.readouterr()
    assert "check_this_error_captured" in captured.out, "Expected error message not printed"
   
            
def test_write_granules_bucket_parallel(tmp_path):
    """Test write_granules_bucket routine with dask distributed client."""
    from dask.distributed import LocalCluster, Client
    
    # Define bucket dir 
    # bucket_dir = "/tmp/test_bucket1"
    bucket_dir = tmp_path
    
    # Define filepaths 
    filepaths = ["2A.GPM.DPR.V9-20211125.20210705-S013942-E031214.041760.V07A.HDF5", # year 2021
                 "2A.GPM.DPR.V9-20211125.20210805-S013942-E031214.041760.V07A.HDF5", # year 2021
                 "2A.GPM.DPR.V9-20211125.20230705-S013942-E031214.041760.V07A.HDF5"] # year 2023 
    
    # Define parallel options
    parallel = True
    max_concurrent_tasks = None
    max_dask_total_tasks = 2
    
    # Define partitioning
    partitioning = GeographicPartitioning(size=(10, 10))
 
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
        partitioning = partitioning, 
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
     'bucket_info.yaml', # always there
     'lon_bin=-5.0',
     'lon_bin=15.0',
     'lon_bin=5.0',
    ]
    assert expected_directories == sorted(os.listdir(bucket_dir))
    
    
def test_merge_granule_buckets(tmp_path):
    """Test merge_granule_buckets routine."""
    
    # Define bucket dir 
    # import pathlib
    # tmp_path = pathlib.Path("/tmp/bucket")
    src_bucket_dir = tmp_path / "src"
    dst_bucket_dir = tmp_path / "dst"
    
    # Define filepaths 
    filepaths = ["2A.GPM.DPR.V9-20211125.20210705-S013942-E031214.041760.V07A.HDF5", # year 2021
                 "2A.GPM.DPR.V9-20211125.20210805-S013942-E031214.041760.V07A.HDF5", # year 2021
                 "2A.GPM.DPR.V9-20211125.20230705-S013942-E031214.041760.V07A.HDF5"] # year 2023 
    
    # Define partitioning
    partitioning = GeographicPartitioning(size=(10, 10))
        
    # Run processing
    write_granules_bucket(
        # Bucket Input/Output configuration
        filepaths=filepaths,
        bucket_dir=src_bucket_dir,
        partitioning = partitioning, 
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
      

