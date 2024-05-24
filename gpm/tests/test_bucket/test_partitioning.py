#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:19:17 2024

@author: ghiggi
"""
import pytest
import pandas as pd 
import numpy as np 
import xarray as xr
import polars as pl
from gpm.bucket.partitioning import XYPartitioning
from gpm.bucket.partitioning import (
    get_n_decimals, 
    get_breaks,
    get_labels, 
    get_midpoints,
    get_breaks_and_labels,
    get_array_combinations,
)


def test_get_n_decimals():
    """Ensure decimal count is accurate."""
    assert get_n_decimals(123.456) == 3
    assert get_n_decimals(100) == 0
    assert get_n_decimals(123.0001) == 4


def test_get_breaks():
    """Verify the correct calculation of breaks."""
    breaks = get_breaks(0.5, 0, 2)
    assert np.array_equal(breaks, np.array([0, 0.5, 1.0, 1.5, 2]))


def test_get_labels():
    """Verify correct label generation."""
    labels = get_labels(0.5, 0, 2)
    expected_labels = ['0.25', '0.75', '1.25', '1.75']
    assert labels.tolist() == expected_labels
    
    labels = get_labels(0.999, 0, 2)
    expected_labels =['0.4995', '1.4985', '2.4975']
    assert labels.tolist() == expected_labels


def test_get_midpoints():
    """Verify correct midpoint generation."""
    midpoints = get_midpoints(0.5, 0, 2)
    expected_midpoints = [0.25, 0.75, 1.25, 1.75]
    np.testing.assert_allclose(midpoints, expected_midpoints)
    
    midpoints = get_midpoints(0.999, 0, 2)
    expected_midpoints = [0.4995, 1.4985, 2.4975]
    np.testing.assert_allclose(midpoints, expected_midpoints)
    

def test_get_breaks_and_labels():
    """Ensure both breaks and labels are returned and accurate."""
    breaks, labels = get_breaks_and_labels(0.5, 0, 2)
    assert np.array_equal(breaks, np.array([0, 0.5, 1.0, 1.5, 2]))
    assert labels.tolist() == ['0.25', '0.75', '1.25', '1.75']


def test_get_array_combinations():
    x = np.array([1, 2, 3])
    y = np.array([4, 5])
    expected_result = np.array(
          [[1, 4],
           [2, 4],
           [3, 4],
           [1, 5],
           [2, 5],
           [3, 5]])
    np.testing.assert_allclose(get_array_combinations(x,y), expected_result)     

class TestXYPartitioning:
    """Tests for the XYPartitioning class."""
    
    def test_initialization(self):
        """Test proper initialization of XYPartitioning objects."""
        partitioning = XYPartitioning(xbin="xbin", ybin="ybin", size=(1, 2), extent=[0, 10, 0, 10])
        assert partitioning.size == (1, 2)
        assert partitioning.partitions == ["xbin", "ybin"]
        assert list(partitioning.extent) == [0, 10, 0, 10]
        assert partitioning.shape == (10, 5)
        assert partitioning.n_partitions == 50
        assert partitioning.n_x == 10
        assert partitioning.n_y == 5 
        np.testing.assert_allclose(partitioning.x_breaks, [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])     
        np.testing.assert_allclose(partitioning.y_breaks, [ 0,  2,  4,  6,  8, 10])    
        np.testing.assert_allclose(partitioning.x_midpoints, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])   
        np.testing.assert_allclose(partitioning.y_midpoints, [1., 3., 5., 7., 9.])   
        assert partitioning.x_labels.tolist() == ['0.5', '1.5', '2.5', '3.5', '4.5', '5.5', '6.5', '7.5', '8.5', '9.5'] 
        assert partitioning.y_labels.tolist() == ['1.0', '3.0', '5.0', '7.0', '9.0']

    def test_invalid_initialization(self):
        """Test initialization with invalid extent and size."""
        with pytest.raises(ValueError):
            XYPartitioning(xbin="xbin", ybin="ybin", size=(0.1, 0.2), extent=[10, 0, 0, 10])

        with pytest.raises(TypeError):
            XYPartitioning(xbin="xbin", ybin="ybin", size="invalid", extent=[0, 10, 0, 10])
            
    def test_add_partitions_pandas(self):
        """Test valid partitions are added to a pandas dataframe."""
        # Create test dataframe
        df = pd.DataFrame({
            'x': [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
            'y': [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
        })
        # Create partitioning
        size=(0.5, 0.25)
        extent=[0, 2, 0, 2]
        partitioning = XYPartitioning(xbin="my_xbin", ybin="my_ybin", 
                                      size=size, extent=extent)
        # Add partitions
        df_out = partitioning.add_partitions(df, x="x", y="y", remove_invalid_rows=True)

        # Test results 
        expected_xbin = [0.25, 0.25, 0.25, 0.75, 1.25, 1.75]
        expected_ybin = [0.125, 0.125, 0.375, 0.875, 1.375, 1.875]
        assert df_out["my_xbin"].dtype.name == "category", "X bin are not of categorical type."
        assert df_out["my_ybin"].dtype.name == "category", "Y bin are not of categorical type."
        assert df_out['my_xbin'].astype(float).tolist() == expected_xbin, "X bin are incorrect."
        assert df_out['my_ybin'].astype(float).tolist() == expected_ybin, "Y bin are incorrect."
        
    def test_add_partitions_polars(self):
        """Test valid partitions are added to a polars dataframe."""
        # Create test dataframe
        df = pl.DataFrame(pd.DataFrame({
            'x': [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
            'y': [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
        }))
        # Create partitioning
        size=(0.5, 0.25)
        extent=[0, 2, 0, 2]
        partitioning = XYPartitioning(xbin="my_xbin", ybin="my_ybin", 
                                      size=size, extent=extent)
        # Add partitions
        df_out = partitioning.add_partitions(df, x="x", y="y", remove_invalid_rows=True)
      
        # Test results 
        expected_xbin = [0.25, 0.25, 0.25, 0.75, 1.25, 1.75]
        expected_ybin = [0.125, 0.125, 0.375, 0.875, 1.375, 1.875]
        assert df_out['my_xbin'].dtype == pl.datatypes.Categorical, "X bin are not of categorical type."
        assert df_out['my_ybin'].dtype == pl.datatypes.Categorical, "X bin are not of categorical type."
        assert df_out['my_xbin'].cast(float).to_list() == expected_xbin, "X bin are incorrect."
        assert df_out['my_ybin'].cast(float).to_list() == expected_ybin, "Y bin are incorrect."

    def test_to_xarray(self):
        """Test valid partitions are added to a pandas dataframe."""
        # Create test dataframe
        df = pd.DataFrame({
            'x': [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
            'y': [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
        })
        # Create partitioning
        size=(0.5, 0.25)
        extent=[0, 2, 0, 2]
        partitioning = XYPartitioning(xbin="my_xbin", ybin="my_ybin", 
                                      size=size, extent=extent)
        # Add partitions
        df = partitioning.add_partitions(df, x="x", y="y", remove_invalid_rows=True)
        
        # Aggregate by partitions
        df_grouped = df.groupby(partitioning.partitions, observed=True).median()
        df_grouped["dummy_var"] = 2
       
        # Convert to Dataset
        ds = partitioning.to_xarray(df_grouped, new_x="lon", new_y="lat")
        
        # Test results 
        expected_xbin = [0.25, 0.75, 1.25, 1.75]
        expected_ybin = [0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875]
        assert isinstance(ds, xr.Dataset), "Not a xr.Dataset"
        assert ds["lon"].data.dtype.name != 'object', "xr.Dataset coordinates should not be a string."
        assert ds["lat"].data.dtype.name != 'object', "xr.Dataset coordinates should not be a string."
        assert ds["lon"].data.dtype.name == 'float64', "xr.Dataset coordinates are not float64."
        assert ds["lat"].data.dtype.name == 'float64', "xr.Dataset coordinates are not float64."
        assert "dummy_var" in ds, "The x columns has not become a xr.Dataset variable"
    
    def test_query_labels(self):
        """Test valid labels queries.""" 
        # Create partitioning
        size=(0.5, 0.25)
        extent=[0, 2, 0, 2]
        partitioning = XYPartitioning(xbin="my_xbin", ybin="my_ybin", 
                                      size=size, extent=extent)
        # Test results 
        assert partitioning.query_x_labels(1).tolist() == ['0.75']
        assert partitioning.query_y_labels(1).tolist() == ['0.875']
        assert partitioning.query_x_labels(np.array(1)).tolist() == ['0.75']
        assert partitioning.query_x_labels(np.array([1])).tolist() == ['0.75']
        assert partitioning.query_x_labels(np.array([1, 1])).tolist() == ['0.75', '0.75']
        assert partitioning.query_x_labels([1, 1]).tolist() == ['0.75', '0.75']
        
        x_labels, y_labels = partitioning.query_labels([1,2], [0,1]) 
        assert x_labels.tolist() == ['0.75', '1.75']
        assert y_labels.tolist() == ['0.125', '0.875']
     
        # Test out of extent 
        assert partitioning.query_x_labels([-1, 1]).tolist() == ['nan', '0.75']
        
        # Test with input nan 
        assert partitioning.query_x_labels(np.nan).tolist() == ['nan']
        assert partitioning.query_x_labels(None).tolist() == ['nan']
        
        # Test with input string 
        with pytest.raises(ValueError):
            partitioning.query_x_labels("dummy")
       
    def test_query_midpoints(self):
        """Test valid midpoint queries.""" 
        
        # Create partitioning
        size=(0.5, 0.25)
        extent=[0, 2, 0, 2]
        partitioning = XYPartitioning(xbin="my_xbin", ybin="my_ybin", 
                                      size=size, extent=extent)
        # Test results 
        np.testing.assert_allclose(partitioning.query_x_midpoints(1), [0.75])
        np.testing.assert_allclose(partitioning.query_y_midpoints(1).tolist(), [0.875])
        np.testing.assert_allclose( partitioning.query_x_midpoints(np.array(1)), [0.75])
        np.testing.assert_allclose( partitioning.query_x_midpoints(np.array([1])), [0.75])
        np.testing.assert_allclose( partitioning.query_x_midpoints(np.array([1, 1])), [0.75, 0.75])
        np.testing.assert_allclose( partitioning.query_x_midpoints([1, 1]), [0.75, 0.75])
        
        x_midpoints, y_midpoints = partitioning.query_midpoints([1,2], [0,1]) 
        np.testing.assert_allclose(x_midpoints.tolist(), [0.75, 1.75])
        np.testing.assert_allclose(y_midpoints.tolist(), [0.125, 0.875])

        # Test out of extent 
        np.testing.assert_allclose(partitioning.query_x_midpoints([-1, 1]), [np.nan, 0.75])
        
        # Test with input nan or None
        np.testing.assert_allclose( partitioning.query_x_midpoints(np.nan).tolist(), [np.nan])
        np.testing.assert_allclose( partitioning.query_x_midpoints(None).tolist(), [np.nan])
        
        # Test with input string 
        with pytest.raises(ValueError):
            partitioning.query_x_midpoints("dummy")

    def test_get_partitions_by_extent(self):
        """Test get_partitions_by_extent.""" 
        # Create partitioning
        size=(0.5, 0.25)
        extent=[0, 2, 0, 2]
        partitioning = XYPartitioning(xbin="my_xbin", ybin="my_ybin", 
                                      size=size, extent=extent)
        # Test results with extent within 
        new_extent = [0, 0.5, 0, 0.5]
        labels = partitioning.get_partitions_by_extent(new_extent)
        expected_labels =  np.array([['0.25', '0.125'],
                                     ['0.25', '0.375']])
        assert expected_labels.tolist() ==  labels.tolist()
        
        # Test results with extent outside 
        new_extent = [3, 4, 3, 4]
        labels = partitioning.get_partitions_by_extent(new_extent)
        assert labels.size == 0
        
        # Test results with extent partially overlapping 
        new_extent = [1.5, 4, 1.75, 4] # BUG 
        labels = partitioning.get_partitions_by_extent(new_extent)
        expected_labels = np.array([['1.25', '1.625'],
                                 ['1.75', '1.625'],
                                 ['1.25', '1.875'],
                                 ['1.75', '1.875']])
        assert expected_labels.tolist() ==  labels.tolist()
               
    def test_get_partitions_around_point(self):
       """Test get_partitions_around_point.""" 
       # Create partitioning
       size=(0.5, 0.25)
       extent=[0, 2, 0, 2]
       partitioning = XYPartitioning(xbin="my_xbin", ybin="my_ybin", 
                                    size=size, extent=extent)
       # Test results with point within 
       labels = partitioning.get_partitions_around_point(x=1, y=1, distance=0)
       assert labels.tolist() == [['0.75', '0.875']]
       
       # Test results with point outside 
       labels = partitioning.get_partitions_around_point(x=3, y=3, distance=0)  
       assert labels.size == 0
       
       # Test results with point outside but area within
       labels = partitioning.get_partitions_around_point(x=3, y=3, distance=1)  
       assert labels.tolist() == [['1.75', '1.875']]
             
    def test_quadmesh(self):
       """Test quadmesh.""" 
       size=(1, 1)
       extent=[0, 2, 1, 3]
       partitioning = XYPartitioning(xbin="my_xbin", ybin="my_ybin", 
                                     size=size, extent=extent)
       # Test results
       assert partitioning.quadmesh.shape == (3, 3, 2)
       x_mesh = np.array([[0, 1, 2],
                          [0, 1, 2],
                          [0, 1, 2]])
       y_mesh = np.array([[1, 1, 1],
                          [2, 2, 2],
                          [3, 3, 3]]) 
       np.testing.assert_allclose(partitioning.quadmesh[:,:, 0], x_mesh)
       np.testing.assert_allclose(partitioning.quadmesh[:,:, 1], y_mesh)
       
       # TODO: origin: y: bottom or upper (RIGHT NOW UPPER !) # BUG: increase by descinding
      
         
    def test_to_dict(self):
        # Create partitioning
        size=(0.5, 0.25)
        extent=[0, 2, 0, 2]
        xbin = "my_xbin"
        ybin = "my_ybin"
        partitioning = XYPartitioning(xbin=xbin, ybin=ybin, 
                                      size=size, extent=extent)
        # Test results
        expected_dict = {"name": "XYPartitioning",
                         "extent": extent, 
                         "size": size, 
                         "xbin": xbin, 
                         "ybin": ybin}
        assert partitioning.to_dict() == expected_dict
 
 


 

   