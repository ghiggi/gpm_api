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
"""This module test the xarray utilities."""

import numpy as np
import pytest
import xarray as xr

from gpm.utils.xarray import (
    check_is_xarray,
    check_is_xarray_dataarray,
    check_is_xarray_dataset,
    check_variable_availabilty,
    ensure_dim_order_dataarray,
    ensure_dim_order_dataset,
    get_dataset_variables,
    get_dimensions_without,
    get_xarray_variable,
    squeeze_unsqueeze_dataarray,
    squeeze_unsqueeze_dataset,
    xr_ensure_dimension_order,
    xr_squeeze_unsqueeze,
)


def test_check_is_xarray() -> None:
    """Test check_is_xarray function."""
    # Should not raise exception
    check_is_xarray(xr.DataArray())
    check_is_xarray(xr.Dataset())

    # Should raise an exception
    for invalid in [None, 0, "string", [], {}]:
        with pytest.raises(TypeError):
            check_is_xarray(invalid)


def test_check_is_xarray_dataarray() -> None:
    """Test check_is_xarray_dataarray function."""
    # Should not raise error
    check_is_xarray_dataarray(xr.DataArray())

    # Should raise an error
    for invalid in [None, 0, "string", [], {}, xr.Dataset()]:
        with pytest.raises(TypeError):
            check_is_xarray_dataarray(invalid)


def test_check_is_xarray_dataset() -> None:
    """Test check_is_xarray_dataset function."""
    # Should not raise error
    check_is_xarray_dataset(xr.Dataset())

    # Should raise an error
    for invalid in [None, 0, "string", [], {}, xr.DataArray()]:
        with pytest.raises(TypeError):
            check_is_xarray_dataset(invalid)


def test_get_dataset_variables() -> None:
    """Test get_dataset_variables function."""
    variables = ["variable2", "variable1"]
    variables_sorted = ["variable1", "variable2"]
    ds = xr.Dataset({v: xr.DataArray() for v in variables})

    assert get_dataset_variables(ds) == variables
    assert get_dataset_variables(ds, sort=True) == variables_sorted


def test_check_variable_availabilty() -> None:
    """Test check_variable_availabilty function."""
    variable = "variable"
    ds = xr.Dataset({"variable": xr.DataArray()})

    # Should not raise any error
    check_variable_availabilty(ds, variable, "arg_name")

    # Should raise an error
    with pytest.raises(ValueError):
        check_variable_availabilty(ds, "other_variable", "arg_name")


def test_get_xarray_variable() -> None:
    """Test get_xarray_variable function."""
    variable = "variable"
    ds = xr.Dataset({"variable": xr.DataArray([1, 2, 3])})
    da = ds["variable"]

    xr.testing.assert_identical(get_xarray_variable(ds, variable), da)
    xr.testing.assert_identical(get_xarray_variable(da, variable), da)

    with pytest.raises(TypeError):
        get_xarray_variable([], variable)

    with pytest.raises(ValueError):
        get_xarray_variable(ds, "other_variable")


def test_get_dimensions_without() -> None:
    """Test get_dimensions_without function."""
    dataarray_3d = xr.DataArray(np.random.rand(2, 2, 2), dims=["range", "cross_track", "along_track"])
    # Test with list input
    returned_dims = get_dimensions_without(dataarray_3d, ["cross_track", "range"])
    expected_dims = ["along_track"]
    assert returned_dims == expected_dims

    # Test with input string
    returned_dims = get_dimensions_without(dataarray_3d, "range")
    expected_dims = ["cross_track", "along_track"]
    assert returned_dims == expected_dims


def test_ensure_dim_order_dataarray():
    """Test the ensure_dim_order_dataarray function."""
    # Create a sample DataArray
    data = np.random.rand(2, 3, 4)
    coords = {"time": [1, 2], "lat": [1, 2, 3], "lon": [10, 20, 30, 40]}
    da = xr.DataArray(data, coords=coords, dims=("time", "lat", "lon"))
    da = da.assign_coords({"height": (("lon", "lat"), np.random.rand(4, 3))})  # coord with different dim order

    # Test with dimensions which are removed
    def remove_dimension(da):
        return da.isel(lat=0).transpose("lon", "time")

    da_result = ensure_dim_order_dataarray(da, remove_dimension)
    assert da_result.dims == ("time", "lon")

    # Test with dimensions that are added
    def add_dimension(da):
        return da.expand_dims(dim="new_dim").transpose("new_dim", "lat", "lon", "time", ...)

    da_result = ensure_dim_order_dataarray(da, add_dimension)
    assert da_result.dims == ("time", "lat", "lon", "new_dim")  # new dimensions as last dimensions

    # Test that also the coordinates are ordered as the input !
    assert da_result["height"].dims == ("lon", "lat")

    # Test raise error if function return another object
    def bad_function(da):
        return 1

    with pytest.raises(TypeError):
        ensure_dim_order_dataarray(da, bad_function)


def test_ensure_dim_order_dataset():
    """Test the ensure_dim_order_dataset function."""
    # Create sample DataArray
    data = np.random.rand(2, 3, 4)
    coords = {"time": [1, 2], "lat": [1, 2, 3], "lon": [10, 20, 30, 40]}
    da1 = xr.DataArray(data, coords=coords, dims=("time", "lat", "lon"))
    da1 = da1.assign_coords({"height": (("lon", "lat"), np.random.rand(4, 3))})
    da2 = da1.copy().transpose("lon", "time", "lat")
    ds = xr.Dataset({"var1": da1, "var2": da2})

    # Test with dimensions which are removed
    def remove_dimension(ds):
        return ds.isel(lat=0).transpose("lon", "time")

    ds_result = ensure_dim_order_dataset(ds, remove_dimension)
    assert ds_result["var1"].dims == ("time", "lon")
    assert ds_result["var2"].dims == ("lon", "time")

    # Test with dimensions that are added
    def add_dimension(ds):
        return ds.expand_dims(dim="new_dim").transpose("new_dim", "lat", "lon", "time", ...)

    ds_result = ensure_dim_order_dataset(ds, add_dimension)
    assert ds_result["var1"].dims == ("time", "lat", "lon", "new_dim")  # new dimensions as last dimensions
    assert ds_result["var2"].dims == ("lon", "time", "lat", "new_dim")

    # Test that also the coordinates are ordered as the input !
    assert ds_result["height"].dims == ("lon", "lat")

    # Test raise error if function return another object
    def bad_function(da):
        return 1

    with pytest.raises(TypeError):
        ensure_dim_order_dataset(ds, bad_function)


def test_xr_ensure_dimension_order():
    """Test the decorator xr_ensure_dimension_order."""
    # Create a sample DataArray
    data = np.random.rand(2, 3, 4)
    coords = {"time": [1, 2], "lat": [1, 2, 3], "lon": [10, 20, 30, 40]}
    da = xr.DataArray(data, coords=coords, dims=("time", "lat", "lon"))

    @xr_ensure_dimension_order
    def custom_func(da):
        return da.isel(lat=0).expand_dims(dim="new_dim").transpose("new_dim", "lon", "time", ...)

    da_result = custom_func(da)
    assert da_result.dims == ("time", "lon", "new_dim")


def test_squeeze_unsqueeze_dataarray():
    """Test the squeeze_unsqueeze_dataarray function."""
    # Create a sample DataArray
    data = np.random.rand(3, 4, 1, 1)
    coords = {
        "lat": [1, 2, 3],
        "lon": [10, 20, 30, 40],
        "dim1": [1],
        "dim2": [2],  # 1 fail the test
        "coord1": ("dim1", [1]),
        "dummy_coord": 0,
    }
    da = xr.DataArray(data, coords=coords, dims=("lat", "lon", "dim1", "dim2"))

    # Test the identity function
    def identity_function(da):
        return da.squeeze()

    # Apply the function using _squeeze_unsqueeze_dataarray
    da_result = squeeze_unsqueeze_dataarray(da, identity_function)

    # Check that the shape is unchanged
    assert da_result.shape == da.shape
    xr.testing.assert_identical(da, da_result)

    # Test a function that drop and add a dimension
    def custom_function(da):
        return da.isel(lat=0).squeeze().expand_dims(dim="new_dim").transpose("new_dim", "lon", ...)

    # Apply the function using _squeeze_unsqueeze_dataarray
    da_result = squeeze_unsqueeze_dataarray(da, custom_function)
    assert da_result.dims == ("lon", "dim1", "dim2", "new_dim")

    # Test raise error if function return another object
    def bad_function(da):
        return 1

    with pytest.raises(TypeError):
        squeeze_unsqueeze_dataarray(da, bad_function)


def test_squeeze_unsqueeze_dataset():
    """Test the squeeze_unsqueeze_dataset function."""
    # Create a sample DataArray
    data = np.random.rand(3, 4, 1, 1)
    coords = {
        "lat": [1, 2, 3],
        "lon": [10, 20, 30, 40],
        "dim1": [1],
        "dim2": [2],  # 1 fail the test
        "coord1": ("dim1", [1]),
        "dummy_coord": 0,
    }
    da1 = xr.DataArray(data, coords=coords, dims=("lat", "lon", "dim1", "dim2"))
    da2 = da1.transpose("dim1", "dim2", ...)
    ds = xr.Dataset({"var1": da1, "var2": da2})

    # Test the identity function and check that the dataset are identical
    def identity_function(ds):
        return ds.squeeze()

    ds_result = squeeze_unsqueeze_dataset(ds, identity_function)
    xr.testing.assert_identical(ds, ds_result)

    # Test a function that drop and add a dimension
    def custom_function(ds):
        return ds.isel(lat=0).squeeze().expand_dims(dim="new_dim").transpose("new_dim", "lon", ...)

    ds_result = squeeze_unsqueeze_dataset(ds, custom_function)
    assert ds_result["var1"].dims == ("lon", "dim1", "dim2", "new_dim")
    assert ds_result["var2"].dims == ("dim1", "dim2", "lon", "new_dim")

    # Test raise error if function return another object
    def bad_function(ds):
        return 1

    with pytest.raises(TypeError):
        squeeze_unsqueeze_dataset(ds, bad_function)


def test_xr_squeeze_unsqueeze():
    """Test the decorator xr_squeeze_unsqueeze."""
    # Create a sample DataArray
    data = np.random.rand(3, 4, 1, 1)
    coords = {
        "lat": [1, 2, 3],
        "lon": [10, 20, 30, 40],
        "dim1": [1],
        "dim2": [2],  # 1 fail the test
        "coord1": ("dim1", [1]),
        "dummy_coord": 0,
    }
    da1 = xr.DataArray(data, coords=coords, dims=("lat", "lon", "dim1", "dim2"))
    da2 = da1.transpose("dim1", "dim2", ...)
    ds = xr.Dataset({"var1": da1, "var2": da2})

    # Test the identity function and check that the dataset are identical
    @xr_squeeze_unsqueeze
    def identity_function(xr_obj):
        return xr_obj.squeeze()

    # - With xarray Dataset
    xr.testing.assert_identical(ds, identity_function(ds))

    # - With xarray DataArray
    xr.testing.assert_identical(da1, identity_function(da1))
