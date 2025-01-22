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
    get_default_variable,
    get_dimensions_without,
    get_xarray_variable,
    squeeze_unsqueeze_dataarray,
    squeeze_unsqueeze_dataset,
    unstack_datarray_dimension,
    unstack_dataset_dimension,
    unstack_dimension,
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


####------------------------------------------------------------------------


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


class TestGetDefaultVariable:
    """Test get_default_variable."""

    def test_no_var(self):
        """Test when the variable is not present in the dataset."""
        ds = xr.Dataset({"other_var": (("x", "y"), np.zeros((2, 2)))})
        with pytest.raises(ValueError, match="None of"):
            get_default_variable(ds, possible_variables="not_present")

    def test_both_vars(self):
        """Test when both variables are present in the dataset."""
        ds = xr.Dataset(
            {
                "var1": (("x", "y"), np.zeros((2, 2))),
                "var2": (("x", "y"), np.zeros((2, 2))),
            },
        )
        with pytest.raises(ValueError, match="Multiple variables found"):
            get_default_variable(ds, possible_variables=["var1", "var2"])

    def test_single_var(self):
        """Test when possible_variables is just a string (one variable)."""
        ds = xr.Dataset(
            {
                "var": (("x", "y"), np.zeros((2, 2))),
            },
        )
        assert get_default_variable(ds, possible_variables="var") == "var"

    def test_custom_list(self):
        """Test when one of possible_variables is available)."""
        ds = xr.Dataset(
            {
                "var1": (("x", "y"), np.zeros((2, 2))),
            },
        )
        assert get_default_variable(ds, ["var1", "var2", "var3"]) == "var1"


def test_get_dimensions_without() -> None:
    """Test get_dimensions_without function."""
    rng = np.random.default_rng()
    data = rng.random((2, 2, 2))
    dataarray_3d = xr.DataArray(data, dims=["range", "cross_track", "along_track"])
    # Test with list input
    returned_dims = get_dimensions_without(dataarray_3d, ["cross_track", "range"])
    expected_dims = ["along_track"]
    assert returned_dims == expected_dims

    # Test with input string
    returned_dims = get_dimensions_without(dataarray_3d, "range")
    expected_dims = ["cross_track", "along_track"]
    assert returned_dims == expected_dims


####------------------------------------------------------------------------


def test_ensure_dim_order_dataarray():
    """Test the ensure_dim_order_dataarray function."""
    # Create a sample DataArray
    rng = np.random.default_rng()
    data = rng.random((2, 3, 4))
    coords = {"time": [1, 2], "lat": [1, 2, 3], "lon": [10, 20, 30, 40]}
    da = xr.DataArray(data, coords=coords, dims=("time", "lat", "lon"))
    da = da.assign_coords({"height": (("lon", "lat"), rng.random((4, 3)))})  # coord with different dim order

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
    rng = np.random.default_rng()
    data = rng.random((2, 3, 4))
    coords = {"time": [1, 2], "lat": [1, 2, 3], "lon": [10, 20, 30, 40]}
    da1 = xr.DataArray(data, coords=coords, dims=("time", "lat", "lon"))
    da1 = da1.assign_coords({"height": (("lon", "lat"), rng.random((4, 3)))})
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
    rng = np.random.default_rng()
    data = rng.random((2, 3, 4))
    coords = {"time": [1, 2], "lat": [1, 2, 3], "lon": [10, 20, 30, 40]}
    da = xr.DataArray(data, coords=coords, dims=("time", "lat", "lon"))

    @xr_ensure_dimension_order
    def custom_func(da):
        return da.isel(lat=0).expand_dims(dim="new_dim").transpose("new_dim", "lon", "time", ...)

    da_result = custom_func(da)
    assert da_result.dims == ("time", "lon", "new_dim")


####------------------------------------------------------------------------
def test_squeeze_unsqueeze_dataarray():
    """Test the squeeze_unsqueeze_dataarray function."""
    # Create a sample DataArray
    rng = np.random.default_rng()
    data = rng.random((3, 4, 1, 1))
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
    rng = np.random.default_rng()
    data = rng.random((3, 4, 1, 1))
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
    rng = np.random.default_rng()
    data = rng.random((3, 4, 1, 1))
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


####------------------------------------------------------------------------


def test_unstack_datarray_dimension_basic():
    """Test splitting a DataArray along a dimension without prefix/suffix."""
    da = xr.DataArray(
        np.array([[1, 2], [3, 4]]),
        dims=("dim1", "dim2"),
        coords={"dim1": ["A", "B"], "dim2": [10, 20]},
        name="var",
    )
    ds = unstack_datarray_dimension(da, dim="dim1")
    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == {"varA", "varB"}
    np.testing.assert_array_equal(ds["varA"].values, [1, 2])
    np.testing.assert_array_equal(ds["varB"].values, [3, 4])


def test_unstack_datarray_dimension_with_prefix_suffix():
    """Test splitting a DataArray with prefix and suffix."""
    da = xr.DataArray(
        np.array([5, 6, 7]),
        dims=("dim1",),
        coords={"dim1": ["X", "Y", "Z"]},
        name="var",
    )
    ds = unstack_datarray_dimension(da, dim="dim1", prefix="split_", suffix="_")
    assert set(ds.data_vars) == {"split_var_X", "split_var_Y", "split_var_Z"}
    np.testing.assert_array_equal(ds["split_var_X"].values, [5])
    np.testing.assert_array_equal(ds["split_var_Y"].values, [6])
    np.testing.assert_array_equal(ds["split_var_Z"].values, [7])


def test_unstack_datarray_dimension_with_coord_dim():
    """Test splitting a DataArray when a coordinate shares the split dimension."""
    da = xr.DataArray(
        np.array([[1, 2], [3, 4]]),
        dims=("dim1", "dim2"),
        coords={
            "dim1": ["A", "B"],
            "dim2": [10, 20],
            "other_coord": ("dim1", [100, 200]),  # coordinate with target dimension
            "another_coord": ("dim1", [100, 200]),  # coordinate with target dimension
        },
        name="var",
    )

    # Test coord_handling="keep"
    ds = unstack_datarray_dimension(da, dim="dim1")
    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == {"varA", "varB"}
    assert "dim2" in ds.dims
    np.testing.assert_array_equal(ds["varA"].values, [1, 2])
    np.testing.assert_array_equal(ds["varB"].values, [3, 4])
    assert "dim1" in ds.dims  # "dim1" is kept
    assert "other_coord" in ds.coords  # coord is not dropped or unstacked
    assert "another_coord" in ds.coords  # coord is not dropped or unstacked
    assert ds.coords["other_coord"].dims == ("dim1",)
    assert ds.coords["another_coord"].dims == ("dim1",)
    np.testing.assert_array_equal(ds.coords["other_coord"].values, [100, 200])

    # Test coord_handling="drop"
    ds = unstack_datarray_dimension(da, dim="dim1", coord_handling="drop")
    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == {"varA", "varB"}
    assert "dim2" in ds.dims
    np.testing.assert_array_equal(ds["varA"].values, [1, 2])
    np.testing.assert_array_equal(ds["varB"].values, [3, 4])
    assert "dim1" not in ds.dims  # "dim1" is dropped
    assert "other_coord" not in ds  # coord is not dropped or unstacked
    assert "another_coord" not in ds.coords  # coord is not dropped or unstacked

    # Test coord_handling="unstack"
    ds = unstack_datarray_dimension(da, dim="dim1", coord_handling="unstack")
    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == {"varA", "varB", "other_coordA", "other_coordB", "another_coordA", "another_coordB"}
    assert "dim2" in ds.dims
    np.testing.assert_array_equal(ds["varA"].values, [1, 2])
    np.testing.assert_array_equal(ds["varB"].values, [3, 4])
    np.testing.assert_array_equal(ds["other_coordA"].values, 100)
    np.testing.assert_array_equal(ds["other_coordB"].values, 200)

    assert "dim1" not in ds.dims  # "dim1" is not a dimension anymore
    assert "other_coord" not in ds.coords  # coord is not dropped or unstacked
    assert "another_coord" not in ds.coords  # coord is not dropped or unstacked
    assert ds["other_coordA"].dims == ()
    assert ds["other_coordB"].dims == ()


def test_unstack_datarray_dimension_empty_dim():
    """Test splitting a DataArray with an empty dimension."""
    da = xr.DataArray(
        np.array([]).reshape(0, 2),
        dims=("dim1", "dim2"),
        coords={"dim1": [], "dim2": [1, 2]},
        name="temperature",
    )
    ds = unstack_datarray_dimension(da, dim="dim1")
    assert isinstance(ds, xr.Dataset)
    assert ds.data_vars.keys() == set()


def test_unstack_dataset_dimension_mixed_vars():
    """Test splitting a Dataset where some variables have the target dimension."""
    # DataArray with target dim
    da1 = xr.DataArray(
        np.array([[1, 2], [3, 4]]),
        dims=("dim1", "time"),
        coords={"dim1": ["A", "B"], "time": [100, 200]},
        name="var",
    )
    # DataArray without target dim
    da2 = xr.DataArray(
        np.array([10, 20]),
        dims=("time",),
        coords={"time": [100, 200]},
        name="other_variable",
    )
    # Create dataset
    ds = xr.Dataset({"var": da1, "other_variable": da2})

    # Test unstacking
    unstacked_ds = unstack_dataset_dimension(ds, dim="dim1", prefix="split_", suffix="_")
    assert set(unstacked_ds.data_vars) == {
        "other_variable",
        "split_var_A",
        "split_var_B",
    }
    np.testing.assert_array_equal(unstacked_ds["split_var_A"].values, [1, 2])
    np.testing.assert_array_equal(unstacked_ds["split_var_B"].values, [3, 4])
    np.testing.assert_array_equal(unstacked_ds["other_variable"].values, [10, 20])


def test_unstack_dataset_dimension_all_vars_have_dim():
    """Test splitting a Dataset where all variables have the target dimension."""
    da1 = xr.DataArray(
        np.array([[1], [2]]),
        dims=("dim1", "dim2"),
        coords={"dim1": ["A", "B"], "dim2": [50]},
        name="temperature",
    )
    da2 = xr.DataArray(
        np.array([[5], [6]]),
        dims=("dim1", "dim2"),
        coords={"dim1": ["A", "B"], "dim2": [50]},
        name="humidity",
    )
    ds = xr.Dataset({"temperature": da1, "humidity": da2})
    unstacked_ds = unstack_dataset_dimension(ds, dim="dim1")
    assert set(unstacked_ds.data_vars) == {"temperatureA", "temperatureB", "humidityA", "humidityB"}
    np.testing.assert_array_equal(unstacked_ds["temperatureA"].values, [1])
    np.testing.assert_array_equal(unstacked_ds["temperatureB"].values, [2])
    np.testing.assert_array_equal(unstacked_ds["humidityA"].values, [5])
    np.testing.assert_array_equal(unstacked_ds["humidityB"].values, [6])


def test_unstack_dataset_dimension_no_vars_have_dim():
    """Test splitting a Dataset where no variables have the target dimension."""
    da = xr.DataArray(
        np.array([1000, 1001]),
        dims=("time",),
        coords={"time": [1, 2]},
        name="pressure",
    )
    ds = xr.Dataset({"pressure": da})
    unstacked_ds = unstack_dataset_dimension(ds, dim="dim1")
    assert unstacked_ds.equals(ds)


def test_unstack_dimension_dataarray():
    """Test unstack_dimension with a DataArray input."""
    da = xr.DataArray(
        np.array([[10, 20], [30, 40]]),
        dims=("dim1", "dim2"),
        coords={"dim1": ["C", "D"], "dim2": [300, 400]},
        name="wind_speed",
    )
    ds = unstack_dimension(da, dim="dim1", prefix="split_", suffix="_d1_")
    assert isinstance(ds, xr.Dataset)
    assert set(ds.data_vars) == {"split_wind_speed_d1_C", "split_wind_speed_d1_D"}
    np.testing.assert_array_equal(ds["split_wind_speed_d1_C"].values, [10, 20])
    np.testing.assert_array_equal(ds["split_wind_speed_d1_D"].values, [30, 40])


def test_unstack_dataset_dimension_with_coord_dim():
    """Test splitting a Dataset when coordinates share the split dimension."""
    # Create a DataArray with shared coordinates
    da = xr.DataArray(
        np.array([[1, 2], [3, 4]]),
        dims=("dim1", "dim2"),
        coords={
            "dim1": ["A", "B"],
            "dim2": [10, 20],
            "other_coord": ("dim1", [100, 200]),  # Coordinate sharing 'dim1'
            "another_coord": ("dim1", [100, 200]),  # Another coordinate sharing 'dim1'
        },
        name="var",
    )

    # Wrap the DataArray into a Dataset
    ds = xr.Dataset({"var": da})

    # Test coord_handling="keep"
    ds_keep = unstack_dataset_dimension(ds, dim="dim1", coord_handling="keep")
    assert isinstance(ds_keep, xr.Dataset), "Result should be a Dataset."
    assert set(ds_keep.data_vars) == {"varA", "varB"}, "Data variables should be split correctly."
    assert "dim2" in ds_keep.dims, "'dim2' should remain as a dimension."
    np.testing.assert_array_equal(ds_keep["varA"].values, [1, 2], "varA should have correct values.")
    np.testing.assert_array_equal(ds_keep["varB"].values, [3, 4], "varB should have correct values.")
    assert "dim1" in ds_keep.dims, "'dim1' should be kept as a dimension."
    assert "other_coord" in ds_keep.coords, "'other_coord' should remain as a coordinate."
    assert "another_coord" in ds_keep.coords, "'another_coord' should remain as a coordinate."
    assert ds_keep.coords["other_coord"].dims == ("dim1",), "'other_coord' should retain its original dimension."
    assert ds_keep.coords["another_coord"].dims == ("dim1",), "'another_coord' should retain its original dimension."
    np.testing.assert_array_equal(
        ds_keep.coords["other_coord"].values,
        [100, 200],
        "'other_coord' should have correct values.",
    )
    np.testing.assert_array_equal(
        ds_keep.coords["another_coord"].values,
        [100, 200],
        "'another_coord' should have correct values.",
    )

    # Test coord_handling="drop"
    ds_drop = unstack_dataset_dimension(ds, dim="dim1", coord_handling="drop")
    assert isinstance(ds_drop, xr.Dataset), "Result should be a Dataset."
    assert set(ds_drop.data_vars) == {"varA", "varB"}, "Data variables should be split correctly."
    assert "dim2" in ds_drop.dims, "'dim2' should remain as a dimension."
    np.testing.assert_array_equal(ds_drop["varA"].values, [1, 2], "varA should have correct values.")
    np.testing.assert_array_equal(ds_drop["varB"].values, [3, 4], "varB should have correct values.")
    assert "dim1" not in ds_drop.dims, "'dim1' should be dropped from dimensions."
    assert "other_coord" not in ds_drop.coords, "'other_coord' should be dropped."
    assert "another_coord" not in ds_drop.coords, "'another_coord' should be dropped."

    # Test coord_handling="unstack"
    ds_unstack = unstack_dataset_dimension(ds, dim="dim1", coord_handling="unstack")
    assert isinstance(ds_unstack, xr.Dataset), "Result should be a Dataset."
    expected_vars = {
        "varA",
        "varB",
        "other_coordA",
        "other_coordB",
        "another_coordA",
        "another_coordB",
    }
    assert set(ds_unstack.data_vars) == expected_vars, "Data and unstacked coordinate variables should be present."
    assert "dim2" in ds_unstack.dims, "'dim2' should remain as a dimension."
    np.testing.assert_array_equal(ds_unstack["varA"].values, [1, 2], "varA should have correct values.")
    np.testing.assert_array_equal(ds_unstack["varB"].values, [3, 4], "varB should have correct values.")
    np.testing.assert_array_equal(ds_unstack["other_coordA"].values, 100)
    np.testing.assert_array_equal(ds_unstack["other_coordB"].values, 200)
    np.testing.assert_array_equal(ds_unstack["another_coordA"].values, 100)
    np.testing.assert_array_equal(ds_unstack["another_coordB"].values, 200)
    assert "dim1" not in ds_unstack.dims, "'dim1' should be removed from dimensions."
    assert "other_coord" not in ds_unstack.coords, "'other_coord' should be unstacked into variables."
    assert "another_coord" not in ds_unstack.coords, "'another_coord' should be unstacked into variables."
    assert ds_unstack["other_coordA"].dims == (), "unstacked 'other_coordA' should have no dimensions."
    assert ds_unstack["other_coordB"].dims == (), "unstacked 'other_coordB' should have no dimensions."
    assert ds_unstack["another_coordA"].dims == (), "unstacked 'another_coordA' should have no dimensions."
    assert ds_unstack["another_coordB"].dims == (), "unstacked 'another_coordB' should have no dimensions."


def test_unstack_dimension_dataset():
    """Test unstack_dimension with a Dataset input."""
    da1 = xr.DataArray(
        np.array([[7, 8], [9, 10]]),
        dims=("dim1", "time"),
        coords={"dim1": ["E", "F"], "time": [500, 600]},
        name="temperature",
    )
    da2 = xr.DataArray(
        np.array([15, 25]),
        dims=("time",),
        coords={"time": [500, 600]},
        name="humidity",
    )
    ds = xr.Dataset({"temperature": da1, "humidity": da2})
    unstacked_ds = unstack_dimension(ds, dim="dim1", prefix="split_", suffix="_d1_")
    assert set(unstacked_ds.data_vars) == {"humidity", "split_temperature_d1_E", "split_temperature_d1_F"}
    np.testing.assert_array_equal(unstacked_ds["split_temperature_d1_E"].values, [7, 8])
    np.testing.assert_array_equal(unstacked_ds["split_temperature_d1_F"].values, [9, 10])
    np.testing.assert_array_equal(unstacked_ds["humidity"].values, [15, 25])


def test_unstack_dimension_invalid_input():
    """Test unstack_dimension raises an error for invalid input types."""
    with pytest.raises(TypeError):
        unstack_dimension("not_an_xarray_object", dim="dim1")


def test_unstack_dimension_empty_dataset():
    """Test unstack_dimension with an empty Dataset."""
    ds = xr.Dataset()
    unstacked_ds = unstack_dimension(ds, dim="dim1")
    assert isinstance(unstacked_ds, xr.Dataset)
    assert len(unstacked_ds.data_vars) == 0
