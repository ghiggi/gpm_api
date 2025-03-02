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
"""This module test the GPM-API Dataset Dimensions."""

import numpy as np
import xarray as xr

from gpm.dataset.dimensions import (
    DIM_DICT,
    _get_dataarray_dim_dict,
    _get_dataset_dim_dict,
    _get_datatree_dim_dict,
    _get_gpm_api_dims_dict,
    _has_a_phony_dim,
    rename_dataarray_dimensions,
    rename_dataset_dimensions,
    rename_datatree_dimensions,
)


def test_has_a_phony_dim():
    """Test _has_a_phony_dim."""
    array = np.zeros(shape=(3,))

    dataarray = xr.DataArray(data=array, dims=["not_phony"])
    assert not _has_a_phony_dim(dataarray)

    dataarray = xr.DataArray(data=array, dims=["phony_dim_0"])
    assert _has_a_phony_dim(dataarray)


def test_get_dataarray_dim_dict():
    """Test _get_dataarray_dim_dict."""
    array = np.zeros(shape=(3, 3))

    dataarray = xr.DataArray(data=array, dims=["phony_dim_1", "phony_dim_2"])
    dataarray.attrs["DimensionNames"] = "replaced_dim_1,replaced_dim_2"

    expected_dict = {
        "phony_dim_1": "replaced_dim_1",
        "phony_dim_2": "replaced_dim_2",
    }
    returned_dict = _get_dataarray_dim_dict(dataarray)
    assert returned_dict == expected_dict


def test_get_dataset_dim_dict():
    """Test _get_dataset_dim_dict."""
    array_1 = np.zeros(shape=(3,))
    array_2 = np.zeros(shape=(3,))
    dataarray_1 = xr.DataArray(data=array_1, dims=["phony_dim_1"])
    dataarray_2 = xr.DataArray(data=array_2, dims=["phony_dim_2"])
    dataarray_1.attrs["DimensionNames"] = "replaced_dim_1"
    dataarray_2.attrs["DimensionNames"] = "replaced_dim_2"
    dataset = xr.Dataset(data_vars={"var_1": dataarray_1, "var_2": dataarray_2})

    expected_dict = {
        "phony_dim_1": "replaced_dim_1",
        "phony_dim_2": "replaced_dim_2",
    }
    returned_dict = _get_dataset_dim_dict(dataset)
    assert returned_dict == expected_dict


def test_get_datatree_dim_dict():
    """Test _get_datatree_dim_dict."""
    array_1 = np.zeros(shape=(3,))
    array_2 = np.zeros(shape=(3,))
    dataarray_1 = xr.DataArray(data=array_1, dims=["phony_dim_1"])
    dataarray_2 = xr.DataArray(data=array_2, dims=["phony_dim_2"])
    dataarray_1.attrs["DimensionNames"] = "replaced_dim_1"
    dataarray_2.attrs["DimensionNames"] = "replaced_dim_2"
    dataset_1 = xr.Dataset(data_vars={"var_1": dataarray_1})
    dataset_2 = xr.Dataset(data_vars={"var_2": dataarray_2})
    datatree = xr.DataTree.from_dict({"dataset_1": dataset_1, "dataset_2": dataset_2})

    expected_dict = {
        "phony_dim_1": "replaced_dim_1",
        "phony_dim_2": "replaced_dim_2",
    }
    returned_dict = _get_datatree_dim_dict(datatree)
    assert returned_dict == expected_dict


def test_get_gpm_dims_dict(monkeypatch):
    """Test _get_gpm_dims_dict."""
    expected_dict = {
        "npixel": "cross_track",
        "nchannel1": "pmw_frequency",
    }

    dim1 = "npixel"
    dim2 = "nchannel1"
    array_1 = np.zeros(shape=(3,))
    array_2 = np.zeros(shape=(3, 3))
    dataarray_1 = xr.DataArray(data=array_1, dims=[dim1])
    dataarray_2 = xr.DataArray(data=array_2, dims=[dim2, "not_replaced"])
    dataset = xr.Dataset(data_vars={"var_1": dataarray_1, "var_2": dataarray_2})
    returned_dict = _get_gpm_api_dims_dict(dataset)
    assert "not_replaced" not in returned_dict
    assert returned_dict == expected_dict


def test_rename_dataarray_dimensions():
    """Test rename_dataarray_dimensions."""
    array = np.zeros(shape=(3, 3))
    dataarray = xr.DataArray(data=array, dims=["phony_dim_1", "not_replaced"])
    dataarray.attrs["DimensionNames"] = "replaced_dim_1"

    returned_dataarray = rename_dataarray_dimensions(dataarray)
    expected_dims = ("replaced_dim_1", "not_replaced")
    assert returned_dataarray.dims == expected_dims


def test_rename_dataset_dimensions(monkeypatch):
    """Test rename_dataset_dimensions."""
    dim2 = "nchannel1"
    expected_dim = DIM_DICT[dim2]

    array_1 = np.zeros(shape=(3,))
    array_2 = np.zeros(shape=(3, 3, 2))
    dataarray_1 = xr.DataArray(data=array_1, dims=["phony_dim_1"])
    dataarray_2 = xr.DataArray(data=array_2, dims=["phony_dim_2", "replaced", "not_replaced"])
    dataarray_1.attrs["DimensionNames"] = "final_1"
    dataarray_2.attrs["DimensionNames"] = f"{dim2},replaced_with_dummy,"
    dataset = xr.Dataset(data_vars={"var_1": dataarray_1, "var_2": dataarray_2})

    # With use_api_defaults=True, which replaces intermediate_2 with final_2
    returned_dataset = rename_dataset_dimensions(dataset)
    expected_dims = ["final_1", expected_dim, "replaced_with_dummy", "not_replaced"]
    assert list(returned_dataset.dims) == expected_dims

    # With use_api_defaults=False
    returned_dataset = rename_dataset_dimensions(dataset, use_api_defaults=False)
    expected_dims = ["final_1", dim2, "replaced_with_dummy", "not_replaced"]
    assert list(returned_dataset.dims) == expected_dims


def test_rename_datatree_dimensions(monkeypatch):
    """Test rename_datatree_dimensions."""
    dim2 = "nchannel1"
    expected_dim = DIM_DICT[dim2]

    array_1 = np.zeros(shape=(3,))
    array_2 = np.zeros(shape=(3, 3, 2))
    dataarray_1 = xr.DataArray(data=array_1, dims=["phony_dim_1"])
    dataarray_2 = xr.DataArray(data=array_2, dims=["phony_dim_2", "replaced", "not_replaced"])
    dataarray_1.attrs["DimensionNames"] = "final_1"
    dataarray_2.attrs["DimensionNames"] = f"{dim2},replaced_with_dummy,"
    dataset_1 = xr.Dataset(data_vars={"var_1": dataarray_1})
    dataset_2 = xr.Dataset(data_vars={"var_2": dataarray_2})
    datatree = xr.DataTree.from_dict({"dataset_1": dataset_1, "dataset_2": dataset_2})

    # With use_api_defaults=True, which replaces intermediate_2 with final_2
    returned_datatree = rename_datatree_dimensions(datatree)
    assert list(returned_datatree["dataset_1"].dims) == ["final_1"]
    assert list(returned_datatree["dataset_2"].dims) == [expected_dim, "replaced_with_dummy", "not_replaced"]

    # With use_api_defaults=False
    returned_datatree = rename_datatree_dimensions(datatree, use_api_defaults=False)
    assert list(returned_datatree["dataset_1"].dims) == ["final_1"]
    assert list(returned_datatree["dataset_2"].dims) == [dim2, "replaced_with_dummy", "not_replaced"]
