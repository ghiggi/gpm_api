import numpy as np
import xarray as xr
from datatree import DataTree

from gpm_api.dataset import dimensions


def test_has_a_phony_dim():
    """Test _has_a_phony_dim"""

    array = np.zeros(shape=(3,))

    dataarray = xr.DataArray(data=array, dims=["not_phony"])
    assert not dimensions._has_a_phony_dim(dataarray)

    dataarray = xr.DataArray(data=array, dims=["phony_dim_0"])
    assert dimensions._has_a_phony_dim(dataarray)


def test_get_dataarray_dim_dict():
    """Test _get_dataarray_dim_dict"""

    array = np.zeros(shape=(3, 3))

    dataarray = xr.DataArray(data=array, dims=["phony_dim_1", "phony_dim_2"])
    dataarray.attrs["DimensionNames"] = "replaced_dim_1,replaced_dim_2"

    expected_dict = {
        "phony_dim_1": "replaced_dim_1",
        "phony_dim_2": "replaced_dim_2",
    }
    returned_dict = dimensions._get_dataarray_dim_dict(dataarray)
    assert returned_dict == expected_dict


def test_get_dataset_dim_dict():
    """Test _get_dataset_dim_dict"""

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
    returned_dict = dimensions._get_dataset_dim_dict(dataset)
    assert returned_dict == expected_dict


def test_get_datatree_dim_dict():
    """Test _get_datatree_dim_dict"""

    array_1 = np.zeros(shape=(3,))
    array_2 = np.zeros(shape=(3,))
    dataarray_1 = xr.DataArray(data=array_1, dims=["phony_dim_1"])
    dataarray_2 = xr.DataArray(data=array_2, dims=["phony_dim_2"])
    dataarray_1.attrs["DimensionNames"] = "replaced_dim_1"
    dataarray_2.attrs["DimensionNames"] = "replaced_dim_2"
    dataset_1 = xr.Dataset(data_vars={"var_1": dataarray_1})
    dataset_2 = xr.Dataset(data_vars={"var_2": dataarray_2})
    datatree = DataTree.from_dict({"dataset_1": dataset_1, "dataset_2": dataset_2})

    expected_dict = {
        "phony_dim_1": "replaced_dim_1",
        "phony_dim_2": "replaced_dim_2",
    }
    returned_dict = dimensions._get_datatree_dim_dict(datatree)
    assert returned_dict == expected_dict


def test_get_gpm_api_dims_dict(monkeypatch):
    """Test _get_gpm_api_dims_dict"""

    # Mock the replaced dimension names
    monkeypatch.setattr(
        "gpm_api.dataset.dimensions.DIM_DICT",
        {
            "name_before_1": "name_after_1",
            "name_before_2": "name_after_2",
        },
    )

    array_1 = np.zeros(shape=(3,))
    array_2 = np.zeros(shape=(3, 3))
    dataarray_1 = xr.DataArray(data=array_1, dims=["name_before_1"])
    dataarray_2 = xr.DataArray(data=array_2, dims=["name_before_2", "not_replaced"])
    dataset = xr.Dataset(data_vars={"var_1": dataarray_1, "var_2": dataarray_2})

    expected_dict = {
        "name_before_1": "name_after_1",
        "name_before_2": "name_after_2",
    }
    returned_dict = dimensions._get_gpm_api_dims_dict(dataset)
    assert returned_dict == expected_dict


def test_rename_datarray_dimensions():
    """Test _rename_datarray_dimensions"""

    array = np.zeros(shape=(3, 3))
    dataarray = xr.DataArray(data=array, dims=["phony_dim_1", "not_replaced"])
    dataarray.attrs["DimensionNames"] = "replaced_dim_1"

    returned_dataarray = dimensions._rename_datarray_dimensions(dataarray)
    expected_dims = ("replaced_dim_1", "not_replaced")
    assert returned_dataarray.dims == expected_dims


def test_rename_dataset_dimensions(monkeypatch):
    """Test _rename_dataset_dimensions"""

    # Mock the replaced dimension names
    monkeypatch.setattr(
        "gpm_api.dataset.dimensions.DIM_DICT",
        {
            "intermediate_2": "final_2",
        },
    )

    array_1 = np.zeros(shape=(3,))
    array_2 = np.zeros(shape=(3, 3))
    dataarray_1 = xr.DataArray(data=array_1, dims=["phony_dim_1"])
    dataarray_2 = xr.DataArray(data=array_2, dims=["phony_dim_2", "not_replaced"])
    dataarray_1.attrs["DimensionNames"] = "final_1"
    dataarray_2.attrs["DimensionNames"] = "intermediate_2"
    dataset = xr.Dataset(data_vars={"var_1": dataarray_1, "var_2": dataarray_2})

    # With use_api_defaults=True, which replaces intermediate_2 with final_2
    returned_dataset = dimensions._rename_dataset_dimensions(dataset)
    expected_dims = ["final_1", "final_2", "not_replaced"]
    assert list(returned_dataset.dims) == expected_dims

    # With use_api_defaults=False
    returned_dataset = dimensions._rename_dataset_dimensions(dataset, use_api_defaults=False)
    expected_dims = ["final_1", "intermediate_2", "not_replaced"]
    assert list(returned_dataset.dims) == expected_dims


def test_rename_datatree_dimensions(monkeypatch):
    """Test _rename_datatree_dimensions"""

    # Mock the replaced dimension names
    monkeypatch.setattr(
        "gpm_api.dataset.dimensions.DIM_DICT",
        {
            "intermediate_2": "final_2",
        },
    )

    array_1 = np.zeros(shape=(3,))
    array_2 = np.zeros(shape=(3, 3))
    dataarray_1 = xr.DataArray(data=array_1, dims=["phony_dim_1"])
    dataarray_2 = xr.DataArray(data=array_2, dims=["phony_dim_2", "not_replaced"])
    dataarray_1.attrs["DimensionNames"] = "final_1"
    dataarray_2.attrs["DimensionNames"] = "intermediate_2"
    dataset_1 = xr.Dataset(data_vars={"var_1": dataarray_1})
    dataset_2 = xr.Dataset(data_vars={"var_2": dataarray_2})
    datatree = DataTree.from_dict({"dataset_1": dataset_1, "dataset_2": dataset_2})

    # With use_api_defaults=True, which replaces intermediate_2 with final_2
    returned_datatree = dimensions._rename_datatree_dimensions(datatree)
    assert list(returned_datatree["dataset_1"].dims) == ["final_1"]
    assert list(returned_datatree["dataset_2"].dims) == ["final_2", "not_replaced"]

    # With use_api_defaults=False
    returned_datatree = dimensions._rename_datatree_dimensions(datatree, use_api_defaults=False)
    assert list(returned_datatree["dataset_1"].dims) == ["final_1"]
    assert list(returned_datatree["dataset_2"].dims) == ["intermediate_2", "not_replaced"]
