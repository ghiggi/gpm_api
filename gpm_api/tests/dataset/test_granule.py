import numpy as np
import xarray as xr
from datatree import DataTree

from gpm_api.dataset import granule


# Tests for public functions ###################################################


def test_get_variables():
    """Test get_variables"""

    da = xr.DataArray()
    dataset = xr.Dataset(data_vars={"var_1": da, "var_2": da})
    expected_variables = ["var_1", "var_2"]
    returned_variables = granule.get_variables(dataset)
    assert returned_variables == expected_variables


def test_get_variables_dims():
    """Test get_variables_dims"""

    array_1 = np.zeros(shape=(3, 3))
    array_2 = np.zeros(shape=(3, 3))
    dataarray_1 = xr.DataArray(array_1, dims=["dim_1", "dim_2"])
    dataarray_2 = xr.DataArray(array_2, dims=["dim_2", "dim_3"])
    dataset = xr.Dataset(data_vars={"var_1": dataarray_1, "var_2": dataarray_2})

    expected_variables_dims = ["dim_1", "dim_2", "dim_3"]
    returned_variables_dims = granule.get_variables_dims(dataset)
    assert np.array_equal(returned_variables_dims, expected_variables_dims)

    # Check dataset with no variables
    dataset = xr.Dataset()
    expected_variables_dims = []
    returned_variables_dims = granule.get_variables_dims(dataset)
    assert returned_variables_dims == expected_variables_dims


def test_unused_var_dims_and_remove():
    """Test unused_var_dims and remove_unused_var_dims"""

    array = np.zeros(shape=(3,))
    dataarray = xr.DataArray(array, dims=["used_dim"])
    dataset = xr.Dataset(data_vars={"var": dataarray})
    dataset = dataset.expand_dims(dim=["unused_dim"])
    # TODO: this does not work, it adds the dimension to all variables. How can this be done?

    # # Check list of unused dimensions
    # expected_unused_dims = ["unused_dim"]
    # returned_unused_dims = granule.unused_var_dims(dataset)
    # assert returned_unused_dims == expected_unused_dims

    # # Remove unused dimensions
    # returned_dataset = granule.remove_unused_var_dims(dataset)
    # expected_dims = ["used_dim"]
    # assert list(returned_dataset.dims) == expected_dims


# Tests for internal functions #################################################


def test_prefix_dataset_group_variables():
    """Test _prefix_dataset_group_variables"""

    da = xr.DataArray()
    dataset = xr.Dataset(data_vars={"var_1": da, "var_2": da})
    group = "group_1"

    expected_data_vars = ["group_1/var_1", "group_1/var_2"]
    returned_dataset = granule._prefix_dataset_group_variables(dataset, group)
    assert isinstance(returned_dataset, xr.Dataset)
    assert list(returned_dataset.data_vars) == expected_data_vars


def test_remove_dummy_variables():
    """Test _remove_dummy_variables"""

    da = xr.DataArray()
    dataset = xr.Dataset(
        data_vars={
            # Dummy variables
            "Latitude": da,
            "Longitude": da,
            "time_bnds": da,
            "lat_bnds": da,
            "lon_bnds": da,
            # Real variables
            "real_var_1": da,
        }
    )

    expected_data_vars = ["real_var_1"]
    returned_dataset = granule._remove_dummy_variables(dataset)
    assert list(returned_dataset.data_vars) == expected_data_vars


def test_subset_dataset_variables():
    """Test _subset_dataset_variables"""

    da = xr.DataArray()
    dataset = xr.Dataset(data_vars={"var_1": da, "var_2": da})

    # Subset variables
    variables = ["var_1"]
    returned_dataset = granule._subset_dataset_variables(dataset, variables)
    assert isinstance(returned_dataset, xr.Dataset)
    assert list(returned_dataset.data_vars) == variables

    # Variables not in dataset
    variables = ["var_1", "var_3"]
    expected_data_vars = ["var_1"]
    returned_dataset = granule._subset_dataset_variables(dataset, variables)
    assert list(returned_dataset.data_vars) == expected_data_vars

    # No variables
    variables = []
    returned_dataset = granule._subset_dataset_variables(dataset, variables)
    assert list(returned_dataset.data_vars) == variables

    # With variables None: return all variables
    expected_data_vars = ["var_1", "var_2"]
    returned_dataset = granule._subset_dataset_variables(dataset, None)
    assert list(returned_dataset.data_vars) == expected_data_vars


def test_process_group_dataset():
    """Test _process_group_dataset"""

    da = xr.DataArray()
    dataset = xr.Dataset(
        data_vars={
            # Dummy variable
            "Latitude": da,
            # Kept variable
            "var_1": da,
            # Removed variable
            "var_2": da,
        }
    )
    variables = ["Latitude", "var_1"]
    group = "group_1"

    expected_data_vars = ["var_1"]
    returned_dataset = granule._process_group_dataset(dataset, group, variables)
    assert isinstance(returned_dataset, xr.Dataset)
    assert list(returned_dataset.data_vars) == expected_data_vars

    # Prefix group variables
    expected_data_vars = ["group_1/var_1"]
    returned_dataset = granule._process_group_dataset(dataset, group, variables, prefix_group=True)
    assert list(returned_dataset.data_vars) == expected_data_vars


def test_get_scan_mode_info():  # TODO
    """Test _get_scan_mode_info"""


def test_get_flattened_scan_mode_dataset():  # TODO
    """Test _get_flattened_scan_mode_dataset"""

    da = xr.DataArray()

    # Build source datatree
    scan_mode = "scan_mode"
    dt = DataTree.from_dict(
        {
            scan_mode: DataTree.from_dict(
                {
                    "group_1": DataTree(),
                    "group_2": DataTree(),
                }
            ),
        }
    )
    dt[scan_mode]["group_1"]["var_1"] = da
    dt[scan_mode]["group_2"]["var_2"] = da
    dt[scan_mode]["var_3"] = da

    # Without specifying group: empty dataset
    group = []
    expected_data_vars = []
    returned_dataset = granule._get_flattened_scan_mode_dataset(dt, scan_mode, group)
    assert isinstance(returned_dataset, xr.Dataset)
    assert list(returned_dataset.data_vars) == expected_data_vars

    # Group same as scan_mode: return only top-level variables of scan_mode
    group = [scan_mode]
    expected_data_vars = ["var_3"]
    returned_dataset = granule._get_flattened_scan_mode_dataset(dt, scan_mode, group)
    assert list(returned_dataset.data_vars) == expected_data_vars

    # Specifying sub-groups: return variables of sub-groups
    group = ["group_1", "group_2"]
    expected_data_vars = ["var_1", "var_2"]
    returned_dataset = granule._get_flattened_scan_mode_dataset(dt, scan_mode, group)
    assert list(returned_dataset.data_vars) == expected_data_vars

    # Check variables filtering and group prefixing
    group = ["group_1", "group_2"]
    variables = ["var_1"]
    expected_data_vars = ["group_1/var_1"]
    returned_dataset = granule._get_flattened_scan_mode_dataset(
        dt, scan_mode, group, variables=variables, prefix_group=True
    )
    assert list(returned_dataset.data_vars) == expected_data_vars


def test_get_scan_mode_dataset():  # TODO
    """Test _get_scan_mode_dataset"""
