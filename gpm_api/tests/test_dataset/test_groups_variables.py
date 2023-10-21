import pytest
import xarray as xr
from datatree import DataTree

from gpm_api.dataset import groups_variables


def test_get_available_groups():
    """Test _get_available_groups"""

    scan_mode = "S1"
    other_scan_mode = "S2"
    dt = DataTree.from_dict(
        {
            scan_mode: DataTree.from_dict(
                {
                    "group_1": DataTree(),
                    "group_2": DataTree(),
                }
            ),
            other_scan_mode: DataTree(),
        }
    )

    # Test with full group path
    expected_groups = [f"/{scan_mode}", f"/{scan_mode}/group_1", f"/{scan_mode}/group_2"]
    returned_groups = groups_variables._get_available_groups(dt, scan_mode, name=False)
    assert expected_groups == returned_groups

    # Test with group name only
    expected_groups = ["", "group_1", "group_2"]
    returned_groups = groups_variables._get_available_groups(dt, scan_mode, name=True)
    assert expected_groups == returned_groups


def test_get_available_variables():
    """Test _get_available_variables"""

    da = xr.DataArray()
    scan_mode = "S1"
    other_scan_mode = "S2"
    dt = DataTree.from_dict(
        {
            scan_mode: DataTree.from_dict(
                {
                    "group_1": DataTree(),
                }
            ),
            other_scan_mode: DataTree(),
        }
    )
    dt[scan_mode]["var_1"] = da
    dt[scan_mode]["group_1"]["var_1"] = da  # var_1 repeated on purpose
    dt[scan_mode]["group_1"]["var_2"] = da
    dt[other_scan_mode]["var_3"] = da

    expected_variables = ["var_1", "var_2"]
    returned_variables = groups_variables._get_available_variables(dt, scan_mode)
    assert expected_variables == returned_variables


def test_get_relevant_groups_variables(monkeypatch):
    """Test _get_relevant_groups_variables"""

    # Mock mandatory variables
    monkeypatch.setattr(groups_variables, "WISHED_COORDS", ["mandatory_var", "mandatory_var_2"])

    da = xr.DataArray()
    scan_mode = "S1"
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
    dt[scan_mode]["group_1"]["mandatory_var"] = da
    dt[scan_mode]["group_1"]["var_1"] = da
    dt[scan_mode]["group_1"]["var_2"] = da
    dt[scan_mode]["group_2"]["var_3"] = da
    dt[scan_mode]["group_2"]["var_4"] = da

    # Check no variable or group requested: return all groups
    expected_groups = ["", "group_1", "group_2"]
    expected_variables = None
    returned_groups, returned_variables = groups_variables._get_relevant_groups_variables(
        dt, scan_mode
    )
    assert expected_groups == returned_groups
    assert expected_variables == returned_variables

    # Check group requested: return groups
    input_groups = ["group_2"]
    expected_groups = ["group_2"]
    expected_variables = None
    returned_groups, returned_variables = groups_variables._get_relevant_groups_variables(
        dt, scan_mode, groups=input_groups
    )
    assert expected_groups == returned_groups
    assert expected_variables == returned_variables

    # Check invalid groups requested
    input_groups = ["group_3"]
    with pytest.raises(ValueError):
        groups_variables._get_relevant_groups_variables(dt, scan_mode, groups=input_groups)

    # Check variable requested: return variables (plus mandatory) and groups containing them
    input_variables = ["var_3"]
    expected_groups = ["group_1", "group_2"]
    expected_variables = ["mandatory_var", "var_3"]
    returned_groups, returned_variables = groups_variables._get_relevant_groups_variables(
        dt, scan_mode, variables=input_variables
    )
    assert expected_groups == returned_groups
    assert expected_variables == returned_variables

    # Check invalid variables requested
    input_variables = ["var_5"]
    with pytest.raises(ValueError):
        groups_variables._get_relevant_groups_variables(dt, scan_mode, variables=input_variables)

    # Check group and variable requested: return requested variables plus variables in requested groups
    input_groups = ["group_2"]
    input_variables = ["var_1"]
    expected_groups = ["group_1", "group_2"]
    expected_variables = ["mandatory_var", "var_1", "var_3", "var_4"]
    returned_groups, returned_variables = groups_variables._get_relevant_groups_variables(
        dt, scan_mode, groups=input_groups, variables=input_variables
    )
    assert expected_groups == returned_groups
    assert expected_variables == returned_variables
