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
"""This module test the retrieval of GPM files groups and variables."""

import pytest
import xarray as xr
from datatree import DataTree

from gpm.dataset import groups_variables


def test_get_available_groups():
    """Test _get_available_groups."""
    scan_mode = "S1"
    other_scan_mode = "S2"
    dt = DataTree.from_dict(
        {
            scan_mode: DataTree.from_dict(
                {
                    "group_1": DataTree(),
                    "group_2": DataTree(),
                },
            ),
            other_scan_mode: DataTree(),
        },
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
    """Test _get_available_variables."""
    da = xr.DataArray()
    scan_mode = "S1"
    other_scan_mode = "S2"
    dt = DataTree.from_dict(
        {
            scan_mode: DataTree.from_dict(
                {
                    "group_1": DataTree(),
                },
            ),
            other_scan_mode: DataTree(),
        },
    )
    dt[scan_mode]["var_1"] = da
    dt[scan_mode]["group_1"]["var_1"] = da  # var_1 repeated on purpose
    dt[scan_mode]["group_1"]["var_2"] = da
    dt[other_scan_mode]["var_3"] = da

    expected_variables = ["var_1", "var_2"]
    returned_variables = groups_variables._get_available_variables(dt, scan_mode)
    assert expected_variables == returned_variables


def test_get_relevant_groups_variables(monkeypatch):
    """Test _get_relevant_groups_variables."""
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
                },
            ),
        },
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
        dt,
        scan_mode,
    )
    assert expected_groups == returned_groups
    assert expected_variables == returned_variables

    # Check group requested: return groups
    input_groups = ["group_2"]
    expected_groups = ["group_2"]
    expected_variables = None
    returned_groups, returned_variables = groups_variables._get_relevant_groups_variables(
        dt,
        scan_mode,
        groups=input_groups,
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
        dt,
        scan_mode,
        variables=input_variables,
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
        dt,
        scan_mode,
        groups=input_groups,
        variables=input_variables,
    )
    assert expected_groups == returned_groups
    assert expected_variables == returned_variables
