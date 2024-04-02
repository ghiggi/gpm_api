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
"""This module contains functions to read GPM file groups, sub-groups and variables."""
import numpy as np

from gpm.utils.list import flatten_list

WISHED_COORDS = ["height", "SCorientation", "dataQuality"]


def _get_groups_path(dt):
    """Return the group path."""
    return list(dt.groups)


def _get_groups_names(dt):
    """Return the groups name."""
    group_path = _get_groups_path(dt)
    return [path.split("/")[-1] for path in group_path]


def _get_variables(dt):
    """Return the group variables."""
    return list(dt.data_vars)


def _get_available_scan_modes(dt):
    return list(dt.children)


def _get_available_groups(dt, scan_mode, name=False):
    """Return available groups in a scan modes.

    If False (the default), return the group path.
    If True, return the group name.
    The scan_mode group is also returned as '' !
    """
    if name:
        groups = _get_groups_names(dt[scan_mode])
        groups = [group if group != f"{scan_mode}" else "" for group in groups]
    else:
        groups = _get_groups_path(dt[scan_mode])
        # groups = [group for group in groups if group != f"/{scan_mode}"]

    return groups


def _get_variables_scan_mode(dt, scan_mode, group):
    """Return variables associated to a scan_mode group."""
    return _get_variables(dt[scan_mode]) if scan_mode == group else _get_variables(dt[scan_mode][group])


def _get_variables_path_dict(dt, scan_mode):
    """Return the path to access each variable.

    The variables associated to the <scan_mode> node are also returned !
    The <scan_mode> is node is represented by ''.
    """
    return {
        var: group
        for group in _get_available_groups(dt, scan_mode, name=False)
        for var in _get_variables(dt[scan_mode][group])
    }


def _get_variables_group_dict(dt, scan_mode):
    """Return the group of each variable.

    The variables associated to the <scan_mode> node are also returned !
    The <scan_mode> is node is represented by ''.
    """
    return {
        var: group
        for group in _get_available_groups(dt, scan_mode, name=True)
        for var in _get_variables(dt[scan_mode][group])
    }


def _get_group_variables_dict(dt, scan_mode, name=True):
    """Return a dictionary with the list of variables for each group."""
    if name:
        dict_group = {
            path.split("/")[-1]: _get_variables(dt[scan_mode][path])
            for path in _get_available_groups(dt, scan_mode, name=False)
        }
    else:
        dict_group = {
            path: _get_variables(dt[scan_mode][path]) for path in _get_available_groups(dt, scan_mode, name=False)
        }
    return dict_group


def _get_available_variables(dt, scan_mode):
    """Return available variables."""
    dict_vars = _get_variables_path_dict(dt, scan_mode)
    return list(dict_vars.keys())


def _get_availables_variables_in_groups(dt, scan_mode, groups):
    """Return available variables in specific groups."""
    dict_group = _get_group_variables_dict(dt, scan_mode, name=True)
    list_variables = [dict_group[group] for group in groups]
    return flatten_list(list_variables)


def _check_valid_variables(variables, dataset_variables):
    """Check valid variables."""
    idx_subset = np.where(np.isin(variables, dataset_variables, invert=True))[0]
    if len(idx_subset) > 0:
        wrong_variables = np.array(variables)[idx_subset].tolist()
        raise ValueError(f"The following variables are not available: {wrong_variables}.")
    return variables


def _check_valid_groups(groups, available_groups):
    """Check valid groups."""
    idx_subset = np.where(np.isin(groups, available_groups, invert=True))[0]
    if len(idx_subset) > 0:
        wrong_groups = np.array(groups)[idx_subset].tolist()
        raise ValueError(f"The following groups are not available: {wrong_groups}.")
    return groups


def _add_mandatory_variables(variables, available_variables):
    """Add wished coordinates to 'variables'.

    Currently it includes:
    - 'height' variable if available (for radar products)
    - 'dataQuality' variable
    - 'SCorientation' variable
    """
    wished_variables = [var for var in WISHED_COORDS if var in available_variables]
    if len(wished_variables) >= 1:
        variables = np.unique(np.append(variables, wished_variables)).tolist()
    return variables


def _get_relevant_groups_variables(dt, scan_mode, variables=None, groups=None):
    """Get groups names that contains the variables of interest.

    If variables and groups is None, return all groups.
    If only groups is specified, gpm_api will select all variables for such groups.
    If only variables is specified, gpm_api selects all variables specified.
    If groups and variables are specified, it selects all variables of the specified 'groups'
    and the variables specified in 'variables'.
    """
    available_groups = _get_available_groups(dt, scan_mode, name=True)
    available_variables = _get_available_variables(dt, scan_mode)
    if variables is not None:
        if isinstance(variables, np.ndarray):
            variables = variables.tolist()
        # Add mandatory variables
        variables = _add_mandatory_variables(variables, available_variables)

        # Check variables validity
        variables = _check_valid_variables(variables, available_variables)
        # Get groups subset
        var_group_dict = _get_variables_group_dict(dt, scan_mode)
        required_groups = np.unique([var_group_dict[var] for var in variables]).tolist()

    if groups is not None:
        if isinstance(groups, np.ndarray):
            groups = groups.tolist()
        groups = _check_valid_groups(groups, available_groups)

    # Identify input combination
    if variables is not None and groups is not None:
        groups_variables = _get_availables_variables_in_groups(dt, scan_mode, groups)
        variables = np.unique(groups_variables + variables).tolist()
        groups = np.unique(groups + required_groups).tolist()
    elif variables is not None and groups is None:
        groups = required_groups
    elif variables is None and groups is None:
        groups = available_groups
        # variables = available_groups # variable=None
    else:  # groups is not None and variable is None
        # variables = _get_availables_variables_in_groups(dt, scan_mode, groups) # variable=None
        pass

    # Remove "ScanTime" from groups
    groups = np.setdiff1d(groups, ["ScanTime"]).tolist()

    # Return groups
    return groups, variables
