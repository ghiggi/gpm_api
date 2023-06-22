#!/usr/bin/env python3
"""
Created on Thu Jun 22 15:09:26 2023

@author: ghiggi
"""
import numpy as np
import xarray as xr

from gpm_api.dataset.dimensions import assign_dataset_dimensions
from gpm_api.utils.utils_HDF5 import hdf5_datasets, hdf5_groups


def _check_valid_variables(variables, dataset_variables):
    """Check valid variables."""
    idx_subset = np.where(np.isin(variables, dataset_variables, invert=True))[0]
    if len(idx_subset) > 0:
        wrong_variables = variables[idx_subset]
        raise ValueError(f"The following variables are not available: {wrong_variables}.")
    return variables


def _check_valid_groups(hdf, scan_mode, groups):
    """Check valid groups."""
    dataset_groups = np.unique(list(hdf5_groups(hdf[scan_mode])) + [""])
    idx_subset = np.where(np.isin(groups, dataset_groups, invert=True))[0]
    if len(idx_subset) > 0:
        wrong_groups = groups[idx_subset]
        raise ValueError(f"The following groups are not available: {wrong_groups}.")
    return groups


def _get_variable_group_dict(hdf):
    """Get variables: group dictionary."""
    # Get dataset variables
    # - scan_mode
    dataset_dict = hdf5_datasets(hdf)
    var_group_dict = {}
    for k in dataset_dict:
        if k.find("/") != -1:
            list_split = k.split("/")
            if len(list_split) == 3:  # scan_mode/group/var
                var_group_dict[list_split[2]] = list_split[1]
            if len(list_split) == 2:  # scan_mode/var
                var_group_dict[list_split[1]] = ""  # to get variables from scan_mode group
    return var_group_dict


def _get_all_groups(hdf, scan_mode):
    """Get all groups within a scan_mode."""
    groups = np.unique(list(hdf5_groups(hdf[scan_mode])) + [""])
    return groups


def _get_hdf_groups(hdf, scan_mode, variables=None, groups=None):
    """Get groups names that contains the variables of interest.

    If variables and groups is None, return all groups.
    If variables is specified, return the groups containing the variables.
    If groups is specified, check groups validity.
    """
    # Select only groups containing the specified variables
    if variables is not None:
        # Get dataset variables
        var_group_dict = _get_variable_group_dict(hdf)
        dataset_variables = list(var_group_dict)
        # Include 'height' variable if available (for radar)
        if "height" in dataset_variables:
            variables = np.unique(np.append(variables, ["height"]))
        # Check variables validity
        variables = _check_valid_variables(variables, dataset_variables)
        # Get groups subset
        groups = np.unique([var_group_dict[var] for var in variables])

    # Check groups validity
    elif groups is not None:
        groups = _check_valid_groups(hdf, scan_mode, groups)

    # Select all groups
    else:
        groups = _get_all_groups(hdf, scan_mode)

    # Remove "ScanTime" from groups
    groups = np.setdiff1d(groups, ["ScanTime"])

    # Return groups
    return groups, variables


def _remove_dummy_variables(ds):
    """Return dummy variables from HDF dataset group."""
    dummy_variables = [
        "Latitude",
        "Longitude",
        # IMERG bnds
        "time_bnds",
        "lat_bnds",
        "lon_bnds",
    ]
    dummy_variables = np.array(dummy_variables)
    variables_to_drop = dummy_variables[np.isin(dummy_variables, list(ds.data_vars))]
    ds = ds.drop_vars(variables_to_drop)
    return ds


def _prefix_dataset_group_variables(ds, group, prefix_group):
    """Prefix group dataset variables."""
    if prefix_group and len(group) > 0:
        rename_var_dict = {var: group + "/" + var for var in ds.data_vars}
        ds = ds.rename_vars(rename_var_dict)
    return ds


def _preprocess_hdf_group(ds, variables, group, prefix_group):
    """Preprocess dataset group read with xarray."""
    # Assign dimensions
    ds = assign_dataset_dimensions(ds)

    # Subset variables
    if variables is not None:
        variables_subset = variables[np.isin(variables, list(ds.data_vars))]
        ds = ds[variables_subset]

    # Remove unuseful variables
    ds = _remove_dummy_variables(ds)

    # Prefix variables with group name
    ds = _prefix_dataset_group_variables(ds, group, prefix_group)

    return ds


def _open_hdf_group(
    filepath,
    scan_mode,
    group,
    variables=None,
    prefix_group=True,
    decode_cf=False,
    chunks=None,
    engine="netcdf4",
):
    # Define hdf group
    # - group == '' to retrieve variables attached to scan_mode group
    hdf_group = scan_mode + "/" + group

    # If chunks is None, read in memory and close connection
    # - But read with chunks="auto" to read just variables of interest !
    if chunks is None:
        with xr.open_dataset(
            filepath, engine=engine, mode="r", group=hdf_group, decode_cf=decode_cf, chunks="auto"
        ) as ds:
            ds = _preprocess_hdf_group(
                ds=ds, variables=variables, group=group, prefix_group=prefix_group
            )
            ds = ds.compute()
    else:
        # Else keep connection open to lazy file on disk
        ds = xr.open_dataset(
            filepath,
            engine=engine,
            mode="r",
            group=hdf_group,
            decode_cf=decode_cf,
            chunks=chunks,
        )
        ds = _preprocess_hdf_group(
            ds=ds, variables=variables, group=group, prefix_group=prefix_group
        )
    return ds
