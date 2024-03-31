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
"""This module contains functions to read a single file into a GPM-API Dataset."""
import warnings

import numpy as np
import xarray as xr

from gpm.dataset.attrs import get_granule_attrs
from gpm.dataset.conventions import finalize_dataset
from gpm.dataset.coords import get_coords
from gpm.dataset.groups_variables import _get_relevant_groups_variables
from gpm.io.checks import (
    check_groups,
    check_scan_mode,
    check_variables,
)
from gpm.io.info import get_product_from_filepath, get_version_from_filepath


def _prefix_dataset_group_variables(ds, group):
    """Prefix group dataset variables."""
    var_dict = {var: group + "/" + var for var in ds.data_vars}
    return ds.rename_vars(var_dict)


def _remove_dummy_variables(ds):
    """Remove dummy variables from HDF dataset group."""
    dummy_variables = [
        "Latitude",
        "Longitude",
        "time_bnds",  # added with coords dictionary !
    ]
    dummy_variables = np.array(dummy_variables)
    variables_to_drop = dummy_variables[np.isin(dummy_variables, list(ds.data_vars))]
    return ds.drop_vars(variables_to_drop)


def _subset_dataset_variables(ds, variables):
    """Selectr xr.Dataset variables included in the variables list.

    'variables' can contain variables not present in the xr.Dataset.
    If variables=None, does not subset the xr.Dataset.
    """
    if variables is not None:
        variables_subset = np.array(variables)[np.isin(variables, list(ds.data_vars))].tolist()
        ds = ds[variables_subset]
    return ds


def _process_group_dataset(ds, group, variables, prefix_group=False):
    """Subset group dataset and change variable names if asked."""
    ds = _subset_dataset_variables(ds, variables)
    ds = _remove_dummy_variables(ds)
    if prefix_group:
        ds = _prefix_dataset_group_variables(ds, group)
    return ds


def _get_scan_mode_info(dt, scan_mode, variables, groups):
    """Retrieve coordinates, attributes and valid variables and groups."""
    # Get global attributes from the root
    attrs = get_granule_attrs(dt)
    attrs["ScanMode"] = scan_mode

    # Get coordinates
    coords = get_coords(dt, scan_mode)

    # Get groups to process (filtering out groups without any `variables`)
    groups, variables = _get_relevant_groups_variables(
        dt,
        scan_mode=scan_mode,
        variables=variables,
        groups=groups,
    )
    return (coords, attrs, groups, variables)


def _get_flattened_scan_mode_dataset(dt, scan_mode, groups, variables=None, prefix_group=False):
    """Retrieve scan mode dataset."""
    list_ds = []
    for group in groups:
        if group == scan_mode:
            ds = dt[scan_mode].to_dataset()
            group = ""
        else:
            ds = dt[scan_mode][group].to_dataset()
        ds = _process_group_dataset(ds, group, variables, prefix_group=prefix_group)
        list_ds.append(ds)
    return xr.merge(list_ds)


def _get_scan_mode_dataset(
    dt,
    scan_mode,
    variables=None,
    groups=None,
    prefix_group=False,
):
    """Retrieve scan mode xr.Dataset."""
    # Retrieve granule info
    coords, attrs, groups, variables = _get_scan_mode_info(
        dt=dt,
        scan_mode=scan_mode,
        variables=variables,
        groups=groups,
    )

    # Create flattened dataset for a specific scan_mode
    ds = _get_flattened_scan_mode_dataset(
        dt,
        scan_mode=scan_mode,
        groups=groups,
        variables=variables,
        prefix_group=prefix_group,
    )

    # Assign coords
    # - Silence warning related to datetime precision
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = ds.assign_coords(coords)
        if "lon_bnds" in ds:
            ds = ds.set_coords("lon_bnds")
        if "lat_bnds" in ds:
            ds = ds.set_coords("lat_bnds")

    # Assign global attributes
    ds.attrs = attrs
    return ds


def get_variables(ds):
    """Retrieve the dataset variables."""
    return list(ds.data_vars)


def get_variables_dims(ds):
    """Retrieve the dimensions used by the xr.Dataset variables."""
    variables = get_variables(ds)
    if len(variables) == 0:
        return []
    return np.unique(np.concatenate([list(ds[var].dims) for var in variables])).tolist()


def unused_var_dims(ds):
    """Retrieve the dimensions not used by the the xr.Dataset variables."""
    var_dims = set(get_variables_dims(ds))
    ds_dims = set(ds.dims)
    unused_dims = ds_dims.difference(var_dims)
    return list(unused_dims)


def remove_unused_var_dims(ds):
    """Remove coordinates and dimensions not used by the xr.Dataset variables."""
    if len(ds.data_vars) >= 1:
        unused_dims = unused_var_dims(ds)
        unused_dims = [dim for dim in unused_dims if dim not in ["latv", "lonv", "nv"]]
        ds = ds.drop_dims(unused_dims)
    return ds


def _open_granule(
    filepath,
    scan_mode,
    groups,
    variables,
    decode_cf,
    chunks,
    prefix_group,
):
    """Open granule file into xarray Dataset."""
    from gpm.dataset.datatree import open_datatree

    # Open datatree
    dt = open_datatree(filepath=filepath, chunks=chunks, decode_cf=decode_cf, use_api_defaults=True)

    # Retrieve the granule dataset (without cf decoding)
    ds = _get_scan_mode_dataset(
        dt=dt,
        scan_mode=scan_mode,
        groups=groups,
        variables=variables,
        prefix_group=prefix_group,
    )

    ###-----------------------------------------------------------------------.
    # Specify datatree closer
    # TODO: implement datatree.close() and datatree._close in datatree repository
    # --> datatree._close as iterator ?
    # --> https://github.com/xarray-contrib/datatree/issues/93
    # --> https://github.com/xarray-contrib/datatree/pull/114/files
    ds.set_close(dt._close)

    ###-----------------------------------------------------------------------.
    # If there are dataset variables, remove coords and dimensions not exploited by data variables
    # - Except for nv, lonv, latv bounds dimensions
    return remove_unused_var_dims(ds)

    ###-----------------------------------------------------------------------.
    # Return xr.Dataset


def open_granule(
    filepath,
    scan_mode=None,
    groups=None,
    variables=None,
    decode_cf=True,
    chunks={},
    prefix_group=False,
):
    """Create a lazy ``xarray.Dataset`` with relevant GPM data and attributes for a specific granule.

    Parameters
    ----------
    filepath : str
        Filepath of GPM granule dataset
    scan_mode : str, optional
        Scan mode of the GPM product. The default is ``None``.
        Use ``gpm.available_scan_modes(product, version)`` to get the available scan modes for a specific product.
        The radar products have the following scan modes:

        - ``'FS'``: Full Scan. For Ku, Ka and DPR (since version 7 products).
        - ``'NS'``: Normal Scan. For Ku band and DPR (till version 6 products).
        - ``'MS'``: Matched Scan. For Ka band and DPR (till version 6 products).
        - ``'HS'``: High-sensitivity Scan. For Ka band and DPR.

    variables : list, str, optional
        Variables to read from the HDF5 file.
        The default is ``None`` (all variables).
    groups : list, str, optional
        HDF5 Groups from which to read all variables.
        The default is ``None`` (all groups).
    chunks : int, dict, 'auto' or None, optional
        Chunk size for dask array:

        - ``chunks=-1`` loads the dataset with dask using a single chunk for all arrays.
        - ``chunks={}`` loads the dataset with dask using the file chunks.
        - ``chunks='auto'`` will use dask ``auto`` chunking taking into account the file chunks.

        If you want to load data in memory directly, specify ``chunks=None``.
        The default is ``{}``.

        Hint: xarray's lazy loading of remote or on-disk datasets is often but not always desirable.
        Before performing computationally intense operations, load the dataset
        entirely into memory by invoking ``ds.compute()``.
    decode_cf: bool, optional
        Whether to decode the dataset. The default is ``False``.
    prefix_group: bool, optional
        Whether to add the group as a prefix to the variable names.
        THe default is ``True``.

    Returns
    -------
    ds:  xarray.Dataset

    """
    # Check variables and groups
    variables = check_variables(variables)
    groups = check_groups(groups)

    # Get product and version
    product = get_product_from_filepath(filepath)
    version = get_version_from_filepath(filepath)

    # Check scan_mode
    scan_mode = check_scan_mode(scan_mode, product, version)

    # Open granule
    ds = _open_granule(
        filepath=filepath,
        scan_mode=scan_mode,
        groups=groups,
        variables=variables,
        decode_cf=False,
        chunks=chunks,
        prefix_group=prefix_group,
    )

    # Finalize granule
    return finalize_dataset(
        ds=ds,
        product=product,
        scan_mode=scan_mode,
        decode_cf=decode_cf,
        start_time=None,
        end_time=None,
    )
