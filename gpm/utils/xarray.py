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
"""This module contains general utility for xarray objects."""
import functools

import numpy as np
import xarray as xr

####-------------------------------------------------------------------
#################
#### Checker ####
#################


def check_is_xarray(x):
    if not isinstance(x, (xr.DataArray, xr.Dataset)):
        raise TypeError("Expecting a xarray.Dataset or xarray.DataArray.")


def check_is_xarray_dataarray(x):
    if not isinstance(x, xr.DataArray):
        raise TypeError("Expecting a xarray.DataArray.")


def check_is_xarray_dataset(x):
    if not isinstance(x, xr.Dataset):
        raise TypeError("Expecting a xarray.Dataset.")


def check_variable_availabilty(ds, variable, argname):
    """Check variable availability in an xarray Dataset."""
    if variable not in ds:
        raise ValueError(
            f"{variable} is not a variable of the xarray.Dataset. Invalid {argname} argument.",
        )


####-------------------------------------------------------------------
###################
#### Utilities ####
###################


def get_dataset_variables(ds, sort=False):
    """Get list of `xarray.Dataset` variables."""
    variables = list(ds.data_vars)
    if sort:
        variables = sorted(variables)
    return variables


def get_xarray_variable(xr_obj, variable=None):
    """Return variable DataArray from xarray object.

    If the input is a xr.DataArray, it return it
    If the input is a xr.Dataset, it return the specified variable.
    """
    check_is_xarray(xr_obj)
    if isinstance(xr_obj, xr.Dataset):
        check_variable_availabilty(xr_obj, variable, argname="variable")
        da = xr_obj[variable]
    else:
        da = xr_obj
    return da


def get_dimensions_without(xr_obj, dims):
    """Return the dimensions of the xarray object without the specified dimensions."""
    if isinstance(dims, str):
        dims = [dims]
    data_dims = np.array(list(xr_obj.dims))
    return data_dims[np.isin(data_dims, dims, invert=True)].tolist()


def has_unique_chunking(ds):
    """Check if a dataset has unique chunking."""
    if not isinstance(ds, xr.Dataset):
        raise ValueError("Input must be an xarray Dataset.")

    # Create a dictionary to store unique chunk shapes for each dimension
    unique_chunks_per_dim = {}

    # Iterate through each variable's chunks
    for var_name in ds.variables:
        if hasattr(ds[var_name].data, "chunks"):  # is dask array
            var_chunks = ds[var_name].data.chunks
            for dim, chunks in zip(ds[var_name].dims, var_chunks):
                if dim not in unique_chunks_per_dim:
                    unique_chunks_per_dim[dim] = set()
                    unique_chunks_per_dim[dim].add(chunks)
                if chunks not in unique_chunks_per_dim[dim]:
                    return False

    # If all chunks are unique for each dimension, return True
    return True


def ensure_unique_chunking(ds):
    """Ensure the dataset has unique chunking.

    Conversion to `dask.dataframe.DataFrame` requires unique chunking.
    If the `xarray.Dataset` does not have unique chunking, perform ``ds.unify_chunks``.

    Variable chunks can be visualized with:

    for var in ds.data_vars:
        print(var, ds[var].chunks)

    """
    if not has_unique_chunking(ds):
        ds = ds.unify_chunks()
    return ds


####-------------------------------------------------------------------
####################
#### Decorators ####
####################


def ensure_dim_order_dataarray(da, func, *args, **kwargs):
    """Ensure that the output DataArray has the same dimensions order as the input.

    New dimensions are moved to the last positions.
    """
    # Get the original dimension order
    original_dims = da.dims
    dict_coord_dims = {coord: da[coord].dims for coord in list(da.coords)}

    # Apply the function to the DataArray
    da_out = func(da, *args, **kwargs)

    # Check output type
    if not isinstance(da_out, xr.DataArray):
        raise TypeError("The function does not return a xr.DataArray.")

    # Check which of the original dimensions are still present
    dim_order = [dim for dim in original_dims if dim in da_out.dims]

    # Transpose the result to ensure the same dimension order
    da_out = da_out.transpose(*dim_order, ...)

    # Transpose the coordinates to
    for coord in list(da_out.coords):
        if coord in dict_coord_dims:
            dim_order = [dim for dim in dict_coord_dims[coord] if dim in da_out[coord].dims]
            da_out[coord] = da_out[coord].transpose(*dim_order, ...)
    return da_out


def ensure_dim_order_dataset(ds, func, *args, **kwargs):
    """Ensure that the output Dataset has the same dimensions order as the input.

    New dimensions are moved to the last positions.
    """
    # Get the original dimension order
    dict_coord_dims = {coord: ds[coord].dims for coord in list(ds.coords)}
    dict_var_dims = {var: ds[var].dims for var in list(ds.data_vars)}

    # Apply the function to the Dataset
    ds_out = func(ds, *args, **kwargs)

    if not isinstance(ds_out, xr.Dataset):
        raise TypeError("The function does not return a xr.Dataset.")

    # Check which of the original variables and dimensions are still present and reorder
    for var in list(ds_out.data_vars):
        if var in dict_var_dims:
            dim_order = [dim for dim in dict_var_dims[var] if dim in ds_out[var].dims]
            ds_out[var] = ds_out[var].transpose(*dim_order, ...)
    for coord in list(ds_out.coords):
        if coord in dict_coord_dims:
            dim_order = [dim for dim in dict_coord_dims[coord] if dim in ds_out[coord].dims]
            ds_out[coord] = ds_out[coord].transpose(*dim_order, ...)

    return ds_out


def xr_ensure_dimension_order(func):
    """Decorator which ensures the output xarray object has same dimension order as input.

    The decorator expects that the functions return the same type of xarray object !

    The decorator can deal with functions that:
    - returns an xarray object with new dimensions
    - returns an xarray object with less dimensions than the originals

    New dimensions are moved to the last positions.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        xr_obj = args[0]  # Assuming the first argument is the dataset
        if isinstance(xr_obj, xr.Dataset):
            return ensure_dim_order_dataset(xr_obj, func, *args[1:], **kwargs)
        return ensure_dim_order_dataarray(xr_obj, func, *args[1:], **kwargs)

    return wrapper


@xr_ensure_dimension_order
def squeeze_unsqueeze_dataarray(da, func, *args, **kwargs):
    """Ensure that the output DataArray has the same dimensions as the input.

    Dimensions of size 1 are kept also if the function drop them !
    New dimensions are moved to the last positions.
    """
    # Retrieve dimension to be squeezed
    original_dims = set(da.dims)
    squeezed_dims = original_dims - set(da.squeeze().dims)

    # List coordinates which are squeezed
    dict_squeezed = {dim: [] for dim in squeezed_dims}
    for dim in squeezed_dims:
        for coord in list(da.coords):
            if dim in da[coord].dims:
                dict_squeezed[dim].append(coord)
    # Squeeze
    da = da.squeeze()

    # Apply function
    da = func(da, *args, **kwargs)  # Call the function with the squeezed dataset

    # Check output type
    if not isinstance(da, xr.DataArray):
        raise TypeError("The function does not return a xr.DataArray.")

    # Unsqueeze back
    for dim, coords in dict_squeezed.items():
        if dim not in da.dims:
            da = da.expand_dims(dim=dim, axis=None)
        for coord in coords:
            if dim not in da[coord].dims:  # coord with same name as dim are automatically expanded !
                da[coord] = da[coord].expand_dims(dim=dim, axis=None)

    # Deal with coordinates named as dimension but without such dimension !
    # for dim, coords in dict_squeezed.items():
    #     if len(coords) == 0 and dim in da.coords:
    #         scalar_coord_value = da[dim].data[0]
    #         da = da.drop_vars(dim)
    #         da = da.assign_coords({"___tmp_coord__": scalar_coord_value}).rename({"___tmp_coord__": dim})
    return da


@xr_ensure_dimension_order
def squeeze_unsqueeze_dataset(ds, func, *args, **kwargs):
    """Ensure that the output Dataset has the same dimensions as the input.

    Dimensions of size 1 are kept also if the function drop them !
    New dimensions are moved to the last positions.
    """
    # Retrieve dimension to be squeezed
    original_dims = set(ds.dims)
    squeezed_dims = original_dims - set(ds.squeeze().dims)

    # List coordinates which are squeezed
    dict_squeezed = {dim: [] for dim in squeezed_dims}
    for dim in squeezed_dims:
        for var in ds.variables:  # coords + variables
            if dim in ds[var].dims:
                dict_squeezed[dim].append(var)
    # Squeeze
    ds = ds.squeeze()

    # Apply function
    ds = func(ds, *args, **kwargs)  # Call the function with the squeezed dataset

    # Check output type
    if not isinstance(ds, xr.Dataset):
        raise TypeError("The function does not return a xr.Dataset.")

    # Unsqueeze back
    for dim, variables in dict_squeezed.items():
        for var in variables:
            if dim not in ds[var].dims:
                ds[var] = ds[var].expand_dims(dim=dim, axis=None)  # not same order as start
    return ds


def xr_squeeze_unsqueeze(func):
    """Decorator that squeeze-unsqueeze the xarray object before passing it to the function.

    This decorator allow to keep the dimensions of the xarray object intact.
    Dimensions of size 1 are kept also if the function drop them.
    The dimension order of the arrays is conserved.
    New dimensions are moved to the last positions.

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        xr_obj = args[0]  # Assuming the first argument is the dataset
        if isinstance(xr_obj, xr.Dataset):
            return squeeze_unsqueeze_dataset(xr_obj, func, *args[1:], **kwargs)
        return squeeze_unsqueeze_dataarray(xr_obj, func, *args[1:], **kwargs)

    return wrapper
