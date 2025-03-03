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
    if variable is None:
        raise ValueError("Please specify a dataset variable.")
    if variable not in ds:
        raise ValueError(
            f"{variable} is not a variable of the xarray.Dataset. Invalid {argname} argument.",
        )


####-------------------------------------------------------------------
###################
#### Utilities ####
###################


def get_dataset_variables(ds, sort=False):
    """Get list of xarray.Dataset variables."""
    variables = list(ds.data_vars)
    if sort:
        variables = sorted(variables)
    return variables


def get_xarray_variable(xr_obj, variable=None):
    """Return variable DataArray from xarray object.

    If variable is a xr.DataArray, it returns it
    If variable is None and the the input is a xr.DataArray, it returns it
    If the input is a xr.Dataset, it returns the specified variable.
    """
    check_is_xarray(xr_obj)
    if isinstance(variable, xr.DataArray):
        return variable
    if isinstance(xr_obj, xr.Dataset):
        check_variable_availabilty(xr_obj, variable, argname="variable")
        da = xr_obj[variable]
    else:
        da = xr_obj
    return da


def get_default_variable(ds: xr.Dataset, possible_variables) -> str:
    """Return one of the possible default variables.

    Check if one of the variables in 'possible_variables' is present in the xarray.Dataset.
    If neither variable is present, raise an error.
    If both are present, raise an error.
    Return the name of the single available variable in the xarray.Dataset

    Parameters
    ----------
    ds : xarray.Dataset
        The xarray dataset to inspect.
    possible_variables : list of str
        The variable names to look for.

    Returns
    -------
    str
        The name of the variable found in the xarray.Dataset.
    """
    if isinstance(possible_variables, str):
        possible_variables = [possible_variables]
    found_vars = [v for v in possible_variables if v in ds.data_vars]
    if len(found_vars) == 0:
        raise ValueError(f"None of {possible_variables} variables were found in the dataset.")
    if len(found_vars) > 1:
        raise ValueError(f"Multiple variables found: {found_vars}. Please specify which to use.")
    return found_vars[0]


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
            for dim, chunks in zip(ds[var_name].dims, var_chunks, strict=False):
                if dim not in unique_chunks_per_dim:
                    unique_chunks_per_dim[dim] = set()
                    unique_chunks_per_dim[dim].add(chunks)
                if chunks not in unique_chunks_per_dim[dim]:
                    return False

    # If all chunks are unique for each dimension, return True
    return True


def ensure_unique_chunking(ds):
    """Ensure the dataset has unique chunking.

    Conversion to :py:class:`dask.dataframe.DataFrame` requires unique chunking.
    If the xarray.Dataset does not have unique chunking, perform ``ds.unify_chunks``.

    Variable chunks can be visualized with:

    for var in ds.data_vars:
        print(var, ds[var].chunks)

    """
    if not has_unique_chunking(ds):
        ds = ds.unify_chunks()
    return ds


def _xr_first_data_array(da, dim):
    """Return first valid value of a DataArray along a dimension."""
    mask = da.notnull()
    first_valid_idx = mask.argmax(dim=dim)
    first_valid_value = da.isel({dim: first_valid_idx})
    first_valid_value = first_valid_value.where(mask.any(dim=dim))
    return first_valid_value


def xr_first(xr_obj, dim):
    """Return the first valid (non-NaN) value along the specified dimension."""
    check_is_xarray(xr_obj)
    if isinstance(xr_obj, xr.Dataset):
        for var in xr_obj.data_vars:
            if dim in xr_obj[var]:
                xr_obj[var] = _xr_first_data_array(xr_obj[var], dim=dim)
        return xr_obj
    return _xr_first_data_array(xr_obj, dim=dim)


def _drop_constant_dimension_datarray(da):
    """Drop DataArray dimensions over which all numeric values are equal."""
    if not np.issubdtype(da.dtype, np.number):
        return da

    for dim in list(da.dims):
        if dim not in da.dims:
            continue
        # If the variable is constant along this dimension, drop the other dimensions.
        if (da.diff(dim=dim).sum(dim=dim) == 0).all():
            da = xr_first(da, dim=dim)
    return da


def xr_drop_constant_dimension(xr_obj):
    """Return the first valid (non-NaN) value along the specified dimension."""
    check_is_xarray(xr_obj)
    if isinstance(xr_obj, xr.Dataset):
        for var in xr_obj.data_vars:
            xr_obj[var] = _drop_constant_dimension_datarray(xr_obj[var])
        return xr_obj
    return _drop_constant_dimension_datarray(xr_obj)


def broadcast_like(xr_obj, other, add_coords=True):
    """Broadcast an xarray object against another one."""
    xr_obj = xr_obj.broadcast_like(other)
    if add_coords:
        xr_obj = xr_obj.assign_coords(other.coords)
    return xr_obj


def xr_sorted_distribution(da, values, dim):
    """
    Compute the ranked frequency distribution of integer values along a given dimension.

    Parameters
    ----------
    da : xarray.DataArray
        The input data array containing integer values along the specified dimension.
    values : array-like
        An array of the expected values (e.g. np.arange(1, 13) for months,
        np.arange(0, 24) for hours, etc.).
    dim : str
        The name of the dimension along which to compute the ranked distribution (e.g., "year").

    Returns
    -------
    ds_out : xarray.Dataset
        A dataset with three DataArrays along a new dimension "rank":
          - sorted_values: The provided values sorted in descending order of occurrence.
          - occurrence: The count of occurrences for each sorted value.
          - percentage: The percentage occurrence (relative to the size along `dim`).

        For each pixel (or location), index along "rank" to retrieve, for example,
        the most frequent value at rank 0, the second most at rank 1, etc.
    """
    values = np.asarray(values)

    def _np_sorted_distribution(arr, values):
        # Convert to integer type if not already.
        arr = arr.astype(np.int64)
        # Count the occurrences for each expected value.
        counts = np.array([np.count_nonzero(arr == v) for v in values])
        # Sort the expected values in descending order of counts.
        sort_idx = np.argsort(counts)[::-1]
        sorted_values = values[sort_idx].copy()
        sorted_counts = counts[sort_idx]
        total = arr.size
        sorted_percentage = sorted_counts / total * 100.0
        return sorted_values, sorted_counts, sorted_percentage

    # Define dask_gufunc_kwargs
    dask_gufunc_kwargs = {}
    dask_gufunc_kwargs["output_sizes"] = {"rank": len(values)}

    # Apply the distribution function along the specified dimension.
    sorted_vals, occurrence, percentage = xr.apply_ufunc(
        _np_sorted_distribution,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[["rank"], ["rank"], ["rank"]],
        vectorize=True,
        dask="parallelized",
        kwargs={"values": values},
        output_dtypes=[int, int, float],
        dask_gufunc_kwargs=dask_gufunc_kwargs,
    )

    ds_out = xr.Dataset(
        {
            "sorted_values": sorted_vals,
            "occurrence": occurrence,
            "percentage": percentage,
        },
    )
    return ds_out


####-------------------------------------------------------------------
#### Unstacking dimension


def _check_coord_handling(coord_handling):
    if coord_handling not in {"keep", "drop", "unstack"}:
        raise ValueError("coord_handling must be one of 'keep', 'drop', or 'unstack'.")


def _unstack_coordinates(xr_obj, dim, prefix, suffix):
    # Identify coordinates that share the target dimension
    coords_with_dim = _get_non_dimensional_coordinates(xr_obj, dim=dim)
    ds = xr.Dataset()
    for coord_name in coords_with_dim:
        coord_da = xr_obj[coord_name]
        # Split the coordinate DataArray along the target dimension, drop coordinate and merge
        split_ds = unstack_datarray_dimension(coord_da, coord_handling="drop", dim=dim, prefix=prefix, suffix=suffix)
        ds.update(split_ds)
    return ds


def _handle_unstack_non_dim_coords(ds, source_xr_obj, coord_handling, dim, prefix, suffix):
    # Deal with coordinates sharing the target dimension
    if coord_handling == "keep":
        return ds
    if coord_handling == "unstack":
        ds_coords = _unstack_coordinates(source_xr_obj, dim=dim, prefix=prefix, suffix=suffix)
        ds.update(ds_coords)
    # Remove non dimensional coordinates (unstack and drop coord_handling)
    ds = ds.drop_vars(_get_non_dimensional_coordinates(ds, dim=dim))
    return ds


def _get_non_dimensional_coordinates(xr_obj, dim):
    return [coord_name for coord_name, coord_da in xr_obj.coords.items() if dim in coord_da.dims and coord_name != dim]


def unstack_datarray_dimension(da, dim, coord_handling="keep", prefix="", suffix=""):
    """
    Split a DataArray along a specified dimension into a Dataset with separate prefixed and suffixed variables.

    Parameters
    ----------
    da : xarray.DataArray
        The DataArray to split.
    dim : str
        The dimension along which to split the DataArray.
    coord_handling : str, optional
        Option to handle coordinates sharing the target dimension.
        Choices are 'keep', 'drop', or 'unstack'. Defaults to 'keep'.
    prefix : str, optional
        String to prepend to each new variable name.
    suffix : str, optional
        String to append to each new variable name.

    Returns
    -------
    xarray.Dataset
        A Dataset with each variable split along the specified dimension.
        The Dataset variables are named  "{prefix}{name}{suffix}{dim_value}".
        Coordinates sharing the target dimension are handled based on `coord_handling`.
    """
    # Retrieve DataArray name
    name = da.name
    # Unstack variables
    ds = da.to_dataset(dim=dim)
    rename_dict = {dim_value: f"{prefix}{name}{suffix}{dim_value}" for dim_value in list(ds.data_vars)}
    ds = ds.rename_vars(rename_dict)
    # Deal with coordinates sharing the target dimension
    return _handle_unstack_non_dim_coords(
        ds=ds,
        source_xr_obj=da,
        coord_handling=coord_handling,
        dim=dim,
        prefix=prefix,
        suffix=suffix,
    )


def unstack_dataset_dimension(ds, dim, coord_handling="keep", prefix="", suffix=""):
    """
    Split Dataset variables with the specified dimension into separate prefixed and suffixed variables.

    Parameters
    ----------
    ds : xarray.Dataset
        The DataArray to split.
    dim : str
        The dimension along which to split the DataArray.
    coord_handling : str, optional
        Option to handle coordinates sharing the target dimension.
        Choices are 'keep', 'drop', or 'unstack'. Defaults to 'keep'.
    prefix : str, optional
        String to prepend to each new variable name.
    suffix : str, optional
        String to append to each new variable name.

    Returns
    -------
    xr.Dataset
        A Dataset with each variable with dimension `dim` split into new variables.
        The new Dataset variables are named "{prefix}{name}{suffix}{dim_value}".
        Coordinates sharing the target dimension are handled based on `coord_handling`.
    """
    # Identify variables that have the target dimension
    variables_to_split = [var for var in ds.data_vars if dim in ds[var].dims]

    # Identify variables that do NOT have the target dimension
    variables_to_keep = [var for var in ds.data_vars if dim not in ds[var].dims]

    # Initialize the new Dataset with variables to keep
    ds_unstacked = ds[variables_to_keep].copy()

    # Loop over DataArray
    for var in variables_to_split:
        ds_unstacked.update(
            unstack_datarray_dimension(ds[var], dim=dim, coord_handling="keep", prefix=prefix, suffix=suffix),
        )

    # Deal with coordinates sharing the target dimension
    ds_unstacked = _handle_unstack_non_dim_coords(
        ds=ds_unstacked,
        source_xr_obj=ds,
        dim=dim,
        coord_handling=coord_handling,
        prefix=prefix,
        suffix=suffix,
    )
    return ds_unstacked


def unstack_dimension(xr_obj, dim, coord_handling="keep", prefix="", suffix=""):
    """
    Split xarray object with the specified dimension into separate prefixed and suffixed Dataset variables.

    Parameters
    ----------
    xr_obj : xarray.DataArray, xarray.Dataset
        The DataArray to split.
    dim : str
        The dimension along which to split the DataArray.
    coord_handling : str, optional
        Option to handle coordinates sharing the target dimension.
        Choices are 'keep', 'drop', or 'unstack'. Defaults to 'keep'.
    prefix : str, optional
        String to prepend to each new variable name.
    suffix : str, optional
        String to append to each new variable name.

    Returns
    -------
    xr.Dataset
        A Dataset with each variable with dimension `dim` split into new variables.
        The new Dataset variables are named "{prefix}{name}{suffix}{dim_value}".
        Coordinates sharing the target dimension are handled based on `coord_handling`.
    """
    check_is_xarray(xr_obj)
    _check_coord_handling(coord_handling)
    if isinstance(xr_obj, xr.DataArray):
        return unstack_datarray_dimension(xr_obj, dim=dim, coord_handling=coord_handling, prefix=prefix, suffix=suffix)
    return unstack_dataset_dimension(xr_obj, dim=dim, coord_handling=coord_handling, prefix=prefix, suffix=suffix)


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
