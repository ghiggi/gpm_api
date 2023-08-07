#!/usr/bin/env python3
"""
Created on Thu Jul 20 17:23:16 2023

@author: ghiggi
"""
import numpy as np
import xarray as xr


def _integrate_concentration(data, heights):
    # Compute the thickness of each level (difference between adjacent heights)
    thickness = np.diff(heights)
    thickness = np.append(thickness, thickness[-1])
    thickness = np.broadcast_to(thickness, data.shape)
    # Identify where along the profile is always nan
    mask_nan = np.all(np.isnan(data), axis=-1)  # all profile is nan
    # Compute the path by summing up the product of concentration and thickness
    path = np.nansum(thickness * data, axis=-1)
    path[mask_nan] = np.nan
    return path


def integrate_profile_concentration(dataarray, name, scale_factor=None, units=None):
    """Utility to convert LWC or IWC to LWP or IWP.

    Input data have unit g/m³.
    Output data will have unit kg/m²

    height a list or array of corresponding heights for each level.
    """
    if scale_factor is not None:
        if units is None:
            raise ValueError("Specify output 'units' when the scale_factor is applied.")
    # Compute integrated value
    data = dataarray.data.copy()
    heights = np.asanyarray(dataarray["range"].data)
    output = _integrate_concentration(data, heights)
    # Scale output value
    if scale_factor is not None:
        output = output / scale_factor
    # Create DataArray
    da_path = dataarray.isel({"range": 0}).copy()
    da_path.name = name
    da_path.data = output
    if scale_factor:
        da_path.attrs["units"] = units
    return da_path


def check_variable_availabilty(ds, variable, argname):
    if variable not in ds:
        raise ValueError(
            f"{variable} is not a variable of the xr.Dataset. Invalid {argname} argument."
        )


def _get_bin_datarray(ds, bin):
    """Get bin dataarray."""
    if not isinstance(bin, (str, xr.DataArray)):
        raise TypeError("'bin' must be a DataArray or a string indicating the Dataset variable.")
    if isinstance(bin, xr.DataArray):
        return bin
    else:
        check_variable_availabilty(ds, bin, argname="bin")
        return ds[bin]


def _get_variable_dataarray(xr_obj, variable):
    if not isinstance(xr_obj, (xr.DataArray, xr.Dataset)):
        raise TypeError("Expecting a xr.Dataset or xr.DataArray")
    if isinstance(xr_obj, xr.Dataset):
        check_variable_availabilty(xr_obj, variable, argname="variable")
        da = xr_obj[variable]
    else:
        da = xr_obj
    return da


def get_variable_at_bin(xr_obj, bin, variable=None):
    """Retrieve variable values at range bin provided by bin_variable.

    Assume bin values goes from 1 to 176.
    """
    # TODO: check behaviour when bin has 4 dimensions (i.e. binRealSurface)
    # Get variable dataarray
    da_variable = _get_variable_dataarray(xr_obj, variable)
    # Get the bin datarray
    da_bin = _get_bin_datarray(xr_obj, bin=bin)
    if da_bin.ndim >= 4:
        raise NotImplementedError("Undefined behaviour for bin DataArray with >= 4 dimensions")
    da_bin = da_bin - 1
    da_bin = da_bin.compute()
    # Put bin data in memory
    da_bin = da_bin.compute()
    # Identify bin nan values
    da_is_nan = np.isnan(da_bin)
    # Set nan bin values temporary to 0
    da_bin = da_bin.where(~da_is_nan, 0).astype(int)
    # Retrieve values at bin
    da = da_variable.isel({"range": da_bin})
    # Mask values at nan bins
    da = da.where(~da_is_nan)
    # Set original chunks if input DataArray is dask-backed
    # - This avoid later need of unify_chunks when converting to dataframe
    if hasattr(da_variable.data, "chunks"):
        da = da.chunk(da_variable.chunks)
    return da


def get_height_at_bin(xr_obj, bin):
    return get_variable_at_bin(xr_obj, bin, variable="height")


def get_range_slices_with_valid_data(xr_obj, variable):
    """Get the 'range' slices with valid data."""
    # Extract DataArray
    da = _get_variable_dataarray(xr_obj, variable)
    da = da.compute()

    # Check has range dimension
    dims = list(da.dims)
    if "range" not in dims:
        raise ValueError(f"The {variable} variable does not have the 'range' dimension.")

    # Remove 'range' from dimensions over which to aggregate
    dims.remove("range")

    # Get bool array where there are some data (not all nan)
    has_data = ~np.isnan(da).all(dim=dims)
    has_data_arr = has_data.data

    # Identify first and last True occurrence
    n_bins = len(has_data)
    first_true_index = np.argwhere(has_data_arr)[0]
    last_true_index = n_bins - np.argwhere(has_data_arr[::-1])[0] - 1
    if len(first_true_index) == 0:
        raise ValueError(f"No valid data for variable {variable}.")
    isel_dict = {"range": slice(first_true_index.item(), last_true_index.item() + 1)}
    return isel_dict


def select_range_with_valid_data(xr_obj, variable):
    """Select the 'range' interval with valid data."""
    isel_dict = get_range_slices_with_valid_data(xr_obj, variable)
    # Susbet the xarray object
    return xr_obj.isel(isel_dict)
