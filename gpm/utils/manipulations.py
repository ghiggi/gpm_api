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
"""This module contains functions for manipulating GPM-API Datasets."""
import numpy as np
import xarray as xr

from gpm.checks import (
    check_has_vertical_dim,
    get_vertical_variables,
    has_spatial_dim,
    has_vertical_dim,
)
from gpm.utils.xarray import (
    check_variable_availabilty,
    get_xarray_variable,
    xr_squeeze_unsqueeze,
)

# --------------------------------------------------------------------------
# IDEAS
# extract_dataset_above/below_bin
# - Develop method to reinsert in original dataset !

# --------------------------------------------------------------------------


def _get_vertical_dim(da):
    """Return the name of the vertical dimension."""
    from gpm.checks import get_vertical_dimension

    vertical_dimension = get_vertical_dimension(da)
    if len(vertical_dimension) == 0:
        variable = da.name
        raise ValueError(f"The {variable} variable does not have a vertical dimension.")
    if len(vertical_dimension) != 1:
        raise ValueError("Only 1 vertical dimension allowed.")

    return vertical_dimension[0]


def get_range_axis(da):
    """Get range dimension axis index."""
    vertical_dim = _get_vertical_dim(da)
    return np.where(np.isin(list(da.dims), vertical_dim))[0].item()


def _integrate_concentration(data, height):
    # Compute the thickness of each level (difference between adjacent heights)
    thickness = np.diff(height, axis=-1)  # assume vertical dimension as last dimension
    thickness = np.concatenate((thickness, thickness[..., [-1]]), axis=-1)
    # Identify where along the profile is always nan
    mask_nan = np.all(np.isnan(data), axis=-1)  # all profile is nan
    # Compute the path by summing up the product of concentration and thickness
    path = np.nansum(thickness * data, axis=-1)
    path[mask_nan] = np.nan
    return path


def integrate_profile_concentration(dataarray, name, scale_factor=None, units=None):
    """Utility to convert LWC or IWC to LWP or IWP.

    Input data have unit g/m³.
    Output data will have unit kg/m² if scale_factor=1000

    height a list or array of corresponding heights for each level.
    """
    if scale_factor is not None and units is None:
        raise ValueError("Specify output 'units' when the scale_factor is applied.")
    vertical_dim = _get_vertical_dim(dataarray)
    dataarray = dataarray.transpose(..., vertical_dim)  # as last dimension
    # Compute integrated value
    data = dataarray.data.copy()
    height = np.asanyarray(dataarray["height"].data)
    output = _integrate_concentration(data, height)
    # Scale output value
    if scale_factor is not None:
        output = output / scale_factor
    # Create DataArray
    da_path = dataarray.isel({vertical_dim: 0}).copy()
    da_path.name = name
    da_path.data = output
    if scale_factor:
        da_path.attrs["units"] = units
    return da_path


def convert_from_decibel(da):
    """Convert dB to unit."""
    return np.power(10.0, da / 10)


def convert_to_decibel(da):
    """Convert unit to dB."""
    return 10 * np.log10(da)


def conversion_factors_degree_to_meter(latitude):
    """
    Calculate conversion factors from degrees to meters as a function of latitude.

    Parameters
    ----------
    latitude : numpy.ndarray
        Latitude in degrees where the conversion is needed

    Returns
    -------
    (cx, cy) : tuple
        Tuple containing conversion factors for longitude and latitude
    """
    # Earth's radius at the equator (in meters)
    R = 6378137

    # Calculate the conversion factor for latitude (constant per degree latitude)
    cy = np.pi * R / 180.0

    # Calculate the conversion factor for longitude (changes with latitude)
    cx = cy * np.cos(np.deg2rad(latitude))

    return cx, cy


####-------------------------------------------------------------------------------------------------------------------.
##########################
#### Range bin slicer ####
##########################


def get_bin_dataarray(xr_obj, bins, mask_first_bin=False, mask_last_bin=False, fillvalue=None):
    """Get bin `xarray.DataArray`."""
    # Retrieve bins DataArray
    da_bin = _get_bin_dataarray(xr_obj, bins=bins)

    # Check bin DataArray dimensions validity (only spatial dimensions)
    if "range" in da_bin.dims:
        raise ValueError("The bin DataArray must not have the 'range' dimension.")
    if "radar_frequency" in da_bin.dims:
        raise ValueError(
            "The bin DataArray must not have the 'radar_frequency' dimension. Please first subset the dataset.",
        )
    if not has_spatial_dim(da_bin, strict=True, squeeze=True):
        raise ValueError("The bin DataArray is allowed to only have spatial dimensions.")

    # Ensure bin value validity
    da_bin, da_mask = _get_valid_da_bin(
        xr_obj,
        da_bin=da_bin,
        mask_first_bin=mask_first_bin,
        mask_last_bin=mask_last_bin,
        fillvalue=fillvalue,
    )
    return da_bin, da_mask


def _get_bin_dataarray(xr_obj, bins):
    """Retrieve bins DataArray. The xr_obj is used only if bins is a string."""
    if not isinstance(bins, (str, xr.DataArray, int, list)):
        raise TypeError("'bins' must be a xarray.DataArray or a string indicating the xarray.Dataset variable.")
    if isinstance(bins, (int, list)):
        raise TypeError("A list or a single bin value are not accepted. Use xr_obj.sel(range=bins) instead !")

    # Retrieve DataArray if string (DataArray coord or xr.Dataset coord/variable)
    if isinstance(bins, str):
        check_variable_availabilty(xr_obj, variable=bins, argname="bins")
        bins = xr_obj[bins]
    return bins


def _get_valid_da_bin(xr_obj, da_bin, mask_first_bin=False, mask_last_bin=False, fillvalue=None):
    """Return a valid bin `xarray.DataArray` with a mask for the invalid/unavailable bins."""
    # Retrieve minimum and maximum available range indices
    vmin = xr_obj["range"].data.min()
    vmax = xr_obj["range"].data.max()
    # Define default fillvalue
    if fillvalue is None:
        fillvalue = vmin if mask_last_bin else vmax
    # Put bin data in memory
    da_bin = da_bin.copy()
    da_bin = da_bin.compute().astype(float)
    # Set to nan first or last bin (options for dataset extraction)
    if mask_first_bin:
        da_bin.data[da_bin.data == vmin] = np.nan

    if mask_last_bin:
        da_bin.data[da_bin.data == vmax] = np.nan

    # Identify bin nan values
    da_is_nan = np.isnan(da_bin)
    # Identify bin values outside of available range gates
    da_invalid = np.logical_or(da_bin < vmin, da_bin > vmax)
    # Raise error if all bin index are nan
    if np.all(da_is_nan.data):
        raise ValueError("All range bin indices are NaN !")
    # Raise error if all bin index are outside the available range bins
    if np.all(da_invalid.data):
        raise ValueError(f"All range bin indices are outside of the available range gates [{vmin}, {vmax}] !")
    # Define mask with invalid bins
    # --> This address np.nan, 0, and out of range values
    da_mask = np.logical_or(da_is_nan, da_invalid)
    # Raise error if all bin index are outside the available range bins
    if np.all(da_mask.data):
        raise ValueError("All range bin indices are invalid !")
    # Set invalid/ nan range bin indices to a fillvalue to enable selection with .sel
    # --> With the function defaults, the last range value vmax
    # --> The gates with invalid range bin indices will be masked out with da_mask
    da_bin = da_bin.where(~da_mask, fillvalue).astype(int)
    return da_bin, da_mask


def _select_range_slice(da, da_bin, da_mask):
    # Retrieve values at specified range gates
    da_slice = da.sel({"range": da_bin})
    # Mask values at invalid range gates
    return da_slice.where(~da_mask)


def slice_range_at_bin(xr_obj, bins):
    """Extract values at the range bins specified by ``bin_variable``.

    ``bin_variable`` can be a bin `xarray.DataArray` or the name of a bin variable of the input `xarray.Dataset`.

    The function extract the gates based on the 'range' coordinate values.
    Bin values are assumed to start at 1, not 0 !

    If you want to extract a slice at a single range bin, use instead ``xr_obj.sel(range=range_bin_value)``.

    Parameters
    ----------
    xr_obj : `xarray.DataArray` or `xarray.Dataset`
        xarray object with the 'range' dimension (and coordinate).
    bins : str or `xarray.DataArray`
        Either a `xarray.DataArray` or a string pointing to the dataset variable with the range bins to extract.
        Bin values are assumed to start at 1, not 0 !

    Returns
    -------
    xr_out : `xarray.Dataset` or `xarray.DataArray`
        xarray object with values at the specified range bins.

    """
    check_has_vertical_dim(xr_obj)

    # Get the bin DataArray
    da_bin, da_mask = get_bin_dataarray(xr_obj, bins=bins)

    # Slice along the 'range' dimension
    is_dataset_input = isinstance(xr_obj, xr.Dataset)
    if is_dataset_input:
        vertical_variables = xr_obj.gpm.vertical_variables
        non_vertical_variables = set(xr_obj.data_vars) - set(vertical_variables)
        # Copy non vertical variables
        xr_out = xr_obj[non_vertical_variables].copy()
        # Slice vertical variables
        for var in vertical_variables:
            xr_out[var] = _select_range_slice(da=xr_obj[var], da_bin=da_bin, da_mask=da_mask)
    else:
        xr_out = _select_range_slice(
            da=xr_obj,
            da_bin=da_bin,
            da_mask=da_mask,
        )
    return xr_out


####------------------------------------------------------------------------------------------------------------------.
############################
#### Range index slicer ####
############################


def get_range_index_at_value(da, value):
    """Retrieve index along the range dimension where the `xarray.DataArray` values is closest to value."""
    vertical_dim = _get_vertical_dim(da)
    return np.abs(da - value).argmin(dim=vertical_dim).compute()


def get_range_index_at_min(da):
    """Retrieve index along the range dimension where the `xarray.DataArray` has minimum values."""
    vertical_dim = _get_vertical_dim(da)
    return da.argmin(dim=vertical_dim).compute()


def get_range_index_at_max(da):
    """Retrieve index along the range dimension where the `xarray.DataArray` has maximum values."""
    vertical_dim = _get_vertical_dim(da)
    return da.argmax(dim=vertical_dim).compute()


def slice_range_at_value(xr_obj, value, variable=None):
    """Slice the 3D arrays where the variable values are close to value."""
    da = get_xarray_variable(xr_obj, variable=variable)
    vertical_dim = _get_vertical_dim(da)
    idx = get_range_index_at_value(da=da, value=value)
    return xr_obj.isel({vertical_dim: idx})


def slice_range_at_max_value(xr_obj, variable=None):
    """Slice the 3D arrays where the variable values are at maximum."""
    da = get_xarray_variable(xr_obj, variable=variable)
    vertical_dim = _get_vertical_dim(da)
    idx = get_range_index_at_max(da=da)
    return xr_obj.isel({vertical_dim: idx})


def slice_range_at_min_value(xr_obj, variable=None):
    """Slice the 3D arrays where the variable values are at minimum."""
    da = get_xarray_variable(xr_obj, variable=variable)
    vertical_dim = _get_vertical_dim(da)
    idx = get_range_index_at_min(da=da)
    return xr_obj.isel({vertical_dim: idx})


def slice_range_at_temperature(ds, temperature, variable_temperature="airTemperature"):
    """Slice the 3D arrays along a specific isotherm."""
    return slice_range_at_value(ds, variable=variable_temperature, value=temperature)


####------------------------------------------------------------------------------------------------------------------.
###############################
#### Range interval slices ####
###############################


def get_range_slices_with_valid_data(xr_obj, variable=None):
    """Get the vertical ('range'/'height') slices with valid data."""
    # Extract DataArray
    da = get_xarray_variable(xr_obj, variable)
    da = da.compute()

    # Retrieve vertical dimension name
    vertical_dim = _get_vertical_dim(da)

    # Remove 'range' from dimensions over which to aggregate
    dims = list(da.dims)
    dims.remove(vertical_dim)

    # Get bool array where there are some data (not all nan)
    has_data = ~np.isnan(da).all(dim=dims)
    has_data_arr = has_data.data
    if not has_data_arr.any():
        raise ValueError(f"No valid data for variable {variable}.")

    # Identify first and last True occurrence
    n_bins = len(has_data)
    first_true_index = np.argwhere(has_data_arr)[0]
    last_true_index = n_bins - np.argwhere(has_data_arr[::-1])[0] - 1
    return {vertical_dim: slice(first_true_index.item(), last_true_index.item() + 1)}


def get_range_slices_within_values(xr_obj, variable=None, vmin=-np.inf, vmax=np.inf):
    """Get the 'range' slices with data within a given data interval."""
    # Extract DataArray
    da = get_xarray_variable(xr_obj, variable)
    da = da.compute()

    # Retrieve vertical dimension name
    vertical_dim = _get_vertical_dim(da)

    # Remove 'range' from dimensions over which to aggregate
    dims = list(da.dims)
    dims.remove(vertical_dim)

    # Get bool array indicating where data are in the value interval
    is_within_interval = np.logical_and(da >= vmin, da <= vmax)
    if not is_within_interval.any():
        raise ValueError(f"No data within the requested value interval for variable {variable}.")

    # Identify first and last True occurrence
    n_bins = len(da[vertical_dim])
    first_true_index = is_within_interval.argmax(dim=vertical_dim).min().item()
    axis_idx = np.where(np.isin(list(da.dims), vertical_dim))[0]
    last_true_index = n_bins - 1 - np.flip(is_within_interval, axis=axis_idx).argmax(dim=vertical_dim).min().item()
    return {vertical_dim: slice(first_true_index, last_true_index + 1)}


def subset_range_with_valid_data(xr_obj, variable=None):
    """Select the 'range' interval with valid data."""
    isel_dict = get_range_slices_with_valid_data(xr_obj, variable=variable)
    # Susbet the xarray object
    return xr_obj.isel(isel_dict)


def subset_range_where_values(xr_obj, variable=None, vmin=-np.inf, vmax=np.inf):
    """Select the 'range' interval where values are within the [vmin, vmax] interval."""
    isel_dict = get_range_slices_within_values(xr_obj, variable=variable, vmin=vmin, vmax=vmax)
    return xr_obj.isel(isel_dict)


####-------------------------------------------------------------------------------------------------------------------.
##########################
#### Height utilities ####
##########################


def get_height_dataarray(xr_obj):
    if "height" in xr_obj.coords:
        da_height = xr_obj["height"]
    elif isinstance(xr_obj, xr.Dataset):
        da_height = get_xarray_variable(xr_obj, variable="height")
    else:
        raise ValueError("Expecting a xarray.DataArray with the 'height' coordinate.")
    return da_height


def slice_range_at_height(xr_obj, value):
    """Slice the 3D array at a given height."""
    return slice_range_at_value(xr_obj, variable="height", value=value)


def get_height_at_temperature(da_height, da_temperature, temperature):
    """Retrieve height at a specific temperature."""
    vertical_dim = _get_vertical_dim(da_height)
    idx_desired_temperature = get_range_index_at_value(da_temperature, temperature)
    return da_height.isel({vertical_dim: idx_desired_temperature})


def get_height_at_bin(xr_obj, bins):
    """Retrieve height values at range bins specified by ``bins``."""
    # Retrieve height DataArray
    da_height = get_height_dataarray(xr_obj)
    # Retrieve bins DataArray
    da_bins = _get_bin_dataarray(xr_obj, bins)
    # Return height
    return slice_range_at_bin(xr_obj=da_height, bins=da_bins)


####------------------------------------------------------------------------------------------------------------------.
#######################
#### Phase utility ####
#######################


def get_bright_band_mask(ds):
    """Retrieve bright band mask defined by ``binBBBottom`` and ``binBBTop`` bin variables.

    The bin is numerated from top to bottom.
    ``binBBTop`` has lower values than ``binBBBottom``.
    ``binBBBottom`` and ``binBBTop`` are ``NaN`` when bright band limit is not detected !
    """
    # Retrieve required DataArrays
    da_bb_bottom = ds["binBBBottom"]
    da_bb_top = ds["binBBTop"]
    # Create 3D array with bin index
    da_bin_index = ds["range"]
    # Identify bright band mask
    return np.logical_and(da_bin_index >= da_bb_top, da_bin_index <= da_bb_bottom)


def get_liquid_phase_mask(ds):
    """Retrieve the mask of the liquid phase profile."""
    da_height = ds["height"]
    da_height_0 = ds["heightZeroDeg"]
    return da_height < da_height_0


def get_solid_phase_mask(ds):
    """Retrieve the mask of the solid phase profile."""
    da_height = ds["height"]
    da_height_0 = ds["heightZeroDeg"]
    return da_height >= da_height_0


####------------------------------------------------------------------------------------------------------------------.
#### Variable and dimension selection


def select_spatial_3d_variables(ds, strict=False, squeeze=True):
    """Return `xarray.Dataset` with only 3D spatial variables."""
    from gpm.checks import get_spatial_3d_variables

    variables = get_spatial_3d_variables(ds, strict=strict, squeeze=squeeze)
    return ds[variables]


def select_spatial_2d_variables(ds, strict=False, squeeze=True):
    """Return `xarray.Dataset` with only 2D spatial variables."""
    from gpm.checks import get_spatial_2d_variables

    variables = get_spatial_2d_variables(ds, strict=strict, squeeze=squeeze)
    return ds[variables]


def select_transect_variables(ds, strict=False, squeeze=True):
    """Return `xarray.Dataset` with only transect variables."""
    from gpm.checks import get_transect_variables

    variables = get_transect_variables(ds, strict=strict, squeeze=squeeze)
    return ds[variables]


def select_vertical_variables(ds):
    """Return `xarray.Dataset` with only variables with vertical dimension."""
    variables = get_vertical_variables(ds)
    return ds[variables]


def select_frequency_variables(ds):
    """Return `xarray.Dataset` with only multifrequency variables."""
    from gpm.checks import get_frequency_variables

    variables = get_frequency_variables(ds)
    return ds[variables]


def select_bin_variables(ds):
    """Return `xarray.Dataset` with only bin variables."""
    from gpm.checks import get_bin_variables

    variables = get_bin_variables(ds)
    return ds[variables]


####------------------------------------------------------------------------------------------------------------------.
###############################
#### 3D Dataset Extraction ####
###############################


def _shorten_dataset_above(ds_new, new_range_size):
    if new_range_size is not None:
        n_bins = len(ds_new["range"])
        ds_new = ds_new.isel({"range": slice(n_bins - new_range_size, None)})
    return ds_new


def _shorten_dataset_below(ds_new, new_range_size):
    if new_range_size is not None:
        ds_new = ds_new.isel({"range": slice(0, new_range_size)})
    return ds_new


def _update_bin_variables_above(ds, da_bin, da_mask, new_range_size):
    for var in ds.gpm.bin_variables:
        # Retrieve new bin index
        new_bin = new_range_size - (da_bin - ds[var])
        # Mask with np.nan invalid bins
        valid_values = np.logical_and(new_bin >= 1, new_bin <= new_range_size)
        ds[var] = new_bin.where(np.logical_and(valid_values, ~da_mask))
    return ds


def _update_bin_variables_below(ds, da_bin, da_mask, new_range_size):
    for var in ds.gpm.bin_variables:
        # Retrieve new bin index
        new_bin = ds[var] - da_bin + 1  # + 1 because range coord start at 1
        # Mask with np.nan invalid bins
        valid_values = np.logical_and(new_bin >= 1, new_bin <= new_range_size)
        ds[var] = new_bin.where(np.logical_and(valid_values, ~da_mask))
    return ds


def _add_new_range_coords(ds, new_range_size):
    ds = ds.assign_coords(
        {
            "gpm_range_id": ("range", np.arange(0, new_range_size)),
            "range": np.arange(1, new_range_size + 1),
        },
    )
    return ds


def get_vertical_coords_and_vars(ds):
    """Return a 'prototype' with only spatial and vertical dimensions."""
    vertical_variables = get_vertical_variables(ds)
    vertical_coords = [coord for coord in ds.coords if has_vertical_dim(ds[coord]) and has_spatial_dim(ds[coord])]
    vertical_variables += vertical_coords
    if len(vertical_variables) == 0:
        raise ValueError("No vertical variables to extract.")
    return vertical_variables


def get_vertical_datarray_prototype(ds, fill_value=np.nan):
    """Return a `xarray.DataArray` 'prototype' with only spatial and vertical dimensions."""
    vertical_variables = get_vertical_coords_and_vars(ds)
    da = ds[vertical_variables[0]]
    da = xr.full_like(da, fill_value=fill_value).compute()
    da.name = "prototype"
    return ensure_vertical_datarray_prototype(da)


def ensure_vertical_datarray_prototype(da):
    """Return a `xarray.DataArray` with only spatial and vertical dimensions."""
    valid_dims = da.gpm.spatial_dimensions + da.gpm.vertical_dimension
    invalid_dims = set(da.dims) - set(valid_dims)
    if invalid_dims:
        da = da.isel({dim: 0 for dim in invalid_dims})
    return da


def reverse_range(ds):
    """Reverse the range dimension of a dataset.

    The bin variables are updated accordingly.
    """
    range_size = len(ds["range"])
    # Reverse
    ds = ds.isel({"range": slice(None, None, -1)})
    # Update bin variables
    for var in ds.gpm.bin_variables:
        ds[var] = range_size - ds[var] + 1
    # Update range coordinates
    ds = _add_new_range_coords(ds, new_range_size=range_size)
    return ds


@xr_squeeze_unsqueeze
def extract_dataset_above_bin(ds, bins, new_range_size=None, strict=False, reverse=False):
    """
    Extract a radar dataset with the range bins above the ``<bins>`` index.

    If ``reverse=False``, the new last range bin corresponds to the ``<bins>`` index.
    If ``reverse=True``, the new first range bin corresponds to the ``<bins>`` index.

    Parameters
    ----------
    ds : `xarray.Dataset`
        GPM RADAR xarray dataset.
    bin_variable : str
        The variable name containing the radar gate bin index of interest.
        GPM bin variables are assumed to start at 1, not 0!
    new_range_size : int, optional
        If specified, the size of the new range dimension.
        The dataset is shortened along the range dimension (from the top).
        The default is ``None``.
    strict: bool, optional
        If ``True``, it extract only radar gates above the bin index.
        If ``False``, it extract also the radar gate at the bin index.
        The default is `False`.
    reverse: bool, optional
        If ``False`` (the default), the last range bin corresponds to the ``<bins>`` index.
        If ``True``, the first range bin corresponds to the ``<bins>`` index.

    Returns
    -------
    ds : `xarray.Dataset`
        xarray dataset with the range bins above the specified bin.

    """
    # Set range to last position (otherwise BUG!)
    ds = ds.transpose(..., "range")

    # Identify vertical variables and coordinates
    vertical_variables = get_vertical_coords_and_vars(ds)

    # Set default new_range_size
    if new_range_size is None:
        new_range_size = len(ds["range"])

    # Get DataArray prototype
    # - The DataArray is subsetted to have only spatial and vertical dimensions
    dst_data_mask = get_vertical_datarray_prototype(ds, fill_value=True)

    # Ensure dataset into memory
    # - Currently dask does not support array1[mask1] = array2[mask2]
    ds = ds.compute()

    # Get bin DataArray
    # - Invalid bin values are set to max available range
    # - Radar gates will be masked out at the end using da_mask
    # - If strict=True, mask out the bin values = 1
    da_bin, da_mask = get_bin_dataarray(ds, bins=bins, mask_first_bin=strict, mask_last_bin=False)

    # Identify mask with True values where equal or above bin
    src_data_mask = ds["range"] < da_bin if strict else ds["range"] <= da_bin
    src_data_mask = src_data_mask.transpose(*dst_data_mask.dims)

    # Identify regions where to move the L1B data
    # - Define new 'range_index' to account for "range" coordinate not starting at 1 !
    n_range = len(ds["range"])
    dst_first_valid_index = n_range - src_data_mask.sum(dim="range") + 1
    dst_first_valid_index = dst_first_valid_index.astype(int)

    dst_data_mask = dst_data_mask.assign_coords({"range_index": ("range", np.arange(1, n_range + 1))})
    dst_data_mask = dst_data_mask.where(dst_data_mask["range_index"] >= dst_first_valid_index, False)
    dst_data_mask = dst_data_mask.astype(bool)

    # DEBUG
    # src_data_mask.plot.imshow(y="range", origin="upper"); plt.show()
    # dst_data_mask.plot.imshow(y="range", origin="upper"); plt.show()

    # assert dst_data_mask.data.sum() == src_data_mask.data.sum()

    # Shift variable with vertical dimension to start from specified bin
    # - Must ensure that mask is applied with same dimension order !
    ds_new = ds.copy()
    for var in vertical_variables:
        ds_new[var] = xr.full_like(ds[var], fill_value=np.nan)
        ds_new[var].data[dst_data_mask.broadcast_like(ds[var]).data] = ds[var].data[
            src_data_mask.broadcast_like(ds[var]).data
        ]
        # Mask spatial locations with invalid bins
        ds_new[var] = ds_new[var].where(~da_mask)

    # Shorten the range dimension of the dataset
    ds_new = _shorten_dataset_above(ds_new, new_range_size)
    new_range_size = len(ds_new["range"])

    # Update bin variables
    ds_new = _update_bin_variables_above(ds_new, da_bin=da_bin, da_mask=da_mask, new_range_size=new_range_size)

    # Add new range and gpm_range_id coordinates
    ds_new = _add_new_range_coords(ds_new, new_range_size=new_range_size)

    if reverse:
        ds_new = reverse_range(ds_new)

    return ds_new


@xr_squeeze_unsqueeze
def extract_dataset_below_bin(ds, bins, new_range_size=None, strict=False, reverse=False):
    """
    Extract a radar dataset with the range bins below the ``<bins>`` index.

    If ``reverse=False``, the new first range bin corresponds to the ``<bins>`` index.
    If ``reverse=True``, the last range bin corresponds to the ``<bins>`` index.

    Parameters
    ----------
    ds : `xarray.Dataset`
        GPM RADAR xarray dataset.
    bins : str
        The variable name containing the radar gate bin index of interest.
        GPM bin variables are assumed to start at 1, not 0!
    new_range_size : int, optional
        If specified, the size of the new range dimension.
        The dataset is shortened along the range dimension (from the top).
        The default is ``None``.
    strict: bool, optional
        If ``True``, it extract only radar gates above the bin index.
        If ``False``, it extract also the radar gate at the bin index.
        The default is `False`.
    reverse: bool, optional
        If ``False`` (the default), the new first range bin corresponds to the ``<bins>`` index.
        If ```True``, the last range bin corresponds to the ``<bins>`` index.

    Returns
    -------
    ds : `xarray.Dataset`
        xarray dataset with the range bins below the specified bin.

    """
    # Set range to last position (otherwise BUG!)
    ds = ds.transpose(..., "range")

    # Identify vertical variables and coordinates
    vertical_variables = get_vertical_coords_and_vars(ds)

    # Set default new_range_size
    if new_range_size is None:
        new_range_size = len(ds["range"])

    # Get DataArray prototype
    # - The DataArray is subsetted to have only spatial and vertical dimensions
    dst_data_mask = get_vertical_datarray_prototype(ds, fill_value=True)

    # Ensure dataset into memory
    # - Currently dask does not support array1[mask1] = array2[mask2]
    ds = ds.compute()

    # Get bin DataArray
    # - Invalid bin values are set to min available range
    # - Radar gates will be masked out at the end using da_mask
    # - If strict=True, mask out the last bin value
    da_bin, da_mask = get_bin_dataarray(ds, bins=bins, mask_first_bin=False, mask_last_bin=strict)

    # Identify mask with True values where equal or below bin
    src_data_mask = ds["range"] > da_bin if strict else ds["range"] >= da_bin
    src_data_mask = src_data_mask.transpose(*dst_data_mask.dims)

    # Identify regions where to move the L1B data
    # - Define new 'range_index' to account for "range" coordinate not starting at 1 !
    n_range = len(ds["range"])
    dst_last_valid_index = src_data_mask.sum(dim="range")
    dst_last_valid_index = dst_last_valid_index.astype(int)

    dst_data_mask = dst_data_mask.assign_coords({"range_index": ("range", np.arange(1, n_range + 1))})
    dst_data_mask = dst_data_mask.where(dst_data_mask["range_index"] <= dst_last_valid_index, False)
    dst_data_mask = dst_data_mask.astype(bool)

    # DEBUG
    # src_data_mask.plot.imshow(x="range")
    # dst_data_mask.plot.imshow(x="range")
    # assert dst_data_mask.data.sum() == src_data_mask.data.sum()

    # Shift variable with vertical dimension to start from specified bin
    # - Must ensure that mask is applied with same dimension order !
    ds_new = ds.copy()
    for var in vertical_variables:
        ds_new[var] = xr.full_like(ds[var], fill_value=np.nan)
        ds_new[var].data[dst_data_mask.broadcast_like(ds[var]).data] = ds[var].data[
            src_data_mask.broadcast_like(ds[var]).data
        ]
        # Mask spatial locations with invalid bins
        ds_new[var] = ds_new[var].where(~da_mask)

    # Shorten the range dimension of the dataset
    ds_new = _shorten_dataset_below(ds_new, new_range_size)
    new_range_size = len(ds_new["range"])

    # Update bin variables
    ds_new = _update_bin_variables_below(ds_new, da_bin=da_bin, da_mask=da_mask, new_range_size=new_range_size)

    # Add new range and gpm_range_id coordinates
    ds_new = _add_new_range_coords(ds_new, new_range_size=new_range_size)

    # Reverse if asked
    if reverse:
        ds_new = reverse_range(ds_new)

    return ds_new


def _check_l2_range_size(ds, new_range_size):
    """Define default range size of L2 RADAR products."""
    if new_range_size is None:
        scan_mode = ds.attrs["ScanMode"]
        if scan_mode in ["FS", "NS", "MS"]:
            new_range_size = 176
        if scan_mode in ["HS"]:
            new_range_size = 88
    return new_range_size


def extract_l2_dataset(
    ds,
    bin_ellipsoid="binEllipsoid",
    shortened_range=True,
    new_range_size=None,
):
    """
    Returns the radar dataset with the last range bin corresponding to the ellipsoid (as in L2 products).

    After extraction, 'echoLowResBinNumber' and 'echoHighResBinNumber' make no sense anymore.
    Retrieve 'sampling_type' before extraction !

    Parameters
    ----------
    ds : `xarray.Dataset`
        GPM RADAR L1B xarray dataset.
    bin_ellipsoid : str, optional
        The variable name containing the bin index of the ellipsoid.
        The default is ``binEllipsoid``.
    shortened_range : bool, optional
        Whether to shorten the range dimension of the dataset.
        This procedure is applied to generate the L2 products.
        The default is ``True``.
        Note that the range is also shortened if ``new_range_size`` is specified.
    new_range_size : int, optional
        The size of the new range dimension.
        If ``shortened_range=True`` and ``new_range_size=None``,
        ``new_range_size``takes the default values employed by the L2 PRE module.
        The default values are ``176`` for Ku and ``88`` for Ka.
        The default is ``None``.

    Returns
    -------
    ds : `xarray.Dataset`
        xarray dataset with the last range bin corresponding to the ellipsoid.

    """
    # Define new_range_size if None or shortened_range=True
    if shortened_range or new_range_size is not None:
        new_range_size = _check_l2_range_size(ds, new_range_size)

    # Extract L2 dataset
    ds_new = extract_dataset_above_bin(ds, bins=bin_ellipsoid, new_range_size=new_range_size, strict=False)

    # Drop not meaningful variables
    for var in ["echoLowResBinNumber", "echoHighResBinNumber"]:
        if var in ds_new:
            ds_new = ds_new.drop_vars(var)
    return ds_new
