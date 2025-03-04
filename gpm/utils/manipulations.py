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
import xoak  # noqa (accessor)

from gpm.checks import (
    check_has_vertical_dim,
    get_spatial_dimensions,
    get_vertical_variables,
    has_spatial_dim,
    has_vertical_dim,
    is_grid,
)
from gpm.utils.decorators import check_is_gpm_object, check_software_availability
from gpm.utils.geospatial import get_geodesic_line, get_great_circle_arc_endpoints
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


def conversion_factors_degree_to_meter(latitude, earth_radius=None):
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
    if earth_radius is None:
        earth_radius = 6378137
        # TODO: retrieve as function of latitude?

    # Calculate the conversion factor for latitude (constant per degree latitude)
    cy = np.pi * earth_radius / 180.0

    # Calculate the conversion factor for longitude (changes with latitude)
    cx = cy * np.cos(np.deg2rad(latitude))

    return cx, cy


####---------------------------------------------------------------------------.
####################
#### Subsetting ####
####################


def crop_around_valid_data(xr_obj, variable=None):
    """
    Return a sub-region of the specified DataArray containing all the non-NaN values.

    Parameters
    ----------
    xr_obj: xarray.DataArray or xarray.Dataset
        A xarray object to crop around valid data (of variable).
    variable : str, optional
        Name of the variable to use to crop the dataset.
        Only to be specified if xr_obj is a xr.Dataset

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Cropped DataArray such that NaN-only outer rows/columns are removed.
    """
    da = get_xarray_variable(xr_obj, variable=variable)

    # Create a boolean mask indicating where da is not NaN
    valid_mask = da.notnull().compute()

    # Raise error if not valid data
    if not np.any(valid_mask).item():
        raise ValueError("No valid data around with to crop.")

    # Define the isel dictionary
    isel_dict = {}
    for dim in da.dims:
        # Collapse all other dims and get a 1D boolean array along 'dim'
        other_dims = [d for d in da.dims if d != dim]
        valid_along_dim = valid_mask.any(dim=other_dims)
        # Find first and last True index
        # - argmax() gives the first True index from the start
        start_idx = int(np.argmax(valid_along_dim.data))
        # To get the last True, reverse the array and use argmax again
        end_idx = len(valid_along_dim) - int(np.argmax(valid_along_dim.data[::-1]))
        # Construct the slice
        isel_dict[dim] = slice(start_idx, end_idx)

    # Apply the slice to the original object (Dataset or DataArray)
    return xr_obj.isel(isel_dict, drop=False)


####---------------------------------------------------------------------------.
##########################
#### Range bin slicer ####
##########################


def get_bin_dataarray(xr_obj, bins, mask_first_bin=False, mask_last_bin=False, fillvalue=None):
    """Get bin xarray.DataArray."""
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
    """Return a valid bin xarray.DataArray with a mask for the invalid/unavailable bins."""
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

    ``bin_variable`` can be a bin xarray.DataArray or the name of a bin variable of the input xarray.Dataset.

    The function extract the gates based on the 'range' coordinate values.
    Bin values are assumed to start at 1, not 0 !

    If you want to extract a slice at a single range bin, use instead ``xr_obj.sel(range=range_bin_value)``.

    Parameters
    ----------
    xr_obj : xarray.DataArray or xarray.Dataset
        xarray object with the 'range' dimension (and coordinate).
    bins : str or xarray.DataArray
        Either a xarray.DataArray or a string pointing to the dataset variable with the range bins to extract.
        Bin values are assumed to start at 1, not 0 !

    Returns
    -------
    xr_out : xarray.Dataset or xarray.DataArray
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
    """Retrieve index along the range dimension where the xarray.DataArray values is closest to value."""
    da_diff = np.abs(da - value)
    idx, mask_all_nan = get_range_index_at_min(da_diff)
    return idx, mask_all_nan


def get_range_index_at_min(da):
    """Retrieve index along the range dimension where the xarray.DataArray has minimum values."""
    # Retrieve vertical dimension
    vertical_dim = _get_vertical_dim(da)
    # Put DataArray in memory
    da = da.compute()
    # Retrieve mask where all values are NaN
    mask_all_nan = np.isnan(da).all(dim=vertical_dim)
    # Add dummy 0 value where all NaN values to avoid 'All-NaN slice encountered' error in da.argmax(dim=vertical_dim)
    da = da.where(~mask_all_nan, 0)
    # Retrieve range index
    idx = da.argmin(dim=vertical_dim)
    return idx, mask_all_nan


def get_range_index_at_max(da):
    """Retrieve index along the range dimension where the xarray.DataArray has maximum values."""
    # Retrieve vertical dimension
    vertical_dim = _get_vertical_dim(da)
    # Put DataArray in memory
    da = da.compute()
    # Retrieve mask where all values are NaN
    mask_all_nan = np.isnan(da).all(dim=vertical_dim)
    # Add dummy 0 value where all NaN values to avoid 'All-NaN slice encountered' error in da.argmax(dim=vertical_dim)
    da = da.where(~mask_all_nan, 0)
    # Retrieve range index
    idx = da.argmax(dim=vertical_dim)
    return idx, mask_all_nan


def slice_range_at_value(xr_obj, value, variable=None):
    """Slice the 3D arrays where the variable values are close to value."""
    da = get_xarray_variable(xr_obj, variable=variable)
    vertical_dim = _get_vertical_dim(da)
    idx, mask_all_nan = get_range_index_at_value(da=da, value=value)
    xr_obj_sliced = xr_obj.isel({vertical_dim: idx})
    return xr_obj_sliced.where(~mask_all_nan)


def slice_range_at_max_value(xr_obj, variable=None):
    """Slice the 3D arrays where the variable values are at maximum."""
    da = get_xarray_variable(xr_obj, variable=variable)
    vertical_dim = _get_vertical_dim(da)
    idx, mask_all_nan = get_range_index_at_max(da=da)
    xr_obj_sliced = xr_obj.isel({vertical_dim: idx})
    return xr_obj_sliced.where(~mask_all_nan)


def slice_range_at_min_value(xr_obj, variable=None):
    """Slice the 3D arrays where the variable values are at minimum."""
    da = get_xarray_variable(xr_obj, variable=variable)
    vertical_dim = _get_vertical_dim(da)
    idx, mask_all_nan = get_range_index_at_min(da=da)
    xr_obj_sliced = xr_obj.isel({vertical_dim: idx})
    return xr_obj_sliced.where(~mask_all_nan)


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

    # Identify which range indices has valid values
    range_has_valid_values = ~np.isnan(da).all(dim=dims)
    valid_range_indices = np.where(range_has_valid_values > 0)[0]

    # Check there are valid data
    if valid_range_indices.size == 0:
        raise ValueError(f"No valid data for variable {variable}.")

    # Identify first and last True occurrence
    first_index = valid_range_indices[0]
    last_index = valid_range_indices[-1]

    # Return isel dictionary
    isel_dict = {vertical_dim: slice(first_index, last_index + 1)}
    return isel_dict


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

    # Identify which range indices has values falling in the desired interval
    range_has_values_within_interval = np.logical_and(da >= vmin, da <= vmax).sum(dim=dims)
    valid_range_indices = np.where(range_has_values_within_interval > 0)[0]

    # Check there are data for requested value interval
    if valid_range_indices.size == 0:
        raise ValueError(f"No data within the requested value interval for variable {variable}.")

    # Identify first and last True occurrence
    first_index = valid_range_indices[0]
    last_index = valid_range_indices[-1]

    # Return isel dictionary
    isel_dict = {vertical_dim: slice(first_index, last_index + 1)}
    return isel_dict


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
        raise ValueError("Expecting an xarray object with the 'height' coordinate.")
    return da_height


def slice_range_at_height(xr_obj, value):
    """Slice the 3D array at a given height."""
    da_heigth = get_height_dataarray(xr_obj)
    vertical_dim = _get_vertical_dim(xr_obj)
    idx, mask_all_nan = get_range_index_at_value(da_heigth, value=value)
    xr_obj_sliced = xr_obj.isel({vertical_dim: idx})
    return xr_obj_sliced.where(~mask_all_nan)


def get_height_at_temperature(da_height, da_temperature, temperature):
    """Retrieve height at a specific temperature."""
    vertical_dim = _get_vertical_dim(da_height)
    idx_desired_temperature, mask_all_nan = get_range_index_at_value(da_temperature, temperature)
    xr_obj_sliced = da_height.isel({vertical_dim: idx_desired_temperature})
    return xr_obj_sliced.where(~mask_all_nan)


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
    """Return xarray.Dataset with only 3D spatial variables."""
    from gpm.checks import get_spatial_3d_variables

    variables = get_spatial_3d_variables(ds, strict=strict, squeeze=squeeze)
    return ds[variables]


def select_spatial_2d_variables(ds, strict=False, squeeze=True):
    """Return xarray.Dataset with only 2D spatial variables."""
    from gpm.checks import get_spatial_2d_variables

    variables = get_spatial_2d_variables(ds, strict=strict, squeeze=squeeze)
    return ds[variables]


def select_cross_section_variables(ds, strict=False, squeeze=True):
    """Return xarray.Dataset with only cross-section variables.

    It select variables with only a single horizontal and vertical dimension.
    """
    from gpm.checks import get_cross_section_variables

    variables = get_cross_section_variables(ds, strict=strict, squeeze=squeeze)
    return ds[variables]


def select_vertical_variables(ds):
    """Return xarray.Dataset with only variables with vertical dimension."""
    variables = get_vertical_variables(ds)
    return ds[variables]


def select_frequency_variables(ds):
    """Return xarray.Dataset with only multifrequency variables."""
    from gpm.checks import get_frequency_variables

    variables = get_frequency_variables(ds)
    return ds[variables]


def select_bin_variables(ds):
    """Return xarray.Dataset with only bin variables."""
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


def get_spatial_2d_datarray_template(ds, fill_value=np.nan):
    """Get spatial 2D DataArray template."""
    possible_variables = ds.gpm.spatial_2d_variables
    if len(possible_variables) == 0:
        raise ValueError("No spatial 2D variables available.")
    da = ds[possible_variables[0]]
    isel_dict = {dim: 0 for dim in da.dims if dim not in ds.gpm.spatial_dimensions}
    da = xr.full_like(da.isel(isel_dict), fill_value=fill_value)
    da.name = "spatial_2d_template"
    return da


def get_spatial_3d_datarray_template(ds, fill_value=np.nan):
    """Get spatial 3D DataArray template."""
    # TODO: rename get_vertical_datarray_prototype --> get_spatial_3d_template
    possible_variables = ds.gpm.spatial_3d_variables
    if len(possible_variables) == 0:
        raise ValueError("No spatial 3D variables available.")
    da = ds[possible_variables[0]]
    isel_dict = {dim: 0 for dim in da.dims if dim not in (ds.gpm.spatial_dimensions + ds.gpm.vertical_dimension)}
    da = xr.full_like(da.isel(isel_dict), fill_value=fill_value)
    da.name = "spatial_3d_template"
    return da


def get_vertical_coords_and_vars(ds):
    """Return a 'prototype' with only spatial and vertical dimensions."""
    vertical_variables = get_vertical_variables(ds)
    vertical_coords = [coord for coord in ds.coords if has_vertical_dim(ds[coord]) and has_spatial_dim(ds[coord])]
    vertical_variables += vertical_coords
    if len(vertical_variables) == 0:
        raise ValueError("No vertical variables to extract.")
    return vertical_variables


def get_vertical_datarray_prototype(ds, fill_value=np.nan):
    """Return a xarray.DataArray 'prototype' with only spatial and vertical dimensions."""
    vertical_variables = get_vertical_coords_and_vars(ds)
    da = ds[vertical_variables[0]]
    da = xr.full_like(da, fill_value=fill_value).compute()
    da.name = "prototype"
    return ensure_vertical_datarray_prototype(da)


def ensure_vertical_datarray_prototype(da):
    """Return a xarray.DataArray with only spatial and vertical dimensions."""
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
    ds : xarray.Dataset
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
    ds : xarray.Dataset
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
    ds : xarray.Dataset
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
    ds : xarray.Dataset
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
    ds : xarray.Dataset
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
    ds : xarray.Dataset
        xarray dataset with the last range bin corresponding to the ellipsoid.

    """
    # Define new_range_size if None or shortened_range=True
    if shortened_range or new_range_size is not None:
        new_range_size = _check_l2_range_size(ds, new_range_size)

    # Extract L2 dataset
    ds_new = extract_dataset_above_bin(ds, bins=bin_ellipsoid, new_range_size=new_range_size, strict=False)

    # Drop non meaningful variables
    for var in ["echoLowResBinNumber", "echoHighResBinNumber"]:
        if var in ds_new:
            ds_new = ds_new.drop_vars(var)
    return ds_new


####------------------------------------------------------------------------------------------------------------------.
#############################
#### Mask below/above bin ###
#############################


def mask_vertical_variables(ds, mask, fillvalue):
    vertical_variables = get_vertical_variables(ds)  # do not include coords !
    ds = ds.copy()
    for var in vertical_variables:
        ds[var] = ds[var].where(mask, fillvalue)
    return ds


def mask_above_bin(xr_obj, bins, strict=True, fillvalue=np.nan):
    """
    Mask the xarray object below the ``<bins>`` index.

    The method does not mask where bins values are NaN or invalid.

    Parameters
    ----------
    xr_obj : xarray.Dataset or xarray.DataArray
        GPM RADAR xarray object.
    bins : str or xarray.DataArray
        Either a xarray.DataArray or a string pointing to the dataset variable
        with the range bins above which to mask.
        GPM bin variables are assumed to start at 1, not 0!
    strict: bool, optional
        If ``False``, it masks only radar gates above the bin index.
        If ``True``, it masks also the radar gate at the bin index.
        The default is `True`.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Masked GPM RADAR xarray object.

    """
    # Get bin DataArray
    # - Invalid bin values are set to the minimum available range (to not mask)
    da_bin, _ = get_bin_dataarray(xr_obj, bins=bins, fillvalue=xr_obj["range"].data.min())
    # Define mask
    mask = xr_obj["range"] > da_bin if strict else xr_obj["range"] >= da_bin
    # If DataArray, mask the DataArray
    if isinstance(xr_obj, xr.DataArray):
        return xr_obj.where(mask, fillvalue)
    # If Dataset, mask vertical variables (do not mask coordinates !)
    return mask_vertical_variables(xr_obj, mask=mask, fillvalue=fillvalue)


def mask_below_bin(xr_obj, bins, strict=True, fillvalue=np.nan):
    """
    Mask the xarray object below the ``<bins>`` index.

    The method does not mask where bins values are NaN or invalid.

    Parameters
    ----------
    xr_obj : xarray.Dataset or xarray.DataArray
        GPM RADAR xarray object.
    bins : str or xarray.DataArray
        Either a xarray.DataArray or a string pointing to the dataset variable
        with the range bins below which to mask.
        GPM bin variables are assumed to start at 1, not 0!
    strict: bool, optional
        If ``False``, it masks only radar gates below the bin index.
        If ``True``, it masks also the radar gate at the bin index.
        The default is `True`.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Masked GPM RADAR xarray object.

    """
    # Get bin DataArray
    # - Invalid bin values are set to the maximum available range (to not mask)
    da_bin, _ = get_bin_dataarray(xr_obj, bins=bins, fillvalue=xr_obj["range"].data.max())
    # Define mask
    mask = xr_obj["range"] < da_bin if strict else xr_obj["range"] <= da_bin
    # If DataArray, mask the DataArray
    if isinstance(xr_obj, xr.DataArray):
        return xr_obj.where(mask, fillvalue)
    # If Dataset, mask vertical variables (do not mask coordinates !)
    return mask_vertical_variables(xr_obj, mask=mask, fillvalue=fillvalue)


def mask_between_bins(xr_obj, bottom_bins, top_bins, strict=True, fillvalue=np.nan):
    """
    Mask the xarray object between bottom and top ``<bins>`` indices.

    The method does not mask where bins values are NaN or invalid.

    Parameters
    ----------
    xr_obj : xarray.Dataset or xarray.DataArray
        GPM RADAR xarray object.
    bottom_bins : str or xarray.DataArray
        Either a xarray.DataArray or a string pointing to the dataset variable
        with the bottom range bins.
        GPM bin variables are assumed to start at 1, not 0!
    top_bins : str or xarray.DataArray
        Either a xarray.DataArray or a string pointing to the dataset variable
        with the top range bins.
        GPM bin variables are assumed to start at 1, not 0!
    strict: bool, optional
        If ``False``, it masks only radar gates between the bin indices.
        If ``True``, it masks also the radar gates at the bin indices.
        The default is `True`.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Masked GPM RADAR xarray object.

    """
    xr_obj = xr_obj.copy()
    da_bottom_bin, _ = get_bin_dataarray(xr_obj, bins=bottom_bins, fillvalue=xr_obj["range"].data.min())
    da_top_bin, _ = get_bin_dataarray(xr_obj, bins=top_bins, fillvalue=xr_obj["range"].data.max())
    # Define mask
    mask = (
        (xr_obj["range"] > da_bottom_bin) | (xr_obj["range"] < da_top_bin)
        if strict
        else (xr_obj["range"] >= da_bottom_bin) | (xr_obj["range"] <= da_top_bin)
    )
    # If DataArray, mask the DataArray
    if isinstance(xr_obj, xr.DataArray):
        return xr_obj.where(mask, fillvalue)
    # If Dataset, mask vertical variables (do not mask coordinates !)
    return mask_vertical_variables(xr_obj, mask=mask, fillvalue=fillvalue)


####------------------------------------------------------------------------------------------------------------------.
#############################
#### Transect Extraction ####
#############################


@check_is_gpm_object
def extract_at_points(xr_obj, points, method="nearest", new_dim="points"):
    """Extract values at a set of points.

    This routine is useful particularly useful to extract values observed close
    to meteorological stations or along a trajectory.

    You could also exploit this function to "nearest-neighbour" remapping values to another
    2D grid/orbit if you stack such object, pass the coordinates to this function and then unstack.
    However for this last application, it is better to use the `remap` function.

    Parameters
    ----------
    xr_obj: xarray.DataArray or xarray.Dataset
        Dataset or DataArray from which to extract values at points.
    points: numpy.ndarray
        An array of shape (N, 2) with the lon, lat points at which to interpolate the data.
    method: str, optional
        The interpolation method. The default method is ``'nearest'``.
        If input data have 2D-coordinates, only ``'nearest'`` method is implemented.
        If input data have 1D-coordinates,  See :py:class:`xarray.DataArray.interp` for other methods.
    new_dim: str, optional
        The name of the new points dimension. Defaults to "points".

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The values at the specified points.
    """
    x, y = xr_obj.gpm.spatial_coordinates

    # Grid case
    if is_grid(xr_obj):
        # Regular grid
        xr_obj_sliced = xr_obj.interp(
            {
                x: xr.DataArray(points[:, 0], dims=new_dim),
                y: xr.DataArray(points[:, 1], dims=new_dim),
            },
            method=method,
        )
        return xr_obj_sliced

    # Orbit case
    # - Use sklearn_geo_balltree to exploit haversine distance (kdtree does not support haversine distance)
    # - Not tested for cases at the antimeridian !
    xr_obj.xoak.set_index([y, x], index_type="sklearn_geo_balltree")

    xr_obj_slice = xr_obj.xoak.sel(
        {
            x: xr.DataArray(points[:, 0], dims=new_dim),
            y: xr.DataArray(points[:, 1], dims=new_dim),
        },
    )
    return xr_obj_slice


@check_is_gpm_object
def extract_transect_at_points(xr_obj, points, method="linear", new_dim="transect"):
    """Obtain an transect through a series of points.

    It allows to extract data along a custom curvilinear track / trajectory.

    Parameters
    ----------
    xr_obj: xarray.DataArray or xarray.Dataset
        Dataset or DataArray from which extract a transect.
    points: numpy.ndarray
        An array of shape (N, 2) with the lon, lat points at which to interpolate the data.
    method: str, optional
        The interpolation method, either ``'linear'`` or ``'nearest'``.
        If input data have 2D-coordinates, only ``'nearest'`` method is implemented.
        If input data have 1D-coordinates, the default method is ``'linear'``.
        See :py:class:`xarray.DataArray.interp` for other methods.
    new_dim: str, optional
        The name of the new transect dimension. Defaults to "transect".

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The transect object, with the ``new_dim`` dimension (of size N).

    See Also
    --------
    :py:class:`gpm.utils.manipulations.extract_transect_between_points` and
    :py:class:`gpm.utils.manipulations.extract_transect_around_point`.

    """
    return extract_at_points(xr_obj, points=points, method=method, new_dim=new_dim)


@check_is_gpm_object
@check_software_availability(software="sklearn", conda_package="scikit-learn")
def extract_transect_between_points(xr_obj, start_point, end_point, steps=100, method="linear", new_dim="transect"):
    """Extract an interpolated transect between two points on a sphere.

    Parameters
    ----------
    xr_obj: xarray.DataArray or xarray.Dataset
        Dataset or DataArray from which extract a transect.
    start_point: tuple
        A longitude-latitude pair designating the start point of the cross section (units are
        degrees east and degrees north).
    end_point: tuple
        A longitude-latitude pair designating the end point of the cross section (units are
        degrees east and degrees north).
    steps: int, optional
        The number of points along the geodesic between the start and the end point
        (including the end points) to use in the cross section. Defaults to 100.
    method: str, optional
        The interpolation method, either ``'linear'`` or ``'nearest'``.
        If input data have 2D-coordinates, only ``'nearest'`` method is implemented.
        If input data have 1D-coordinates, the default method is ``'linear'``.
        See :py:class:`xarray.DataArray.interp` for other methods.
    new_dim: str, optional
        The name of the new transect dimension. Defaults to "transect".

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The transect object, with the ``new_dim`` dimension (of size ``steps``).

    See Also
    --------
    :py:class:`gpm.utils.manipulations.extract_transect_at_points` and
    :py:class:`gpm.utils.manipulations.extract_transect_around_point`.

    """
    # Get the points along the geodesic line
    points = get_geodesic_line(start_point=start_point, end_point=end_point, steps=steps)

    # Return the interpolated data
    return extract_transect_at_points(xr_obj, points=points, method=method, new_dim=new_dim)


@check_is_gpm_object
def extract_transect_around_point(xr_obj, point, azimuth, distance, steps=100, method="linear", new_dim="transect"):
    """
    Extract a transect following the great circle arc centered on the specified point.

    Parameters
    ----------
    xr_obj : xarray.DataArray or xarray.Dataset
        Dataset or DataArray from which extract a transect.
    point : tuple of float
        A tuple representing the middle point (longitude, latitude) of the great circle arc.
    azimuth : float
        The azimuth (in degrees) from the starting point. 0 correspond to the North. 180 to the South.
        The opposite direction will be automatically calculated as (azimuth + 180) % 360.
    distance : float
        The distance (in meters) to the points from the center point.
    steps: int, optional
        The number of points along the geodesic between the start and the end point
        (including the end points) to use in the cross section. Defaults to 100.
    method: str, optional
        The interpolation method, either ``'linear'`` or ``'nearest'``.
        If input data have 2D-coordinates, only ``'nearest'`` method is implemented.
        If input data have 1D-coordinates, the default method is ``'linear'``.
        See :py:class:`xarray.DataArray.interp` for other methods.
    new_dim: str, optional
        The name of the new transect dimension. Defaults to "transect".

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The transect object, with the ``new_dim`` dimension (of size ``steps``).

    See Also
    --------
    :py:class:`gpm.utils.manipulations.extract_transect_at_points` and
    :py:class:`gpm.utils.manipulations.extract_transect_between_points`.

    """
    start_point, end_point = get_great_circle_arc_endpoints(point=point, azimuth=azimuth, distance=distance)
    return extract_transect_between_points(
        xr_obj,
        start_point=start_point,
        end_point=end_point,
        steps=steps,
        method=method,
        new_dim=new_dim,
    )


def locate_points(xr_obj, points):
    """Return a list of isel dictionary corresponding to the nearest location of the set of points."""
    # Retrieve spatial dimensions
    spatial_dims = get_spatial_dimensions(xr_obj)

    # Define dummy coordinates with integer indices
    dummy_coords = {f"dummy_{d}": (d, np.arange(len(xr_obj[d]))) for d in spatial_dims}
    xr_obj = xr_obj.assign_coords(dummy_coords)

    # Identify index over which to slice
    xr_point = extract_at_points(xr_obj, points=points)
    isel_dict = [{d: xr_point[f"dummy_{d}"].data[i].item() for d in spatial_dims} for i in range(len(points))]

    # Drop dummy coordinates
    xr_obj = xr_obj.drop_vars(list(dummy_coords))

    # Return isel_dict
    return isel_dict


def define_transect_isel_dict(xr_obj, point, dim):
    """Define the isel dictionary required to extract a transect along the specified dimension."""
    # Check specified dimension
    spatial_dims = get_spatial_dimensions(xr_obj)
    if dim not in spatial_dims:
        raise ValueError(f"'dim' must be one of object spatial dimensions: {spatial_dims}.")
    if len(spatial_dims) != 2:
        raise ValueError("The object does not have 2 spatial dimensions.")

    # Define dimension over which to slice
    subset_dim = (set(spatial_dims) - {dim}).pop()

    # Identify index over which to slice
    isel_dict = locate_points(xr_obj, points=np.atleast_2d(point))[0]
    transect_isel_dict = {subset_dim: isel_dict[subset_dim]}
    return transect_isel_dict


@check_is_gpm_object
def extract_transect_along_dimension(xr_obj, point, dim):
    """
    Extract a transect along the specified spatial dimension passing through the specified location.

    Parameters
    ----------
    xr_obj : xarray.DataArray or xarray.Dataset
        Dataset or DataArray from which extract a transect.
    point : tuple of float
        A tuple representing the middle point (longitude, latitude) of the great circle arc.
    dim : str
        The desired spatial dimension of the transect.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        The transect object with spatial dimension ``dim``.
    """
    transect_isel_dict = define_transect_isel_dict(xr_obj, point=point, dim=dim)
    return xr_obj.isel(transect_isel_dict)


####------------------------------------------------------------------------------------------------------------------.
####################
#### Infilling  ####
####################


def _infill_datarray(da, da_bin, potential_infill_mask, valid_mask):
    # Create a copy of the input to avoid modifying the original
    result = da.copy(deep=True)

    # Get the values at the specified indices
    values_at_indices = da.gpm.slice_range_at_bin(da_bin)

    # Create a mask for points where both the index and value are valid
    valid_values_mask = valid_mask & ~np.isnan(values_at_indices)

    # Define infill mask
    infill_mask = potential_infill_mask & valid_values_mask

    # Broadcast values to apply
    values_broadcast = values_at_indices.broadcast_like(da)

    # Apply the infill operation
    with xr.set_options(keep_attrs=True):
        result = xr.where(infill_mask, values_broadcast, result)
        result.name = da.name
    return result


def infill_below_bin(xr_obj, bins):
    """
    Infill values below a spatially variable range bin.

    Parameters
    ----------
    xr_obj : xarray.Dataset or xarray.DataArray
        GPM RADAR xarray object.
    bins : str or xarray.DataArray
        Either a xarray.DataArray or a string pointing to the dataset variable
        with the range bins.
        GPM bin variables are assumed to start at 1, not 0!

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Infilled GPM RADAR xarray object.
    """
    check_has_vertical_dim(xr_obj)

    # Get the bin DataArray
    da_bin, da_mask = get_bin_dataarray(xr_obj, bins=bins)

    # Get height DataArray
    da_height = xr_obj["height"]

    # Get the name of the third dimension (range or height)
    z_dim = xr_obj.gpm.vertical_dimension[0]

    # Check if the range/height dimension is increasing or decreasing
    other_dims = [dim for dim in xr_obj.dims if dim != z_dim]
    pixel_isel_dict = {dim: 0 for dim in other_dims}
    z_values = da_height.isel(pixel_isel_dict, missing_dims="ignore").to_numpy()
    is_increasing = z_values[1] > z_values[0]

    # Handle NaN values in the bin array and retrieve 0-indexing
    # - Use 0 as placeholder for invalid indices
    valid_mask = ~da_mask
    idx_int = xr.where(valid_mask, da_bin, 1).astype(int) - 1

    # Create a template for our mask along the z dimension
    z_indices = xr.DataArray(np.arange(len(xr_obj[z_dim])), dims=[z_dim], coords={z_dim: xr_obj[z_dim]})

    # Define potential infilling mask
    potential_infill_mask = z_indices <= idx_int if is_increasing else z_indices >= idx_int

    # Infill DataArrays
    if isinstance(xr_obj, xr.DataArray):
        xr_obj = _infill_datarray(
            xr_obj,
            da_bin=da_bin,
            potential_infill_mask=potential_infill_mask,
            valid_mask=valid_mask,
        )
    else:
        for var in xr_obj.gpm.vertical_variables:
            xr_obj[var] = _infill_datarray(
                xr_obj[var],
                da_bin=da_bin,
                potential_infill_mask=potential_infill_mask,
                valid_mask=valid_mask,
            )

    # Re-assign height back (because current heights inherited form valid_values_mask)
    xr_obj = xr_obj.assign_coords({"height": da_height})
    return xr_obj


####------------------------------------------------------------------------------------------------------------------.
#############################
#### Location Utilities  ####
#############################


def locate_max_value(da, return_isel_dict=False):
    """Find the geographic point where the maximum value occur in the data array.

    Parameters
    ----------
    da : xarray.DataArray
        The data array to analyze.
    return_isel_dict: bool, optional
        If True, returns a dictionary with the spatial dimension indices corresponding to the maximum value.
        If False (the default), returns a (lon, lat) tuple of the point where the maximum value occurs.

    Returns
    -------
    tuple or dict
        If return_isel_dict=True, returns a dictionary
        with the spatial dimension and indices corresponding to the maximum value.
        If return_isel_dict=False (the default), returns a (lon, lat) tuple
        of the point where the maximum value occurs.
    """
    isel_dict = _get_max_value_spatial_isel_dict(da)
    if return_isel_dict:
        return isel_dict

    da_point = da.isel(isel_dict)
    point = (da_point[da.gpm.x].values.item(), da_point[da.gpm.y].values.item())
    return point


def locate_min_value(da, return_isel_dict=False):
    """
    Find the geographic point where the minimum value occurs in the data array.

    Parameters
    ----------
    da : xarray.DataArray
        The data array to analyze.

    return_isel_dict: bool, optional
        If True, returns a dictionary with the spatial dimension indices corresponding to the minimum value.
        If False (the default), returns a (lon, lat) tuple of the point where the minimum value occurs.

    Returns
    -------
    tuple or dict
        If return_isel_dict=True, returns a dictionary
        with the spatial dimension and indices corresponding to the minimum value.
        If return_isel_dict=False (the default), returns a (lon, lat) tuple
        of the point where the minimum value occurs.

    """
    isel_dict = _get_min_value_spatial_isel_dict(da)
    if return_isel_dict:
        return isel_dict
    da_point = da.isel(isel_dict)
    point = (da_point[da.gpm.x].values.item(), da_point[da.gpm.y].values.item())
    return point


def _get_max_value_isel_dict(da):
    """Find the dimension indices where the maximum value occur in the data array..

    Parameters
    ----------
    da : xarray.DataArray
        The data array to analyze.

    Returns
    -------
    dict
        A dictionary with dimension names as keys and indices where the maximum value occurs as values.
    """
    da = da.compute()
    dict_argmax = da.argmax(da.dims)
    isel_dict = {k: v.data.item() for k, v in dict_argmax.items()}
    return isel_dict


def _get_max_value_spatial_isel_dict(da):
    """Find the spatial dimensions indices where the maximum value occur in the data array.

    Parameters
    ----------
    da : xarray.DataArray
        The data array to analyze.

    Returns
    -------
    dict
        A dictionary with spatial dimension names as keys and indices where the maximum value occurs as values.
    """
    isel_dict = _get_max_value_isel_dict(da)
    spatial_dims = da.gpm.spatial_dimensions
    isel_dict = {k: isel_dict[k] for k in spatial_dims}
    return isel_dict


def _get_min_value_isel_dict(da):
    """
    Find the dimension indices where the minimum value occurs in the data array.

    Parameters
    ----------
    da : xarray.DataArray
        The data array to analyze.

    Returns
    -------
    dict
        A dictionary with dimension names as keys and indices where the minimum value occurs as values.
    """
    da = da.compute()
    dict_argmin = da.argmin(da.dims)
    isel_dict = {k: v.data.item() for k, v in dict_argmin.items()}
    return isel_dict


def _get_min_value_spatial_isel_dict(da):
    """
    Find the spatial dimension indices where the minimum value occurs in the data array.

    Parameters
    ----------
    da : xarray.DataArray
        The data array to analyze.

    Returns
    -------
    dict
        A dictionary with spatial dimension names as keys and indices where the minimum value occurs as values.
    """
    isel_dict = _get_min_value_isel_dict(da)
    spatial_dims = da.gpm.spatial_dimensions
    isel_dict = {k: isel_dict[k] for k in spatial_dims}
    return isel_dict
