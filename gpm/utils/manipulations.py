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
    Output data will have unit kg/m² if scale_factor=1000

    height a list or array of corresponding heights for each level.
    """
    if scale_factor is not None and units is None:
        raise ValueError("Specify output 'units' when the scale_factor is applied.")
    # Compute integrated value
    data = dataarray.data.copy()
    vertical_dim = _get_vertical_dim(dataarray)
    heights = np.asanyarray(dataarray[vertical_dim].data)
    output = _integrate_concentration(data, heights)
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


def check_variable_availabilty(ds, variable, argname):
    if variable not in ds:
        raise ValueError(
            f"{variable} is not a variable of the xr.Dataset. Invalid {argname} argument.",
        )


def _get_bin_datarray(ds, bin):
    """Get bin dataarray."""
    if not isinstance(bin, (str, xr.DataArray)):
        raise TypeError("'bin' must be a DataArray or a string indicating the Dataset variable.")
    if isinstance(bin, xr.DataArray):
        return bin
    check_variable_availabilty(ds, bin, argname="bin")
    return ds[bin]


def get_variable_dataarray(xr_obj, variable):
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
    # TODO: slice_range_at_bin

    # Get variable dataarray
    da_variable = get_variable_dataarray(xr_obj, variable)
    vertical_dim = _get_vertical_dim(da_variable)
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
    da = da_variable.isel({vertical_dim: da_bin})
    # Mask values at nan bins
    da = da.where(~da_is_nan)
    # Set original chunks if input DataArray is dask-backed
    # - This avoid later need of unify_chunks when converting to dataframe
    if hasattr(da_variable.data, "chunks"):
        da = da.chunk(da_variable.chunks)
    return da


def get_height_at_bin(xr_obj, bin):
    return get_variable_at_bin(xr_obj, bin, variable="height")


def get_range_slices_with_valid_data(xr_obj, variable=None):
    """Get the vertical ('range'/'height') slices with valid data."""
    # TODO: maybe add option for minimum_number of valid_data !

    # Extract DataArray
    da = get_variable_dataarray(xr_obj, variable)
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


def slice_range_with_valid_data(xr_obj, variable=None):
    """Select the 'range' interval with valid data."""
    isel_dict = get_range_slices_with_valid_data(xr_obj, variable=variable)
    # Susbet the xarray object
    return xr_obj.isel(isel_dict)


def get_range_slices_within_values(xr_obj, variable=None, vmin=-np.inf, vmax=np.inf):
    """Get the 'range' slices with data within a given data interval."""
    # Extract DataArray
    da = get_variable_dataarray(xr_obj, variable)
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


def slice_range_where_values(xr_obj, variable=None, vmin=-np.inf, vmax=np.inf):
    """Select the 'range' interval where values are within the [vmin, vmax] interval."""
    isel_dict = get_range_slices_within_values(xr_obj, variable=variable, vmin=vmin, vmax=vmax)
    return xr_obj.isel(isel_dict)


def get_range_index_at_value(da, value):
    """Retrieve index along the range dimension where the DataArray values is closest to value."""
    vertical_dim = _get_vertical_dim(da)
    return np.abs(da - value).argmin(dim=vertical_dim).compute()


def get_range_index_at_min(da):
    """Retrieve index along the range dimension where the DataArray has minimum values."""
    vertical_dim = _get_vertical_dim(da)
    return da.argmin(dim=vertical_dim).compute()


def get_range_index_at_max(da):
    """Retrieve index along the range dimension where the DataArray has maximum values."""
    vertical_dim = _get_vertical_dim(da)
    return da.argmax(dim=vertical_dim).compute()


def slice_range_at_value(xr_obj, value, variable=None):
    """Slice the 3D arrays where the variable values are close to value."""
    da = get_variable_dataarray(xr_obj, variable=variable)
    vertical_dim = _get_vertical_dim(da)
    idx = get_range_index_at_value(da=da, value=value)
    return xr_obj.isel({vertical_dim: idx})


def slice_range_at_max_value(xr_obj, variable=None):
    """Slice the 3D arrays where the variable values are at maximum."""
    da = get_variable_dataarray(xr_obj, variable=variable)
    vertical_dim = _get_vertical_dim(da)
    idx = get_range_index_at_max(da=da)
    return xr_obj.isel({vertical_dim: idx})


def slice_range_at_min_value(xr_obj, variable=None):
    """Slice the 3D arrays where the variable values are at minimum."""
    da = get_variable_dataarray(xr_obj, variable=variable)
    vertical_dim = _get_vertical_dim(da)
    idx = get_range_index_at_min(da=da)
    return xr_obj.isel({vertical_dim: idx})


def slice_range_at_temperature(ds, temperature, variable_temperature="airTemperature"):
    """Slice the 3D arrays along a specific isotherm."""
    return slice_range_at_value(ds, variable=variable_temperature, value=temperature)


def slice_range_at_height(xr_obj, height):
    """Slice the 3D array at a given height."""
    return slice_range_at_value(xr_obj, variable="height", value=height)


def get_height_at_temperature(da_height, da_temperature, temperature):
    """Retrieve height at a specific temperature."""
    vertical_dim = _get_vertical_dim(da_height)
    idx_desired_temperature = get_range_index_at_value(da_temperature, temperature)
    return da_height.isel({vertical_dim: idx_desired_temperature})


def get_range_axis(da):
    """Get range dimension axis index."""
    vertical_dim = _get_vertical_dim(da)
    return np.where(np.isin(list(da.dims), vertical_dim))[0].item()


def get_dims_without(da, dims):
    """Remove specified 'dims' for list of DataArray dimensions."""
    data_dims = np.array(list(da.dims))
    return data_dims[np.isin(data_dims, dims, invert=True)].tolist()


def get_xr_shape(xr_obj, dims):
    """Get xarray shape for specific dimensions."""
    dims_dict = xr_obj.sizes
    return [dims_dict[key] for key in dims]


def create_bin_idx_data_array(xr_obj):
    """Create a 3D DataArray with the bin index along the range dimension.

    The GPM bin index start at 1 !
    GPM bin index is equivalent to gpm_range_id + 1
    """
    vertical_dim = _get_vertical_dim(xr_obj)
    dims = ["cross_track", "along_track", vertical_dim]
    shape = get_xr_shape(xr_obj, dims=dims)
    bin_start = xr_obj["gpm_range_id"][0]
    bin_end = xr_obj["gpm_range_id"][-1]
    idx_bin = np.arange(bin_start + 1, bin_end + 1 + 1)
    idx_bin = np.broadcast_to(idx_bin, shape)
    return xr.DataArray(idx_bin, dims=dims)


def get_bright_band_mask(ds):
    """Retrieve bright band mask defined by binBBBottom and binBBTop.

    The bin is numerated from top to bottom.
    binBBTop has lower values than binBBBottom.
    binBBBottom and binBBTop are 0 when bright band limit is not detected !
    """
    # Retrieve required DataArrays
    da_bb_bottom = ds["binBBBottom"]
    da_bb_top = ds["binBBTop"]
    # Create 3D array with bin idex
    da_idx_bin = create_bin_idx_data_array(ds)
    # Identify bright band mask
    return np.logical_and(da_idx_bin >= da_bb_top, da_idx_bin <= da_bb_bottom)


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


def select_radar_frequency(xr_obj, radar_frequency):
    """Select data related to a specific radar frequency."""
    return xr_obj.sel({"radar_frequency": radar_frequency})


def select_spatial_3d_variables(ds, strict=False, squeeze=True):
    """Return xr.Dataset with only 3D spatial variables."""
    from gpm.checks import get_spatial_3d_variables

    variables = get_spatial_3d_variables(ds, strict=strict, squeeze=squeeze)
    return ds[variables]


def select_spatial_2d_variables(ds, strict=False, squeeze=True):
    """Return xr.Dataset with only 2D spatial variables."""
    from gpm.checks import get_spatial_2d_variables

    variables = get_spatial_2d_variables(ds, strict=strict, squeeze=squeeze)
    return ds[variables]


def select_transect_variables(ds, strict=False, squeeze=True):
    """Return xr.Dataset with only transect variables."""
    from gpm.checks import get_transect_variables

    variables = get_transect_variables(ds, strict=strict, squeeze=squeeze)
    return ds[variables]
