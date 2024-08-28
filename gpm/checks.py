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
"""This module defines functions providing GPM-API Dataset information."""
from itertools import chain

import numpy as np
import xarray as xr

from gpm.dataset.dimensions import (
    FREQUENCY_DIMS,
    GRID_SPATIAL_DIMS,
    ORBIT_SPATIAL_DIMS,
    SPATIAL_DIMS,
    VERTICAL_DIMS,
)
from gpm.utils.xarray import (
    check_is_xarray,
    get_dataset_variables,
)

# Refactor Notes
# - GRID_SPATIAL_DIMS, ORBIT_SPATIAL_DIMS to be refactored
# - is_grid, is_orbit currently also depends on gpm.dataset.crs._get_proj_dim_coords

# - Code could be generalized to work with any satellite data format ???
# - GPM ORBIT = pyresample SwathDefinition
# - GPM GRID = pyresample AreaDefinition
# - GPM ORBIT dimensions: (cross-track, along-track)
# - GPM GRID dimensions: (lon, lat)
# - satpy dimensions (y, x) (for both ORBIT and GRID)
# --> Accept both (lat, lon), (latitude, longitude), (y,x), (...) coordinates
# --> Adapt plotting, crop utility to deal with different coordinate names
# --> Then this functions can be used with whatever satellite products

####-----------------------------------------------------------------------------------------------------------------.
####################
#### Dimensions ####
####################


def get_frequency_dimension(xr_obj):
    """Return the name of the available frequency dimension."""
    return np.array(FREQUENCY_DIMS)[np.isin(FREQUENCY_DIMS, list(xr_obj.dims))].tolist()


def get_vertical_dimension(xr_obj):
    """Return the name of the available vertical dimension."""
    vertical_dim = np.array(VERTICAL_DIMS)[np.isin(VERTICAL_DIMS, list(xr_obj.dims))].tolist()
    if len(vertical_dim) > 1:
        raise ValueError(f"Only one vertical dimension is allowed. Got {vertical_dim}.")
    return vertical_dim


def get_spatial_dimensions(xr_obj):
    """Return the name of the available spatial dimensions."""
    dims = list(xr_obj.dims)
    flattened_spatial_dims = list(chain.from_iterable(SPATIAL_DIMS))
    spatial_dimensions = np.array(flattened_spatial_dims)[np.isin(flattened_spatial_dims, dims)].tolist()
    if len(spatial_dimensions) > 2:
        raise ValueError(f"Only two horizontal spatial dimensions are allowed. Got {spatial_dimensions}.")
    return spatial_dimensions


def _has_spatial_dim_dataarray(da, strict):
    """Check if the xarray.DataArray has spatial horizontal dimensions."""
    spatial_dims = get_spatial_dimensions(da)
    if not spatial_dims:
        return False
    if strict:  # only spatial dimensions
        return bool(np.all(np.isin(da.dims, spatial_dims)))
    return True


def _has_vertical_dim_dataarray(da, strict):
    """Check if the xarray.DataArray has a vertical dimension."""
    vertical_dims = list(get_vertical_dimension(da))
    if not vertical_dims:
        return False
    only_vertical_dim = len(da.dims) == 1
    if strict and not only_vertical_dim:  # noqa
        return False
    return True


def _has_frequency_dim_dataarray(da, strict):
    """Check if the xarray.DataArray has a frequency dimension."""
    frequency_dims = list(get_frequency_dimension(da))
    if not frequency_dims:
        return False
    only_frequency_dim = len(da.dims) == 1
    if strict and not only_frequency_dim:  # noqa
        return False
    return True


def _has_vertical_dim_dataset(ds, strict):
    """Check if at least one xarray.DataArrays of a xarray.Dataset have a vertical dimension."""
    has_vertical = np.any(
        [_has_vertical_dim_dataarray(ds[var], strict=strict) for var in get_dataset_variables(ds)],
    ).item()
    return bool(has_vertical)


def _has_spatial_dim_dataset(ds, strict):
    """Check if at least one xarray.DataArrays of a xarray.Dataset have at least one spatial dimension."""
    has_spatial = np.any(
        [_has_spatial_dim_dataarray(ds[var], strict=strict) for var in get_dataset_variables(ds)],
    ).item()
    return bool(has_spatial)


def _has_frequency_dim_dataset(ds, strict):
    """Check if at least one xarray.DataArrays of a xarray.Dataset have a frequency dimension."""
    has_spatial = np.any(
        [_has_frequency_dim_dataarray(ds[var], strict=strict) for var in get_dataset_variables(ds)],
    ).item()
    return bool(has_spatial)


def _check_xarray_conditions(da_condition, ds_condition, xr_obj, strict, squeeze):
    check_is_xarray(xr_obj)
    if squeeze:
        xr_obj = xr_obj.squeeze()  # remove dimensions of size 1
    if isinstance(xr_obj, xr.Dataset):
        return ds_condition(xr_obj, strict=strict)
    return da_condition(xr_obj, strict=strict)


def has_spatial_dim(xr_obj, strict=False, squeeze=True):
    """Check if the xarray.DataArray or xarray.Dataset have a spatial dimension.

    If ``squeeze=True`` (default), dimensions of size=1 are removed prior testing.
    If ``strict=True`` , the xarray.DataArray can have only spatial dimensions.
    If ``strict=False`` (default), the xarray.DataArray can also have other dimensions.
    """
    return _check_xarray_conditions(
        _has_spatial_dim_dataarray,
        _has_spatial_dim_dataset,
        xr_obj=xr_obj,
        strict=strict,
        squeeze=squeeze,
    )


def has_vertical_dim(xr_obj, strict=False, squeeze=True):
    """Check if the xarray.DataArray or xarray.Dataset have a vertical dimension.

    If ``squeeze=True`` (default), dimensions of size=1 are removed prior testing.
    If ``strict=True``  , the xarray.DataArray must have just the vertical dimension.
    If ``strict=False`` (default), the xarray.DataArray can also have additional dimensions.
    """
    return _check_xarray_conditions(
        _has_vertical_dim_dataarray,
        _has_vertical_dim_dataset,
        xr_obj=xr_obj,
        strict=strict,
        squeeze=squeeze,
    )


def has_frequency_dim(xr_obj, strict=False, squeeze=True):
    """Check if the xarray.DataArray or xarray.Dataset have a frequency dimension.

    If ``squeeze=True`` (default), dimensions of size=1 are removed prior testing.
    If ``strict=True`` , the xarray.DataArray must have just the frequency dimension.
    If ``strict=False`` (default), the xarray.DataArray can also have additional dimensions.
    """
    return _check_xarray_conditions(
        _has_frequency_dim_dataarray,
        _has_frequency_dim_dataset,
        xr_obj=xr_obj,
        strict=strict,
        squeeze=squeeze,
    )


####-------------------------------------------------------------------------------------
#######################
#### GRID vs ORBIT ####
#######################


def _is_grid_expected_spatial_dims(spatial_dims):
    """Check if the GRID spatial dimensions have the expected names."""
    is_grid = set(spatial_dims) == set(GRID_SPATIAL_DIMS)
    is_lonlat = set(spatial_dims) == {"latitude", "longitude"}
    is_xy = set(spatial_dims) == {"y", "x"}
    return bool(is_grid or is_lonlat or is_xy)


def _is_orbit_expected_spatial_dims(spatial_dims):
    """Check if the ORBIT spatial dimensions have the expected names.

    Allow to have only one dimension: cross_track or along_track.
    """
    # is_orbit = set(spatial_dims) == set(ORBIT_SPATIAL_DIMS)
    # is_xy = set(spatial_dims) == {"y", "x"}

    # Check if spatial_dims is a non-empty subset of ORBIT_SPATIAL_DIMS
    is_orbit = set(spatial_dims).issubset(ORBIT_SPATIAL_DIMS) and bool(spatial_dims)
    is_xy = set(spatial_dims).issubset({"y", "x"}) and bool(spatial_dims)
    return bool(is_orbit or is_xy)


def _is_expected_spatial_dims(spatial_dims):
    """Check that the spatial_dims are the expected two."""
    is_orbit = _is_orbit_expected_spatial_dims(spatial_dims)
    is_grid = _is_grid_expected_spatial_dims(spatial_dims)
    return bool(is_orbit or is_grid)


def is_orbit(xr_obj):
    """Check whether the xarray object is a GPM ORBIT.

    An ORBIT cross-section (nadir view) or transect is considered ORBIT.
    An ORBIT object must have the coordinates available.
    """
    from gpm.dataset.crs import _get_swath_dim_coords

    # Check dimension names
    spatial_dims = get_spatial_dimensions(xr_obj)
    if not _is_orbit_expected_spatial_dims(spatial_dims):
        return False

    # Check that swath coords exists
    # - Swath objects are determined by 1D (nadir looking) and 2D coordinates
    x_coord, y_coord = _get_swath_dim_coords(xr_obj)
    return bool(x_coord is not None and y_coord is not None)


def is_grid(xr_obj):
    """Check whether the xarray object is a GPM GRID.

    A GRID slice is not considered a GRID object !
    An GRID object must have the coordinates available !
    """
    from gpm.dataset.crs import _get_proj_dim_coords

    # Check dimension names
    spatial_dims = get_spatial_dimensions(xr_obj)
    if not _is_grid_expected_spatial_dims(spatial_dims):
        return False

    # Check that 1D coords exists
    # - Area objects can be determined by 1D and 2D coordinates
    # - 1D coordinates: projection coordinates
    # - 2D coordinates: lon/lat coordinates of each pixel
    x_coord, y_coord = _get_proj_dim_coords(xr_obj)
    return bool(x_coord is not None and y_coord is not None)


####-------------------------------------------------------------------------------------
#######################
#### ORBIT TYPES   ####
#######################


def _is_spatial_2d_dataarray(da, strict):
    """Check if the xarray.DataArray is a spatial 2D array."""
    spatial_dims = get_spatial_dimensions(da)
    if not _is_expected_spatial_dims(spatial_dims) or len(spatial_dims) != 2:
        return False

    vertical_dims = get_vertical_dimension(da)
    if vertical_dims:
        return False
    if strict and len(da.dims) != 2:  # noqa
        return False

    return True


def _is_spatial_3d_dataarray(da, strict):
    """Check if the xarray.DataArray is a spatial 3D array."""
    spatial_dims = get_spatial_dimensions(da)
    if not _is_expected_spatial_dims(spatial_dims) or len(spatial_dims) != 2:
        return False

    vertical_dims = get_vertical_dimension(da)
    if not vertical_dims:
        return False
    if strict and len(da.dims) != 3:  # noqa
        return False

    return True


def _is_cross_section_dataarray(da, strict):
    """Check if the xarray.DataArray is a cross-section array."""
    spatial_dims = list(get_spatial_dimensions(da))
    if len(spatial_dims) != 1:
        return False
    vertical_dims = list(get_vertical_dimension(da))

    if not vertical_dims:
        return False

    if strict and len(da.dims) != 2:  # noqa
        return False

    return True


def _is_transect_dataarray(da, strict):
    """Check if the xarray.DataArray is a transect array."""
    spatial_dims = list(get_spatial_dimensions(da))
    if len(spatial_dims) != 1:
        return False
    if strict and len(da.dims) != 1:  # noqa
        return False
    return True


def _check_dataarrays_condition(condition, ds, strict):
    if not ds:  # Empty dataset (no variables)
        return False
    all_valid = np.all(
        [condition(ds[var], strict=strict) for var in get_dataset_variables(ds)],
    )
    return bool(all_valid)


def _is_spatial_2d_dataset(ds, strict):
    """Check if all xarray.DataArrays of a xarray.Dataset are spatial 2D objects."""
    return _check_dataarrays_condition(_is_spatial_2d_dataarray, ds=ds, strict=strict)


def _is_spatial_3d_dataset(ds, strict):
    """Check if all xarray.DataArrays of a xarray.Dataset are spatial 3D objects."""
    return _check_dataarrays_condition(_is_spatial_3d_dataarray, ds=ds, strict=strict)


def _is_cross_section_dataset(ds, strict):
    """Check if all xarray.DataArrays of a xarray.Dataset are cross-section objects."""
    return _check_dataarrays_condition(_is_cross_section_dataarray, ds=ds, strict=strict)


def _is_transect_dataset(ds, strict):
    """Check if all xarray.DataArrays of a xarray.Dataset are transects objects."""
    return _check_dataarrays_condition(_is_transect_dataarray, ds=ds, strict=strict)


def is_spatial_2d(xr_obj, strict=True, squeeze=True):
    """Check if the xarray.DataArray or xarray.Dataset is a spatial 2D object.

    If ``squeeze=True`` (default), dimensions of size=1 are removed prior testing.
    If ``strict=True``  (default), the xarray.DataArray must have just the 2D spatial dimensions.
    If ``strict=False`` , the xarray.DataArray can have additional dimensions (except vertical).
    """
    return _check_xarray_conditions(
        _is_spatial_2d_dataarray,
        _is_spatial_2d_dataset,
        xr_obj=xr_obj,
        strict=strict,
        squeeze=squeeze,
    )


def is_spatial_3d(xr_obj, strict=True, squeeze=True):
    """Check if the xarray.DataArray or xarray.Dataset i as spatial 3d object.

    If ``squeeze=True`` (default), dimensions of size=1 are removed prior testing.
    If ``strict=True``  (default), the xarray.DataArray must have just the 3D spatial dimensions.
    If ``strict=False`` , the xarray.DataArray can also have additional dimensions.
    """
    return _check_xarray_conditions(
        _is_spatial_3d_dataarray,
        _is_spatial_3d_dataset,
        xr_obj=xr_obj,
        strict=strict,
        squeeze=squeeze,
    )


def is_cross_section(xr_obj, strict=True, squeeze=True):
    """Check if the xarray.DataArray or xarray.Dataset is a cross-section object.

    If ``squeeze=True`` (default), dimensions of size=1 are removed prior testing.
    If ``strict=True``  (default), the xarray.DataArray must have just the
    vertical dimension and a horizontal dimension.
    If ``strict=False`` , the xarray.DataArray can have additional dimensions but only
    a single horizontal and vertical dimension.
    """
    return _check_xarray_conditions(
        _is_cross_section_dataarray,
        _is_cross_section_dataset,
        xr_obj=xr_obj,
        strict=strict,
        squeeze=squeeze,
    )


def is_transect(xr_obj, strict=True, squeeze=True):
    """Check if the xarray.DataArray or xarray.Dataset is a transect object.

    If ``squeeze=True`` (default), dimensions of size=1 are removed prior testing.
    If ``strict=True``  (default), the xarray.DataArray must have just an horizontal dimension.
    If ``strict=False`` , the xarray.DataArray can have additional dimensions but only a single
    horizontal dimension.
    """
    return _check_xarray_conditions(
        _is_transect_dataarray,
        _is_transect_dataset,
        xr_obj=xr_obj,
        strict=strict,
        squeeze=squeeze,
    )


####-------------------------------------------------------------------------------------------------------------.
#################
#### Checks  ####
#################


def check_is_orbit(xr_obj):
    """Check is a GPM ORBIT object."""
    if not is_orbit(xr_obj):
        raise ValueError("Expecting a GPM ORBIT object.")


def check_is_grid(xr_obj):
    """Check is a GPM GRID object."""
    if not is_grid(xr_obj):
        raise ValueError("Expecting a GPM GRID object.")


def check_is_gpm_object(xr_obj):
    """Check is a GPM object (GRID or ORBIT)."""
    if not is_orbit(xr_obj) and not is_grid(xr_obj):
        raise ValueError("Unrecognized GPM xarray object.")


def check_has_cross_track_dim(xr_obj, dim="cross_track"):
    if dim not in xr_obj.dims:
        raise ValueError(f"The 'cross-track' dimension {dim} is not available.")


def check_has_along_track_dim(xr_obj, dim="along_track"):
    if dim not in xr_obj.dims:
        raise ValueError(f"The 'along_track' dimension {dim} is not available.")


def check_is_spatial_2d(xr_obj, strict=True, squeeze=True):
    """Check if the xarray.DataArray or xarray.Dataset is a spatial 2D field.

    If ``squeeze=True`` (default), dimensions of size=1 are removed prior testing.
    If ``strict=True``  (default), the xarray.DataArray must have just the 2D spatial dimensions.
    If ``strict=False`` , the xarray.DataArray can also have additional dimensions (except vertical).
    """
    if not is_spatial_2d(xr_obj, strict=strict, squeeze=squeeze):
        raise ValueError("Expecting a 2D GPM field.")


def check_is_spatial_3d(xr_obj, strict=True, squeeze=True):
    """Check if the xarray.DataArray or xarray.Dataset is a spatial 3D field.

    If ``squeeze=True`` (default), dimensions of size=1 are removed prior testing.
    If ``strict=True``  (default), the xarray.DataArray must have just the 3D spatial dimensions.
    If ``strict=False`` , the xarray.DataArray can also have additional dimensions.
    """
    if not is_spatial_3d(xr_obj, strict=strict, squeeze=squeeze):
        raise ValueError("Expecting a 3D GPM field.")


def check_is_cross_section(xr_obj, strict=True, squeeze=True):
    """Check if the xarray.DataArray or xarray.Dataset is a cross-section.

    If ``squeeze=True`` (default), dimensions of size=1 are removed prior testing.
    If ``strict=True``  (default), the xarray.DataArray must have just the
    vertical dimension and a horizontal dimension.
    If ``strict=False`` , the xarray.DataArray can also have additional dimensions,
    but only a single vertical and horizontal dimension.
    """
    if not is_cross_section(xr_obj, strict=strict, squeeze=squeeze):
        raise ValueError("Expecting a cross-section extracted from a 3D GPM field.")


def check_is_transect(xr_obj, strict=True, squeeze=True):
    """Check if the xarray.DataArray or xarray.Dataset is a transect.

    If ``squeeze=True`` (default), dimensions of size=1 are removed prior testing.
    If ``strict=True``  (default), the xarray.DataArray must have just an horizontal dimension.
    If ``strict=False`` , the xarray.DataArray can also have additional dimensions,
    but only an horizontal dimension.
    """
    if not is_transect(xr_obj, strict=strict, squeeze=squeeze):
        raise ValueError("Expecting a transect object.")


def check_has_vertical_dim(xr_obj, strict=False, squeeze=True):
    """Check if the xarray.DataArray or xarray.Dataset have a vertical dimension.

    If ``squeeze=True`` (default), dimensions of size=1 are removed prior testing.
    If ``strict=False`` (default), the xarray.DataArray can also have additional dimensions.
    If ``strict=True`` , the xarray.DataArray must have just the vertical dimension.
    """
    if not has_vertical_dim(xr_obj, strict=strict, squeeze=squeeze):
        only = "only " if strict else ""
        raise ValueError(f"Expecting an xarray object with {only}a vertical dimension.")


def check_has_spatial_dim(xr_obj, strict=False, squeeze=True):
    """Check if the xarray.DataArray or xarray.Dataset has at least one spatial horizontal dimension.

    If ``squeeze=True`` (default), dimensions of size=1 are removed prior testing.
    If ``strict=False`` (default), the xarray.DataArray can also have additional dimensions.
    If ``strict=True`` , the xarray.DataArray must have just the spatial dimensions.
    """
    if not has_spatial_dim(xr_obj, strict=strict, squeeze=squeeze):
        only = "only " if strict else ""
        raise ValueError(f"Expecting an xarray object with {only}spatial dimensions.")


def check_has_frequency_dim(xr_obj, strict=False, squeeze=True):
    """Check if the xarray.DataArray or xarray.Dataset has a frequency dimension.

    If ``squeeze=True`` (default), dimensions of size=1 are removed prior testing.
    If ``strict=False`` (default), the xarray.DataArray can also have additional dimensions.
    If ``strict=True`` , the xarray.DataArray must have just the spatial dimensions.
    """
    if not has_frequency_dim(xr_obj, strict=strict, squeeze=squeeze):
        only = "only " if strict else ""
        raise ValueError(f"Expecting an xarray object with {only}a frequency dimension.")


####-----------------------------------------------------------------------------------------------------------------.
###############################
#### Variables information ####
###############################


def get_spatial_2d_variables(ds, strict=False, squeeze=True):
    """Get list of xarray.Dataset 2D spatial variables.

    If ``strict=False`` (default), the potential variables for which a 2D spatial field can be derived.
    If ``strict=True``, the variables that are already a 2D spatial field.
    """
    variables = [var for var in get_dataset_variables(ds) if is_spatial_2d(ds[var], strict=strict, squeeze=squeeze)]
    return sorted(variables)


def get_spatial_3d_variables(ds, strict=False, squeeze=True):
    """Get list of xarray.Dataset 3D spatial variables.

    If ``strict=False`` (default), the potential variables for which a 3D spatial field can be derived.
    If ``strict=True``, the variables that are already a 3D spatial field.
    """
    variables = [var for var in get_dataset_variables(ds) if is_spatial_3d(ds[var], strict=strict, squeeze=squeeze)]
    return sorted(variables)


def get_cross_section_variables(ds, strict=False, squeeze=True):
    """Get list of xarray.Dataset cross-section variables.

    If ``strict=False`` (default), the potential variables for which a strict cross-section can be derived.
    If ``strict=True``, the variables that are already a cross-section.
    """
    variables = [var for var in get_dataset_variables(ds) if is_cross_section(ds[var], strict=strict, squeeze=squeeze)]
    return sorted(variables)


# def get_transect_variables(ds, strict=False, squeeze=True):
#     """Get list of xarray.Dataset transect variables.

#     If ``strict=False`` (default), the potential variables for which a strict transect can be derived.
#     If ``strict=True``, the variables that are already a transect.
#     """
#     variables = [var for var in get_dataset_variables(ds) if is_transect(ds[var], strict=strict, squeeze=squeeze)]
#     return sorted(variables)


def get_vertical_variables(ds):
    """Get list of xarray.Dataset variables with vertical dimension."""
    variables = [var for var in get_dataset_variables(ds) if has_vertical_dim(ds[var], strict=False, squeeze=True)]
    return sorted(variables)


def get_frequency_variables(ds):
    """Get list of xarray.Dataset variables with frequency-related dimension."""
    variables = [var for var in get_dataset_variables(ds) if has_frequency_dim(ds[var], strict=False, squeeze=True)]
    return sorted(variables)


def get_bin_variables(ds):
    """Get list of xarray.Dataset radar product variables with name starting with `bin` or ending with `Bin`.

    In CMB products, bin variables end with the `Bin`  suffix.
    In L1 and L2 RADAR products, bin variables starts with the `bin`  prefix.
    """
    variables = [var for var in get_dataset_variables(ds) if var.startswith("bin") or var.endswith("Bin")]
    return sorted(variables)
