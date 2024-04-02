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

# ----------------------------------------------------------------------------.


def check_is_xarray(x):
    if not isinstance(x, (xr.DataArray, xr.Dataset)):
        raise TypeError("Expecting a xr.Dataset or xr.DataArray.")


def check_is_xarray_dataarray(x):
    if not isinstance(x, xr.DataArray):
        raise TypeError("Expecting a xr.DataArray.")


def check_is_xarray_dataset(x):
    if not isinstance(x, xr.Dataset):
        raise TypeError("Expecting a xr.Dataset.")


def get_dataset_variables(ds, sort=False):
    """Get list of xr.Dataset variables."""
    variables = list(ds.data_vars)
    if sort:
        variables = sorted(variables)
    return variables


def _get_available_spatial_dims(xr_obj):
    """Get xarray object available spatial dimensions."""
    dims = list(xr_obj.dims)
    flattened_spatial_dims = list(chain.from_iterable(SPATIAL_DIMS))
    return tuple(np.array(flattened_spatial_dims)[np.isin(flattened_spatial_dims, dims)].tolist())


def _get_available_vertical_dims(xr_obj):
    """Get xarray object available vertical dimensions."""
    dims = list(xr_obj.dims)
    return tuple(np.array(VERTICAL_DIMS)[np.isin(VERTICAL_DIMS, dims)].tolist())


def _get_available_frequency_dims(xr_obj):
    """Get xarray object available frequency dimensions."""
    dims = list(xr_obj.dims)
    return tuple(np.array(FREQUENCY_DIMS)[np.isin(FREQUENCY_DIMS, dims)].tolist())


def _is_grid_expected_spatial_dims(spatial_dims):
    """Check if the GRID spatial dimensions have the expected names."""
    # TODO: refactor ! GRID_SPATIAL_DIMS
    is_grid = set(spatial_dims) == set(GRID_SPATIAL_DIMS)
    is_lonlat = set(spatial_dims) == {"latitude", "longitude"}
    is_xy = set(spatial_dims) == {"y", "x"}
    if is_grid or is_lonlat or is_xy:
        return True
    return False


def _is_swath_expected_spatial_dims(spatial_dims):
    """Check if the ORBIT spatial dimensions have the expected names."""
    # TODO: refactor ! ORBIT_SPATIAL_DIMS
    is_orbit = set(spatial_dims) == set(ORBIT_SPATIAL_DIMS)
    is_xy = set(spatial_dims) == {"y", "x"}
    if is_orbit or is_xy:
        return True
    return False


def _is_expected_spatial_dims(spatial_dims):
    """Check that the spatial_dims are the expected two."""
    is_orbit = _is_swath_expected_spatial_dims(spatial_dims)
    is_grid = _is_grid_expected_spatial_dims(spatial_dims)
    if is_orbit or is_grid:
        return True
    return False


def is_orbit(xr_obj):
    """Check whether the GPM xarray object is an orbit."""
    from gpm.dataset.crs import _get_proj_dim_coords

    # Check dimension names
    spatial_dims = _get_available_spatial_dims(xr_obj)
    if not _is_swath_expected_spatial_dims(spatial_dims):
        return False

    # Check that no 1D coords exists
    # - Swath objects are determined by 2D coordinates only
    x_coord, y_coord = _get_proj_dim_coords(xr_obj)
    if x_coord is None and y_coord is None:
        return True
    return False


def is_grid(xr_obj):
    """Check whether the GPM xarray object is a grid."""
    from gpm.dataset.crs import _get_proj_dim_coords

    # Check dimension names
    spatial_dims = _get_available_spatial_dims(xr_obj)
    if not _is_grid_expected_spatial_dims(spatial_dims):
        return False

    # Check that 1D coords exists
    # - Area objects can be determined by 1D and 2D coordinates
    # - 1D coordinates: projection coordinates
    # - 2D coordinates: lon/lat coordinates of each pixel^
    x_coord, y_coord = _get_proj_dim_coords(xr_obj)
    if x_coord is not None and y_coord is not None:
        return True
    return False


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


def check_has_cross_track_dimension(xr_obj):
    if "cross_track" not in xr_obj.dims:
        raise ValueError("The 'cross-track' dimension is not available.")


def check_has_along_track_dimension(xr_obj):
    if "along_track" not in xr_obj.dims:
        raise ValueError("The 'along_track' dimension is not available.")


def _is_spatial_2d_datarray(da, strict):
    """Check if a DataArray is a spatial 2D array."""
    spatial_dims = _get_available_spatial_dims(da)
    if not _is_expected_spatial_dims(spatial_dims):
        return False
    vertical_dims = _get_available_vertical_dims(da)

    if vertical_dims:
        return False

    if strict and len(da.dims) != 2:
        return False

    return True


def _is_spatial_3d_datarray(da, strict):
    """Check if a DataArray is a spatial 3D array."""
    spatial_dims = _get_available_spatial_dims(da)
    if not _is_expected_spatial_dims(spatial_dims):
        return False
    vertical_dims = _get_available_vertical_dims(da)

    if not vertical_dims:
        return False

    if strict and len(da.dims) != 3:
        return False

    return True


def _is_transect_datarray(da, strict):
    """Check if a DataArray is a spatial 3D array."""
    spatial_dims = list(_get_available_spatial_dims(da))
    if len(spatial_dims) != 1:
        return False
    vertical_dims = list(_get_available_vertical_dims(da))

    if not vertical_dims:
        return False

    if strict and len(da.dims) != 2:
        return False

    return True


def _is_spatial_2d_dataset(ds, strict):
    """Check if all DataArrays of a xr.Dataset are spatial 2D array."""
    all_2d_spatial = np.all(
        [_is_spatial_2d_datarray(ds[var], strict=strict) for var in get_dataset_variables(ds)],
    ).item()
    if all_2d_spatial:
        return True
    return False


def _is_spatial_3d_dataset(ds, strict):
    """Check if all DataArrays of a xr.Dataset are spatial 3D array."""
    all_3d_spatial = np.all(
        [_is_spatial_3d_datarray(ds[var], strict=strict) for var in get_dataset_variables(ds)],
    ).item()
    if all_3d_spatial:
        return True
    return False


def _is_transect_dataset(ds, strict):
    """Check if all DataArrays of a xr.Dataset are spatial profile array."""
    all_profile_spatial = np.all(
        [_is_transect_datarray(ds[var], strict=strict) for var in get_dataset_variables(ds)],
    ).item()
    if all_profile_spatial:
        return True
    return False


def is_spatial_2d(xr_obj, strict=True, squeeze=True):
    """Check if is spatial 2d xarray object.

    If squeeze=True (default), dimensions of size=1 are removed prior testing.
    If strict=True (default), the DataArray must have just the 2D spatial dimensions.
    If strict=False, the DataArray can have additional dimensions (except vertical).
    """
    check_is_xarray(xr_obj)
    if squeeze:
        xr_obj = xr_obj.squeeze()  # remove dimensions of size 1
    if isinstance(xr_obj, xr.Dataset):
        return _is_spatial_2d_dataset(xr_obj, strict=strict)
    return _is_spatial_2d_datarray(xr_obj, strict=strict)


def is_spatial_3d(xr_obj, strict=True, squeeze=True):
    """Check if is spatial 3d xarray object."""
    check_is_xarray(xr_obj)
    if squeeze:
        xr_obj = xr_obj.squeeze()  # remove dimensions of size 1
    if isinstance(xr_obj, xr.Dataset):
        return _is_spatial_3d_dataset(xr_obj, strict=strict)
    return _is_spatial_3d_datarray(xr_obj, strict=strict)


def is_transect(xr_obj, strict=True, squeeze=True):
    """Check if is spatial profile xarray object."""
    check_is_xarray(xr_obj)
    if squeeze:
        xr_obj = xr_obj.squeeze()  # remove dimensions of size 1
    if isinstance(xr_obj, xr.Dataset):
        return _is_transect_dataset(xr_obj, strict=strict)
    return _is_transect_datarray(xr_obj, strict=strict)


def check_is_spatial_2d(da, strict=True, squeeze=True):
    if not is_spatial_2d(da, strict=strict, squeeze=squeeze):
        raise ValueError("Expecting a 2D GPM field.")


def check_is_spatial_3d(da, strict=True, squeeze=True):
    if not is_spatial_3d(da, strict=strict, squeeze=squeeze):
        raise ValueError("Expecting a 3D GPM field.")


def check_is_transect(da, strict=True, squeeze=True):
    if not is_transect(da, strict=strict, squeeze=squeeze):
        raise ValueError("Expecting a transect of a 3D GPM field.")


def get_spatial_2d_variables(ds, strict=False, squeeze=True):
    """Get list of xr.Dataset 2D spatial variables."""
    variables = [var for var in get_dataset_variables(ds) if is_spatial_2d(ds[var], strict=strict, squeeze=squeeze)]
    return sorted(variables)


def get_spatial_3d_variables(ds, strict=False, squeeze=True):
    """Get list of xr.Dataset 3D spatial variables."""
    variables = [var for var in get_dataset_variables(ds) if is_spatial_3d(ds[var], strict=strict, squeeze=squeeze)]
    return sorted(variables)


def get_transect_variables(ds, strict=False, squeeze=True):
    """Get list of xr.Dataset trasect variables."""
    variables = [var for var in get_dataset_variables(ds) if is_transect(ds[var], strict=strict, squeeze=squeeze)]
    return sorted(variables)


def get_frequency_variables(ds):
    """Get list of xr.Dataset variables with frequency-related dimension."""
    variables = [var for var in get_dataset_variables(ds) if _get_available_frequency_dims(ds[var])]
    return sorted(variables)


def get_vertical_dimension(xr_obj):
    """Return the name of the vertical dimension."""
    return list(_get_available_vertical_dims(xr_obj))


def get_spatial_dimensions(xr_obj):
    """Return the name of the spatial dimensions."""
    return list(_get_available_spatial_dims(xr_obj))
