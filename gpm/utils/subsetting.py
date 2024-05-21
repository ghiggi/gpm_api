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
"""This module contains functions for subsetting and aligning GPM ORBIT Datasets."""
import numpy as np
from xarray.core.utils import either_dict_or_kwargs


def is_1d_non_dimensional_coord(xr_obj, coord):
    """Checks if a coordinate is a 1d, non-dimensional coordinate."""
    if coord not in xr_obj.coords:
        return False
    if xr_obj[coord].ndim != 1:
        return False
    if xr_obj[coord].dims[0] == coord:
        return False
    return True


def is_dimensional_coord(xr_obj, coord):
    """Checks if a coordinate is a dimensional coordinate."""
    if coord not in xr_obj.coords:
        return False
    if coord not in xr_obj.dims:
        return False
    return True


def _get_isel_indices_from_sel_indices(xr_obj, coord, sel_indices):
    """Get isel_indices corresponding to sel_indices."""
    da_coord = xr_obj[coord]
    dim = da_coord.dims[0]
    da_coord = da_coord.assign_coords({"isel_indices": (dim, np.arange(0, da_coord.size))})
    da_subset = da_coord.swap_dims({dim: coord}).sel({coord: sel_indices})
    isel_indices = da_subset["isel_indices"].data
    # isel_indices = xr_obj[coord].isin(sel_indices) # this was not keep the ordering of sel_indices !
    # isel_indices = np.where(isel_indices.data)[0]
    # if len(isel_indices) == 0: # return empty instead
    #     raise ValueError(f"The coordinate {coord} does not have matching values.")
    return isel_indices


def _get_dim_of_1d_non_dimensional_coord(xr_obj, coord):
    """Get the dimension of a 1D non-dimension coordinate."""
    if not is_1d_non_dimensional_coord(xr_obj, coord):
        raise ValueError(f"{coord} is not a 1D non-dimensional coordinate.")
    dim = xr_obj[coord].dims[0]
    return dim


def _get_dim_isel_on_non_dim_coord_from_isel(xr_obj, coord, isel_indices):
    """Get dimension and isel_indices related to a 1D non-dimension coordinate.

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        A xarray object.
    coord : str
        Name of the coordinate wishing to subset with .sel
    isel_indices : (str, int, float, list, np.array)
        Coordinate indices wishing to be selected.

    Returns
    -------
    dim : str
        Dimension related to the 1D non-dimension coordinate.
    isel_indices : (int, list, slice)
        Indices for index-based selection.
    """
    dim = _get_dim_of_1d_non_dimensional_coord(xr_obj, coord)
    return dim, isel_indices


def _get_dim_isel_on_non_dim_coord_from_sel(xr_obj, coord, sel_indices):
    """
    Return the dimension and isel_indices related to a 1D non-dimension coordinate.

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        A xarray object.
    coord : str
        Name of the coordinate wishing to subset with .sel
    sel_indices : (str, int, float, list, np.array)
        Coordinate values wishing to be selected.

    Returns
    -------
    dim : str
        Dimension related to the 1D non-dimension coordinate.
    isel_indices : np.ndarray
        Indices for index-based selection.
    """
    dim = _get_dim_of_1d_non_dimensional_coord(xr_obj, coord)
    isel_indices = _get_isel_indices_from_sel_indices(xr_obj, coord=coord, sel_indices=sel_indices)
    return dim, isel_indices


def get_dim_isel_indices_from_isel_indices(xr_obj, key, isel_indices):
    """Return the dimension and isel_indices related to isel_indices of a coordinate."""
    # Non-dimensional coordinate case
    if key not in xr_obj.dims:
        key, isel_indices = _get_dim_isel_on_non_dim_coord_from_isel(xr_obj, coord=key, isel_indices=isel_indices)
    return key, isel_indices


def get_dim_isel_indices_from_sel_indices(xr_obj, key, sel_indices):
    """Return the dimension and isel_indices related to sel_indices of a coordinate."""
    # Dimension case
    if key in xr_obj.dims:
        if key not in xr_obj.coords:
            raise ValueError("Can not subset with sel a dimension if it is not also a coordinate.")
        isel_indices = _get_isel_indices_from_sel_indices(xr_obj, coord=key, sel_indices=sel_indices)

    # Non-dimensional coordinate case
    else:
        key, isel_indices = _get_dim_isel_on_non_dim_coord_from_sel(xr_obj, coord=key, sel_indices=sel_indices)
    return key, isel_indices


def get_dim_isel_indices(xr_obj, key, indices, func="isel"):
    if func == "sel":
        key, isel_indices = get_dim_isel_indices_from_sel_indices(xr_obj, key=key, sel_indices=indices)
    elif func == "isel":
        key, isel_indices = get_dim_isel_indices_from_isel_indices(xr_obj, key=key, isel_indices=indices)
    return key, isel_indices


def _subset(xr_obj, indexers=None, func="isel", drop=False, **indexers_kwargs):
    """Perform selection with isel or isel."""
    # Check valid func
    valid_func = ["isel", "sel"]
    if func not in valid_func:
        raise ValueError(f"Valid 'func' values are {valid_func}")

    # Retrieve indexers
    indexers = either_dict_or_kwargs(indexers, indexers_kwargs, func)

    # Define isel_dict
    isel_dict = {}
    for key, indices in indexers.items():
        key, isel_indices = get_dim_isel_indices(xr_obj, key=key, indices=indices, func=func)
        if key in isel_dict:
            raise ValueError(f"Multiple indexers point to {key} dimension.")
        isel_dict[key] = isel_indices

    # Subset and update area
    xr_obj = xr_obj.isel(isel_dict, drop=drop)
    return xr_obj


def isel(xr_obj, indexers=None, drop=False, **indexers_kwargs):
    """Perform index-based selection."""
    return _subset(xr_obj, indexers=indexers, func="isel", drop=drop, **indexers_kwargs)


def sel(xr_obj, indexers=None, drop=False, **indexers_kwargs):
    """Perform value-based selection."""
    return _subset(xr_obj, indexers=indexers, func="sel", drop=drop, **indexers_kwargs)


####------------------------------------------------------------------------------------------------------------------.
#### Alignment


def check_gpm_id(xr_obj):
    if "gpm_id" not in xr_obj.coords:
        msg = "The dataset does not have the 'gpm_id' coordinate. Impossible to align."
        raise ValueError(msg)


def check_gpm_cross_track_id(xr_obj):
    if "gpm_cross_track_id" not in xr_obj.coords:
        msg = "The dataset does not have the 'gpm_cross_track_id' coordinate. Impossible to align."
        raise ValueError(msg)


def _check_coord_exist(xr_obj, coord):
    if coord not in xr_obj.coords:
        msg = f"The dataset does not have the '{coord}' coordinate. Impossible to align."
        raise ValueError(msg)


def _split_gpm_id_key(s):
    prefix, suffix = s.split("-")
    return (prefix, int(suffix))


def _align_spatial_coord(coord, *args):
    """
    Align GPM / GPM-GEO xarray objects along a coordinate.

    Parameters
    ----------
    coords: str
        Coordinate name
    args : list
        A list of GPM / GPM-GEO xr.Dataset or xr.DataArray.

    Returns
    -------
    list_aligned : list
        A list of aligned GPM / GPM-GEO xr.Dataset or xr.DataArray.

    """
    from functools import reduce

    list_xr_obj = args
    # Check the coordinate is always available
    _ = [_check_coord_exist(xr_obj, coord) for xr_obj in list_xr_obj]
    # Retrieve list of coordinate values
    list_id = [xr_obj[coord].data.astype(str) for xr_obj in list_xr_obj]
    # Retrieve intersection of coordinates values
    # - np.atleast_1d ensure that the dimension is not dropped if only 1 value
    # - np.intersect1d returns the sorted array of common unique elements
    # - PROBLEM: for 'gpm_id': '36083-999' comes after '36083-4483'. So ad-hoc reorder
    sel_indices = np.atleast_1d(reduce(np.intersect1d, list_id))
    if len(sel_indices) == 0:
        raise ValueError(f"No common {coord}.")
    if coord == "gpm_id":
        sel_indices = np.array(sorted(sel_indices, key=_split_gpm_id_key))
    # Subset datasets
    list_aligned = []
    for xr_obj in list_xr_obj:
        xr_obj = sel(xr_obj, {coord: sel_indices})
        list_aligned.append(xr_obj)
    return list_aligned


def align_along_track(*args):
    """
    Align GPM / GPM-GEO xarray objects in the along-track direction.

    Parameters
    ----------
    args : list
        A list of GPM / GPM-GEO xr.Dataset or xr.DataArray.

    Returns
    -------
    list_aligned : list
        A list of aligned GPM / GPM-GEO xr.Dataset or xr.DataArray.

    """
    return _align_spatial_coord("gpm_id", *args)


def align_cross_track(*args):
    """
    Align GPM / GPM-GEO xarray objects in the cross-track direction.

    Parameters
    ----------
    args : list
        A list of GPM / GPM-GEO xr.Dataset or xr.DataArray.

    Returns
    -------
    list_aligned : list
        A list of aligned GPM / GPM-GEO xr.Dataset or xr.DataArray.

    """
    return _align_spatial_coord("gpm_cross_track_id", *args)
