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
"""This module contains pyresample utility functions."""

import warnings

import numpy as np
import xarray as xr

from gpm.utils.decorators import check_software_availability


@check_software_availability(software="pyresample", conda_package="pyresample")
def remap(src_ds, dst_ds, radius_of_influence=20000, fill_value=np.nan):
    """Remap dataset to another one using nearest-neighbour.

    The spatial non-dimensional coordinates of the source dataset are not remapped. !
    The output dataset has the spatial coordinates of the destination dataset !
    """
    from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler

    from gpm.checks import get_spatial_dimensions
    from gpm.dataset.crs import _get_crs_coordinates, _get_proj_dim_coords, _get_swath_dim_coords, set_dataset_crs

    # TODO: segmentation fault occurs if input dataset to remap is a numpy array !

    # Get x and y dimensions
    x_dim, y_dim = get_spatial_dimensions(src_ds)

    # Reorder dimensions to be y, x, ...
    src_ds = src_ds.transpose(y_dim, x_dim, ...)
    dst_ds = dst_ds.transpose(y_dim, x_dim, ...)

    # Rename dimensions to x, y for pyresample compatibility
    src_ds = src_ds.swap_dims({y_dim: "y", x_dim: "x"})

    # Retrieve source and destination area
    src_area = src_ds.gpm.pyresample_area
    dst_area = dst_ds.gpm.pyresample_area

    # Retrieve source and destination crs coordinate
    src_crs_coords = _get_crs_coordinates(src_ds)[0]
    dst_crs_coords = _get_crs_coordinates(dst_ds)[0]

    # Define spatial coordinates of new object
    if dst_ds.gpm.is_orbit:  # SwathDefinition
        x_coord, y_coord = _get_swath_dim_coords(dst_ds)  # TODO: dst_ds.gpm.x, # dst_ds.gpm.y
        dst_spatial_coords = {
            x_coord: xr.DataArray(dst_ds[x_coord].data, dims=list(dst_ds[x_coord].dims), attrs=dst_ds[x_coord].attrs),
            y_coord: xr.DataArray(dst_ds[y_coord].data, dims=list(dst_ds[y_coord].dims), attrs=dst_ds[y_coord].attrs),
        }
    else:  # AreaDefinition
        x_arr, y_arr = dst_area.get_proj_coords()
        x_coord, y_coord = _get_proj_dim_coords(dst_ds)  # TODO: dst_ds.gpm.x, # dst_ds.gpm.y
        dst_spatial_coords = {
            x_coord: xr.DataArray(x_arr, dims=list(dst_ds[x_coord].dims), attrs=dst_ds[x_coord].attrs),
            y_coord: xr.DataArray(y_arr, dims=list(dst_ds[y_coord].dims), attrs=dst_ds[y_coord].attrs),
        }
        # Update units attribute if was rad or radians for geostationary data !
        if dst_spatial_coords[x_coord].attrs.get("units", "") in ["rad", "radians"]:
            dst_spatial_coords[x_coord].attrs["units"] = "deg"
            dst_spatial_coords[y_coord].attrs["units"] = "deg"

    # Define resampler
    resampler = KDTreeNearestXarrayResampler(src_area, dst_area)

    # Precompute resampler
    # - stuffs are recomputed if radius_of_influence and other args are not equally specified in .resample()
    # resampler.precompute(radius_of_influence=radius_of_influence)

    # Retrieve valid variables
    # - Variables with at least the (x,y) dimension
    variables = [var for var in src_ds.data_vars if set(src_ds[var].dims).issuperset({"x", "y"})]

    # Remap DataArrays
    with warnings.catch_warnings(record=True):
        da_dict = {
            var: resampler.resample(src_ds[var], radius_of_influence=radius_of_influence, fill_value=fill_value)
            for var in variables
        }

    # Create Dataset
    ds = xr.Dataset(da_dict)

    # Drop source crs coordinate
    ds = ds.drop_vars(src_crs_coords)

    # Drop crs added by pyresample
    if "crs" in ds:
        ds = ds.drop_vars("crs")

    # Revert to original spatial dimensions (of destination dataset)
    x_dim, y_dim = get_spatial_dimensions(dst_ds)
    ds = ds.swap_dims({"y": y_dim, "x": x_dim})

    # Add spatial coordinates
    ds = ds.assign_coords(dst_spatial_coords)

    # Add destination crs
    ds = set_dataset_crs(
        ds,
        crs=dst_area.crs,
        grid_mapping_name=dst_crs_coords,
    )

    # Coordinates specifics to gpm-api
    gpm_api_coords = ["gpm_id", "gpm_time", "gpm_granule_id", "gpm_along_track_id", "gpm_cross_track_id"]
    gpm_api_coords_dict = {c: dst_ds.reset_coords()[c] for c in gpm_api_coords if c in dst_ds.coords}
    ds = ds.assign_coords(gpm_api_coords_dict)

    # Drop pyresample area attribute
    for var in ds.data_vars:
        ds[var].attrs.pop("area", None)

    # Transpose variable back to the expected dimension
    # TODO: ds = ds.transpose(y_dim, x_dim, ...)

    # # Add relevant coordinates of dst_ds
    # dst_available_coords = list(dst_ds.coords)
    # useful_coords = [coord for coord in dst_available_coords if np.all(np.isin(dst_ds[coord].dims, ds.dims))]
    # dict_coords = {coord: dst_ds[coord] for coord in useful_coords}
    # ds = ds.assign_coords(dict_coords)
    # ds = ds.drop(src_crs_coords)
    return ds


@check_software_availability(software="pyresample", conda_package="pyresample")
def get_pyresample_area(xr_obj):
    """It returns the corresponding pyresample area."""
    import pyresample  # noqa
    from gpm.dataset.crs import get_pyresample_area as _get_pyresample_area

    # Ensure correct dimension order for Swath
    if "cross_track" in xr_obj.dims:
        xr_obj = xr_obj.transpose("cross_track", "along_track", ...)
    # Return pyresample area
    return _get_pyresample_area(xr_obj)
