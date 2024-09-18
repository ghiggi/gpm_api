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


def remap(src_ds, dst_ds, radius_of_influence=20000, fill_value=np.nan):
    """Remap dataset to another one using nearest-neighbour."""
    from gpm.checks import get_spatial_dimensions
    from gpm.dataset.crs import (
        _get_crs_coordinates, 
        _get_swath_dim_coords,
        _get_proj_dim_coords,
        set_dataset_crs
    )
    try:
        from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler
    except ImportError:
        raise ImportError(
            "The 'pyresample' package is required but not found. "
            "Please install it using the following command: "
            "conda install -c conda-forge pyresample",
        )

    # Retrieve source and destination area
    src_area = src_ds.gpm.pyresample_area
    dst_area = dst_ds.gpm.pyresample_area
    
    # Retrieve source and destination crs coordinate
    src_crs_coords = _get_crs_coordinates(src_ds)[0]
    dst_crs_coords = _get_crs_coordinates(dst_ds)[0]
        
    # Rename dimensions to x, y for pyresample compatibility
<<<<<<< HEAD
    x_dim, y_dim = get_spatial_dimensions(src_ds)
    src_ds = src_ds.swap_dims({y_dim: "y", x_dim: "x"})  
             
    # Define spatial coordinates of new object
    if dst_ds.gpm.is_orbit: # SwathDefinition
        x_coord, y_coord = _get_swath_dim_coords(dst_ds) # dst_ds.gpm.x, # dst_ds.gpm.y
        dst_spatial_coords = {
            x_coord: xr.DataArray(dst_ds[x_coord].data, 
                                 dims=list(dst_ds[x_coord].dims),
                                 attrs=dst_ds[x_coord].attrs),
            y_coord: xr.DataArray(dst_ds[y_coord].data, 
                                 dims=list(dst_ds[y_coord].dims),
                                 attrs=dst_ds[y_coord].attrs),
        }
    else: # AreaDefinition
        x_arr, y_arr = dst_area.get_proj_coords()
        x_coord, y_coord = _get_proj_dim_coords(dst_ds) # dst_ds.gpm.x, # dst_ds.gpm.y
        dst_spatial_coords = {
            x_coord: xr.DataArray(x_arr, 
                                  dims=list(dst_ds[x_coord].dims),
                                  attrs=dst_ds[x_coord].attrs),
            y_coord: xr.DataArray(y_arr,  
                                  dims=list(dst_ds[y_coord].dims),
                                  attrs=dst_ds[y_coord].attrs),
        }
        # Update units attribute if was rad or radians for geostationary data !
        if dst_spatial_coords[x_coord].attrs.get("units", "") in ["rad", "radians"]:
            dst_spatial_coords[x_coord].attrs["units"] = "deg"
            dst_spatial_coords[y_coord].attrs["units"] = "deg"
            
=======
    if src_ds.gpm.is_orbit:
        src_ds = src_ds.swap_dims({"cross_track": "y", "along_track": "x"})
    elif not np.all(np.isin(["x", "y"], list(src_ds.dims))):
        # TODO: GENERALIZE to allow also latitude, longitude !
        src_ds = src_ds.swap_dims({"lat": "y", "lon": "x"})

>>>>>>> e740a7b5f4b1f3d796f1207a41d1706653876c2a
    # Define resampler
    resampler = KDTreeNearestXarrayResampler(src_area, dst_area)
    resampler.precompute(radius_of_influence=radius_of_influence)

    # Retrieve valid variables
    variables = [var for var in src_ds.data_vars if set(src_ds[var].dims).issuperset({"x", "y"})]

    # Remap DataArrays
    with warnings.catch_warnings(record=True):
        da_dict = {var: resampler.resample(src_ds[var], fill_value=fill_value) for var in variables}

    # Create Dataset
    ds = xr.Dataset(da_dict)
    
    # Drop source crs coordinate
    ds = ds.drop(src_crs_coords)
    
    # Drop crs added by pyresample 
    if "crs" in ds:
        ds = ds.drop("crs")
        
    # Revert to original spatial dimensions (of destination dataset)
    x_dim, y_dim = get_spatial_dimensions(dst_ds)
    ds = ds.swap_dims({"y": y_dim, "x": x_dim })

    # Add spatial coordinates
    ds = ds.assign_coords(dst_spatial_coords)        
    
    # Add destination crs 
    ds = set_dataset_crs(ds, 
                         crs=dst_area.crs, 
                         grid_mapping_name=dst_crs_coords,
                         )  
    # Coordinates specifics to gpm-api 
    gpm_api_coords = ["gpm_id", "gpm_time", "gpm_granule_id", "gpm_along_track_id", "gpm_cross_track_id"]
    gpm_api_coords_dict = {c: dst_ds.reset_coords()[c] for c in gpm_api_coords if c in dst_ds.coords}
    ds = ds.assign_coords(gpm_api_coords_dict)   
    
    # # Add relevant coordinates of dst_ds
    # dst_available_coords = list(dst_ds.coords)
    # useful_coords = [coord for coord in dst_available_coords if np.all(np.isin(dst_ds[coord].dims, ds.dims))]
    # dict_coords = {coord: dst_ds[coord] for coord in useful_coords}
    # ds = ds.assign_coords(dict_coords)
    # ds = ds.drop(src_crs_coords)
    return ds


def get_pyresample_area(xr_obj):
    """It returns the corresponding pyresample area."""
    try:
        import pyresample  # noqa
        from gpm.dataset.crs import get_pyresample_area as _get_pyresample_area
    except ImportError:
        raise ImportError(
            "The 'pyresample' package is required but not found. "
            "Please install it using the following command: "
            "conda install -c conda-forge pyresample",
        )

    # Ensure correct dimension order for Swath
    if "cross_track" in xr_obj.dims:
        xr_obj = xr_obj.transpose("cross_track", "along_track", ...)
    # Return pyresample area
    return _get_pyresample_area(xr_obj)
