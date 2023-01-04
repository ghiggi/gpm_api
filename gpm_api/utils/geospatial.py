#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 09:30:29 2022

@author: ghiggi
"""
import numpy as np
import xarray as xr


def unwrap_longitude_degree(x, period=360):
    """Unwrap longitude array."""
    x = np.asarray(x)
    mod = period / 2
    return (x + mod) % (2 * mod) - mod


def crop_by_country(xr_obj, name):
    """
    Crop an xarray object based on the specified country name.

    Parameters
    ----------
    xr_obj : xr.DataArray or xr.Dataset
        xarray object.
    name : str
        Country name.

    Returns
    -------
    xr_obj : xr.DataArray or xr.Dataset
        Cropped xarray object.

    """

    from gpm_api.utils.countries import get_country_extent

    extent = get_country_extent(name)
    return crop(xr_obj=xr_obj, bbox=extent)


def crop(xr_obj, bbox):
    """
    Crop a xarray object based on the provided bounding box.

    Parameters
    ----------
    xr_obj : xr.DataArray or xr.Dataset
        xarray object.
    bbox : list or tuple
        The bounding box over which to crop the xarray object.
        `bbox` must follow the matplotlib and cartopy extent conventions:
        bbox = [x_min, x_max, y_min, y_max]

    Returns
    -------
    xr_obj : xr.DataArray or xr.Dataset
        Cropped xarray object.

    """
    # TODO: Check bbox
    lon = xr_obj["lon"].data
    lat = xr_obj["lat"].data
    # Crop orbit data
    # - Subset only along_track to allow concat on cross_track !
    if is_orbit(xr_obj):
        lon = xr_obj["lon"].data
        lat = xr_obj["lat"].data
        idx_row, idx_col = np.where(
            (lon >= bbox[0]) & (lon <= bbox[1]) & (lat >= bbox[2]) & (lat <= bbox[3])
        )
        if idx_row.size == 0:  # or idx_col.size == 0:
            raise ValueError("No data inside the provided bounding box.")
        # TODO: Check continuous ... otherwise warn and return a list of datasets
        xr_obj_subset = xr_obj.isel(
            along_track=slice((min(idx_row)), (max(idx_row) + 1))
        )
    elif is_grid(xr_obj):
        idx_row = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]
        idx_col = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]
        # If no data in the bounding box in current granule, return empty list
        if idx_row.size == 0 or idx_col.size == 0:
            raise ValueError("No data inside the provided bounding box.")
        else:
            xr_obj_subset = xr_obj.isel({"lon": idx_row, "lat": idx_col})
    else:
        orbit_dims = ("cross_track", "along_track")
        grid_dims = ("lon", "lat")
        raise ValueError(
            f"Dataset not recognized. Expecting dimensions {orbit_dims} or {grid_dims}."
        )

    return xr_obj_subset


#### TODO MOVE TO utils.checks !!!
def check_valid_geolocation(xr_obj, verbose=True):
    # TODO implement
    pass


def is_orbit(xr_obj):
    """Check whether the GPM xarray object is an orbit."""
    if "along_track" in list(xr_obj.dims):
        return True
    else:
        return False


def is_grid(xr_obj):
    """Check whether the GPM xarray object is a grid."""
    if "longitude" in list(xr_obj.dims) or "lon" in list(xr_obj.dims):
        return True
    else:
        return False


def is_spatial_2D_field(xr_obj):
    """Check whether the GPM xarray object is a 2D fields.

    It returns True if the object has only two spatial dimensions.
    The xarray object is squeezed before testing, so that (i.e. time)
    dimension of size 1 are "removed".
    """
    # Remove i.e. time/range dimension if len(1)
    xr_obj = xr_obj.squeeze()
    # Check if spatial 2D fields
    if set(xr_obj.dims) == set(("cross_track", "along_track")):
        return True
    elif set(xr_obj.dims) == set(("y", "x")):
        return True
    elif set(xr_obj.dims) == set(("latitude", "longitude")):
        return True
    elif set(xr_obj.dims) == set(
        ("lat", "lon")
    ):  # TOOD: Enforce latitude, longitude (i.e. with IMERG)
        return True
    else:
        return False


#### TODO MOVE TO pyresample accessor !!!


def get_pyresample_area(xr_obj):
    """It returns the corresponding pyresample area."""
    from pyresample import SwathDefinition, AreaDefinition

    # TODO: Implement as pyresample accessor
    # --> ds.pyresample.area

    # If Orbit Granule --> Swath Definition
    if is_orbit(xr_obj):
        # Define SwathDefinition with xr.DataArray lat/lons
        # - Otherwise fails https://github.com/pytroll/satpy/issues/1434
        lons = xr_obj["lon"].values
        lats = xr_obj["lat"].values

        # TODO: this might be needed
        # - otherwise ValueError 'ndarray is not C-contiguous' when resampling
        # lons = np.ascontiguousarray(lons)
        # lats = np.ascontiguousarray(lats)

        lons = xr.DataArray(lons, dims=["y", "x"])
        lats = xr.DataArray(lats, dims=["y", "x"])
        swath_def = SwathDefinition(lons, lats)
        return swath_def
    # If Grid Granule --> AreaDefinition
    elif is_grid(xr_obj):
        # Define AreaDefinition
        # TODO: derive area_extent, projection, ...
        raise NotImplementedError()
    # Unknown
    else:
        raise NotImplementedError()
