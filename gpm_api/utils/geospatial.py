#!/usr/bin/env python3
"""
Created on Wed Aug 17 09:30:29 2022

@author: ghiggi
"""
import warnings

import numpy as np
import xarray as xr

from gpm_api.checks import is_grid, is_orbit
from gpm_api.utils.slices import get_list_slices_from_indices

# Shapely bounds: (xmin, ymin, xmax, ymax)
# Matlotlib extent: (xmin, xmax, ymin, ymax)
# Cartopy extent: (xmin, xmax, ymin, ymax)
# GPM-API extent: (xmin, xmax, ymin, ymax)

#### TODO:
# - croup_around(point, distance)
# - get_extent_around(point, distance)
# - rename file crop.py?


def unwrap_longitude_degree(x, period=360):
    """Unwrap longitude array."""
    x = np.asarray(x)
    mod = period / 2
    return (x + mod) % (2 * mod) - mod


def get_extent(xr_obj, padding=0):
    """Get geographic extent.

    The extent follows the matplotlib/cartopy format (xmin, xmax, ymin, ymax)
    The padding tuple is expected to follow the format (x, y)
    """
    if isinstance(padding, (int, float)):
        padding = (padding, padding)
    elif isinstance(padding, (tuple, list)):
        if len(padding) != 2:
            raise ValueError("Expecting a padding (x, y) tuple of length 2.")
    else:
        raise TypeError("Accepted padding type are int, float, list or tuple.")
    lon = xr_obj["lon"].data
    lat = xr_obj["lat"].data
    lon_min = max(-180, np.nanmin(lon).item() - padding[0])
    lon_max = min(180, np.nanmax(lon).item() + padding[0])
    lat_min = max(-90, np.nanmin(lat).item() - padding[1])
    lat_max = min(90, np.nanmax(lat).item() + padding[1])

    if lon_min > lon_max:
        raise NotImplementedError(
            "The object cross the dateline. The extent can't be currently be defined."
        )
    extent = tuple([lon_min, lon_max, lat_min, lat_max])
    return extent


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
    return crop(xr_obj=xr_obj, extent=extent)


def crop_by_continent(xr_obj, name):
    """
    Crop an xarray object based on the specified continent name.

    Parameters
    ----------
    xr_obj : xr.DataArray or xr.Dataset
        xarray object.
    name : str
        Continent name.

    Returns
    -------
    xr_obj : xr.DataArray or xr.Dataset
        Cropped xarray object.

    """

    from gpm_api.utils.continents import get_continent_extent

    extent = get_continent_extent(name)
    return crop(xr_obj=xr_obj, extent=extent)


def get_crop_slices_by_extent(xr_obj, extent):
    """Compute the xarray object slices which are within the specified extent.


    If the input is a GPM Orbit, it returns a list of along-track slices
    If the input is a GPM Grid, it returns a dictionary of the lon/lat slices.

    Parameters
    ----------
    xr_obj : xr.DataArray or xr.Dataset
        xarray object.
    extent : list or tuple
        The extent over which to crop the xarray object.
        `extent` must follow the matplotlib and cartopy conventions:
        extent = [x_min, x_max, y_min, y_max]
    """

    if is_orbit(xr_obj):
        xr_obj = xr_obj.transpose("cross_track", "along_track", ...)
        lon = xr_obj["lon"].data
        lat = xr_obj["lat"].data
        idx_row, idx_col = np.where(
            (lon >= extent[0]) & (lon <= extent[1]) & (lat >= extent[2]) & (lat <= extent[3])
        )
        if idx_col.size == 0:
            raise ValueError("No data inside the provided bounding box.")

        # Retrieve list of along_track slices isel_dict
        list_slices = get_list_slices_from_indices(idx_col)
        list_isel_dicts = [{"along_track": slc} for slc in list_slices]
        return list_isel_dicts

    elif is_grid(xr_obj):
        lon = xr_obj["lon"].data
        lat = xr_obj["lat"].data
        idx_col = np.where((lon >= extent[0]) & (lon <= extent[1]))[0]
        idx_row = np.where((lat >= extent[2]) & (lat <= extent[3]))[0]
        # If no data in the bounding box in current granule, return empty list
        if idx_row.size == 0 or idx_col.size == 0:
            raise ValueError("No data inside the provided bounding box.")
        lat_slices = get_list_slices_from_indices(idx_row)[0]
        lon_slices = get_list_slices_from_indices(idx_col)[0]
        isel_dict = {"lon": lon_slices, "lat": lat_slices}
        return isel_dict
    else:
        raise NotImplementedError("")


def get_crop_slices_by_continent(xr_obj, name):
    """Compute the xarray object slices which are within the specified continent.

    If the input is a GPM Orbit, it returns a list of along-track slices
    If the input is a GPM Grid, it returns a dictionary of the lon/lat slices.

    Parameters
    ----------
    xr_obj : xr.DataArray or xr.Dataset
        xarray object.
    name : str
        Continent name.
    """
    from gpm_api.utils.continents import get_continent_extent

    extent = get_continent_extent(name)
    return get_crop_slices_by_extent(xr_obj=xr_obj, extent=extent)


def get_crop_slices_by_country(xr_obj, name):
    """Compute the xarray object slices which are within the specified country.

    If the input is a GPM Orbit, it returns a list of along-track slices
    If the input is a GPM Grid, it returns a dictionary of the lon/lat slices.

    Parameters
    ----------
    xr_obj : xr.DataArray or xr.Dataset
        xarray object.
    name : str
        Country name.
    """
    from gpm_api.utils.countries import get_country_extent

    extent = get_country_extent(name)
    return get_crop_slices_by_extent(xr_obj=xr_obj, extent=extent)


def crop(xr_obj, extent):
    """
    Crop a xarray object based on the provided bounding box.

    Parameters
    ----------
    xr_obj : xr.DataArray or xr.Dataset
        xarray object.
    extent : list or tuple
        The bounding box over which to crop the xarray object.
        `extent` must follow the matplotlib and cartopy extent conventions:
        extent = [x_min, x_max, y_min, y_max]

    Returns
    -------
    xr_obj : xr.DataArray or xr.Dataset
        Cropped xarray object.

    """
    # TODO: Check extent
    if is_orbit(xr_obj):
        # - Subset only along_track
        list_isel_dicts = get_crop_slices_by_extent(xr_obj, extent)
        if len(list_isel_dicts) > 1:
            raise ValueError(
                "The orbit is crossing the extent multiple times. Use get_crop_slices_by_extent !."
            )
        xr_obj_subset = xr_obj.isel(list_isel_dicts[0])

    elif is_grid(xr_obj):
        isel_dict = get_crop_slices_by_extent(xr_obj, extent)
        xr_obj_subset = xr_obj.isel(isel_dict)
    else:
        orbit_dims = ("cross_track", "along_track")
        grid_dims = ("lon", "lat")
        raise ValueError(
            f"Dataset not recognized. Expecting dimensions {orbit_dims} or {grid_dims}."
        )

    return xr_obj_subset


####---------------------------------------------------------------------------.
#### TODO MOVE TO pyresample accessor !!!


def remap(src_ds, dst_ds, radius_of_influence=20000, fill_value=np.nan):
    """Remap data from one dataset to another one."""
    from pyresample.future.resamplers.nearest import KDTreeNearestXarrayResampler

    # Retrieve source and destination area
    src_area = src_ds.gpm_api.pyresample_area
    dst_area = dst_ds.gpm_api.pyresample_area

    # Rename dimensions to x, y for pyresample compatibility
    if src_ds.gpm_api.is_orbit:
        src_ds = src_ds.swap_dims({"cross_track": "y", "along_track": "x"})
    else:
        src_ds = src_ds.swap_dims({"lat": "y", "lon": "x"})

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

    # Set correct dimensions
    if dst_ds.gpm_api.is_orbit:
        ds = ds.swap_dims({"y": "cross_track", "x": "along_track"})
    else:
        ds = ds.swap_dims({"y": "lat", "x": "lon"})

    # Add relevant coordinates of dst_ds
    dst_available_coords = list(dst_ds.coords)
    useful_coords = []
    for coord in dst_available_coords:
        if np.all(np.isin(dst_ds[coord].dims, ds.dims)):
            useful_coords.append(coord)
    dict_coords = {coord: dst_ds[coord] for coord in useful_coords}
    ds = ds.assign_coords(dict_coords)
    return ds


def get_pyresample_area(xr_obj):
    """It returns the corresponding pyresample area."""
    from pyresample import SwathDefinition

    # TODO: Implement as pyresample accessor
    # --> ds.pyresample.area
    # ds.crs.to_pyresample_area
    # ds.crs.to_pyresample_swath

    # If Orbit Granule --> Swath Definition
    if is_orbit(xr_obj):
        # Define SwathDefinition with xr.DataArray lat/lons
        # - Otherwise fails https://github.com/pytroll/satpy/issues/1434

        # Ensure correct dimension order
        if "cross_track" in xr_obj.dims:
            xr_obj = xr_obj.transpose("cross_track", "along_track", ...)
            lons = xr_obj["lon"].values
            lats = xr_obj["lat"].values
            # This has been fixed in pyresample very likely
            # --> otherwise ValueError 'ndarray is not C-contiguous' when resampling
            # lons = np.ascontiguousarray(lons)
            # lats = np.ascontiguousarray(lats)
            lons = xr.DataArray(lons, dims=["y", "x"])
            lats = xr.DataArray(lats, dims=["y", "x"])
            swath_def = SwathDefinition(lons, lats)
        else:
            try:
                from gpm_api.dataset.crs import get_pyresample_swath

                swath_def = get_pyresample_swath(xr_obj)
            except Exception:
                raise ValueError("Not a swath object.")
        return swath_def

    # If Grid Granule --> AreaDefinition
    elif is_grid(xr_obj):
        # Define AreaDefinition
        # TODO: derive area_extent, projection, ...
        raise NotImplementedError()
    # Unknown
    else:
        raise NotImplementedError()
