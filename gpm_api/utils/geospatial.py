#!/usr/bin/env python3
"""
Created on Wed Aug 17 09:30:29 2022

@author: ghiggi
"""
import numpy as np
import xarray as xr

from gpm_api.utils.slices import get_list_slices_from_indices

#### TODO:
# - croup_around(point, distance)
# - get_extent_around(point, distance)


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
        # Retrieve list of along_track slices
        list_slices = get_list_slices_from_indices(idx_col)
        return list_slices
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
        slices_dict = {"lon": lon_slices, "lat": lat_slices}
        return slices_dict
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
        list_slices = get_crop_slices_by_extent(xr_obj, extent)
        if len(list_slices) > 1:
            raise ValueError(
                "The orbit is crossing the extent multiple times. Use get_crop_slices_by_extent !."
            )
        xr_obj_subset = xr_obj.isel(along_track=list_slices[0])

    elif is_grid(xr_obj):
        slice_dict = get_crop_slices_by_extent(xr_obj, extent)
        xr_obj_subset = xr_obj.isel(slice_dict)
    else:
        orbit_dims = ("cross_track", "along_track")
        grid_dims = ("lon", "lat")
        raise ValueError(
            f"Dataset not recognized. Expecting dimensions {orbit_dims} or {grid_dims}."
        )

    return xr_obj_subset


####---------------------------------------------------------------------------.
#### TODO MOVE TO utils.checks !!!


def is_orbit(xr_obj):
    # TODO: --> MOVED TO gpm_api.dataset
    """Check whether the GPM xarray object is an orbit."""
    return "along_track" in list(xr_obj.dims)


def is_grid(xr_obj):
    # TODO: --> MOVED TO gpm_api.dataset
    """Check whether the GPM xarray object is a grid."""
    return bool("longitude" in list(xr_obj.dims) or "lon" in list(xr_obj.dims))


def is_spatial_2d(xr_obj):
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
        else:
            raise ValueError("Can not derive SwathDefinition area without cross-track dimension.")

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
