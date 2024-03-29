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
"""This module contains functions for geospatial processing."""
import difflib
import os
import warnings
from collections import namedtuple
from typing import Union

import numpy as np
import xarray as xr

from gpm import _root_path
from gpm.checks import is_grid, is_orbit
from gpm.utils.slices import get_list_slices_from_indices
from gpm.utils.yaml import read_yaml

# Shapely bounds: (xmin, ymin, xmax, ymax)
# Matlotlib extent: (xmin, xmax, ymin, ymax)
# Cartopy extent: (xmin, xmax, ymin, ymax)
# GPM-API extent: (xmin, xmax, ymin, ymax)

#### TODO:
# - croup_around(point, distance)
# - get_extent_around(point, distance)

# Define the namedtuple
Extent = namedtuple("Extent", "xmin xmax ymin ymax")


def _check_padding(padding: Union[int, float, tuple, list] = 0):
    """
    Check and normalize the padding input.

    This function accepts padding defined as an integer, float, tuple, or list.
    It normalizes the input into a tuple of four elements, each representing the
    number of degrees to extend the extent in each direction (left, right, top, bottom).

    Parameters
    ----------
    padding : int, float, tuple, list
        The padding value(s) provided. The function interprets the input as follows:
        - int or float: The same padding is applied to all four sides.
        - tuple or list:
            - If two values are provided (x, y), they are interpreted as horizontal
              and vertical padding, respectively, applied symmetrically (left=x, right=x, top=y, bottom=y).
            - If four values are provided, they directly correspond to padding for
              each side (left, right, top, bottom).
          The function will raise an error if a tuple or list does not contain 2 or 4 elements.

    Returns
    -------
    tuple
        A tuple of four elements (left, right, top, bottom), each representing the padding
        for that side.

    Raises
    ------
    ValueError
        If a tuple or list is provided with a length other than 2 or 4.
    TypeError
        If the input is not an int, float, tuple, or list.
    """
    if isinstance(padding, (int, float, np.floating, np.integer)):
        padding = tuple([padding] * 4)
    elif isinstance(padding, (tuple, list)):
        if len(padding) not in [2, 4]:
            raise ValueError("Expecting a padding (x, x, y, y) or (x, y) tuple.")
        if len(padding) == 2:
            padding = (padding[0], padding[0], padding[1], padding[1])
    else:
        raise TypeError("Accepted padding type are int, float, list or tuple.")
    return padding


def extend_geographic_extent(extent, padding: Union[int, float, tuple, list] = 0):
    """
    Extend the lat/lon extent by x degrees in every direction.

    Parameters
    ----------
    extent : (tuple)
        A tuple of four values representing the lat/lon extent.
        The extent format must be [xmin, xmax, ymin, ymax]
    padding : int, float, tuple, list
        The number of degrees to extend the extent in each direction.
        If padding is a single number, the same padding is applied in all directions.
        If padding is a tuple or list, it must contain 2 or 4 elements.
        If two values are provided (x, y), they are interpreted as longitude and latitude padding, respectively.
        If four values are provided, they directly correspond to padding for each side (left, right, top, bottom).

    Returns
    -------
    new_extent, tuple
        The extended extent.
    """
    padding = _check_padding(padding)
    xmin, xmax, ymin, ymax = extent
    xmin = max(xmin - padding[0], -180)
    xmax = min(xmax + padding[1], 180)
    ymin = max(ymin - padding[2], -90)
    ymax = min(ymax + padding[3], 90)
    new_extent = Extent(xmin, xmax, ymin, ymax)
    return new_extent


def read_countries_extent_dictionary():
    """
    Reads a YAML file containing countries extent information and returns it as a dictionary.

    Returns:
        dict: A dictionary containing countries extent information.
    """
    countries_extent_filepath = os.path.join(
        _root_path, "gpm", "etc", "geospatial", "country_extent.yaml"
    )
    countries_extent_dict = read_yaml(countries_extent_filepath)
    return countries_extent_dict


def get_country_extent(name, padding=0.2):
    """
    Retrieves the extent of a country.

    Parameters
    ----------
    name : str
        The name of the country.
    padding : int, float, tuple, list
        The number of degrees to extend the extent in each direction.
        If padding is a single number, the same padding is applied in all directions.
        If padding is a tuple or list, it must contain 2 or 4 elements.
        If two values are provided (x, y), they are interpreted as longitude and latitude padding, respectively.
        If four values are provided, they directly correspond to padding for each side (left, right, top, bottom).
        Default is 0.2.

    Returns
    -------
    extent : tuple
        A tuple containing the longitude and latitude extent of the country.

    Raises
    ------
    TypeError
        If the country name is not provided as a string.
    ValueError
        If the country name is not valid or if there is no matching country.

    Notes
    -----
    This function retrieves the extent of a country from a dictionary of country extents.
    The country extent is defined as the longitude and latitude range that encompasses the country's borders.
    The extent is returned as a tuple of four values: (xmin, xmax, ymin, ymax).
    The extent can be optionally padded by specifying the padding parameter.

    """
    # Check country format
    if not isinstance(name, str):
        raise TypeError("Please provide the country name as a string.")
    # Get country extent dictionary
    countries_extent_dict = read_countries_extent_dictionary()
    # Create same dictionary with lower case keys
    countries_lower_extent_dict = {s.lower(): v for s, v in countries_extent_dict.items()}
    # Get list of valid countries
    valid_countries = list(countries_extent_dict.keys())
    valid_countries_lower = list(countries_lower_extent_dict)
    if name.lower() in valid_countries_lower:
        extent = countries_lower_extent_dict[name.lower()]
        extent = extend_geographic_extent(extent, padding=padding)
        return extent
    else:
        possible_match = difflib.get_close_matches(name, valid_countries, n=1, cutoff=0.6)
        if len(possible_match) == 0:
            raise ValueError("Provide a valid country name.")
        else:
            possible_match = possible_match[0]
            raise ValueError(f"No matching country. Maybe are you looking for '{possible_match}'?")


def read_continents_extent_dictionary():
    """
    Read and return a dictionary containing the extents of continents.

    Returns:
        dict: A dictionary containing the extents of continents.
    """
    continents_extent_filepath = os.path.join(
        _root_path, "gpm", "etc", "geospatial", "continent_extent.yaml"
    )
    continents_extent_dict = read_yaml(continents_extent_filepath)
    return continents_extent_dict


def get_continent_extent(name: str, padding: Union[int, float, tuple, list] = 0):
    """
    Retrieves the extent of a continent.

    Parameters:
    -----------
    name : str
        The name of the continent.
    padding : int, float, tuple, list
        The number of degrees to extend the extent in each direction.
        If padding is a single number, the same padding is applied in all directions.
        If padding is a tuple or list, it must contain 2 or 4 elements.
        If two values are provided (x, y), they are interpreted as longitude and latitude padding, respectively.
        If four values are provided, they directly correspond to padding for each side (left, right, top, bottom).
        Default is 0.

    Returns:
    --------
    extent : tuple
        A tuple containing the longitude and latitude extent of the continent.

    Raises:
    -------
    TypeError:
        If the continent name is not provided as a string.
    ValueError:
        If the provided continent name is not valid or does not match any continent.
        If a similar continent name is found and suggested as a possible match.
    """
    # Check country format
    if not isinstance(name, str):
        raise TypeError("Please provide the continent name as a string.")

    # Create same dictionary with lower case keys
    continent_extent_dict = read_continents_extent_dictionary()
    continent_lower_extent_dict = {s.lower(): v for s, v in continent_extent_dict.items()}
    # Get list of valid continents
    valid_continent = list(continent_extent_dict.keys())
    valid_continent_lower = list(continent_lower_extent_dict)
    if name.lower() in valid_continent_lower:
        extent = continent_lower_extent_dict[name.lower()]
        extent = extend_geographic_extent(extent, padding=padding)
        return extent
    else:
        possible_match = difflib.get_close_matches(name, valid_continent, n=1, cutoff=0.6)
        if len(possible_match) == 0:
            raise ValueError(f"Provide a valid continent name from {valid_continent}.")
        else:
            possible_match = possible_match[0]
            raise ValueError(
                f"No matching continent. Maybe are you looking for '{possible_match}'?"
            )


def unwrap_longitude_degree(x, period=360):
    """Unwrap longitude array."""
    x = np.asarray(x)
    mod = period / 2
    return (x + mod) % (2 * mod) - mod


def _is_crossing_dateline(lon: Union[list, np.ndarray]):
    """Check if the longitude array is crossing the dateline."""
    lon = np.asarray(lon)
    diff = np.diff(lon)
    return np.any(np.abs(diff) > 180)


def get_extent(xr_obj, padding: Union[int, float, tuple, list] = 0):
    """Get the geographic extent from an xarray object.

    Parameters
    ----------
    xr_obj : xr.DataArray or xr.Dataset
        xarray object.
    padding : int, float, tuple, list
        The number of degrees to extend the extent in each direction.
        If padding is a single number, the same padding is applied in all directions.
        If padding is a tuple or list, it must contain 2 or 4 elements.
        If two values are provided (x, y), they are interpreted as longitude and latitude padding, respectively.
        If four values are provided, they directly correspond to padding for each side (left, right, top, bottom).
        Default is 0.

    Returns
    -------
    extent : tuple
        A tuple containing the longitude and latitude extent of the xarray object.
        The extent follows the matplotlib/cartopy format (xmin, xmax, ymin, ymax)

    """
    padding = _check_padding(padding=padding)

    lon = xr_obj["lon"].values
    lat = xr_obj["lat"].values

    if _is_crossing_dateline(lon):
        raise NotImplementedError(
            "The object cross the dateline. The extent can't be currently defined."
        )
    extent = Extent(np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat))
    extent = extend_geographic_extent(extent, padding=padding)
    return extent


def crop_by_country(xr_obj, name: str):
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
    extent = get_country_extent(name)
    return crop(xr_obj=xr_obj, extent=extent)


def crop_by_continent(xr_obj, name: str):
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
        lon = xr_obj["lon"].values
        lat = xr_obj["lat"].values
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
        lon = xr_obj["lon"].values
        lat = xr_obj["lat"].values
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
    src_area = src_ds.gpm.pyresample_area
    dst_area = dst_ds.gpm.pyresample_area

    # Rename dimensions to x, y for pyresample compatibility
    if src_ds.gpm.is_orbit:
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
    if dst_ds.gpm.is_orbit:
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
                from gpm.dataset.crs import get_pyresample_swath

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
