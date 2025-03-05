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
from collections import namedtuple

import numpy as np
import pyproj

from gpm import _root_path
from gpm.checks import is_grid, is_orbit
from gpm.utils.decorators import check_is_gpm_object
from gpm.utils.slices import get_list_slices_from_indices
from gpm.utils.yaml import read_yaml

# Shapely bounds: (xmin, ymin, xmax, ymax)
# Matlotlib extent: (xmin, xmax, ymin, ymax)
# Cartopy extent: (xmin, xmax, ymin, ymax)
# GPM-API extent: (xmin, xmax, ymin, ymax)


#### Extent (namedtuple)
Extent = namedtuple("Extent", "xmin xmax ymin ymax")


def _check_size(size: int | float | tuple | list = 0):
    """Check and normalize the size input.

    This function accepts size defined as an integer, float, tuple, or list.
    It normalizes the input into a tuple of two elements, each representing the
    desired size in degrees of the extent in the longitude and latitude direction.

    Parameters
    ----------
    size : int, float, tuple, list
        The size value(s) provided. The function interprets the input as follows:
        - int or float: The same size is enforced in both directions.
        - tuple or list: Check that only two values are provided.

    Returns
    -------
    tuple
        A tuple of two elements (x_size, y_size)

    Raises
    ------
    ValueError
        If a tuple or list is provided with a length other than 2.
    TypeError
        If the input is not an int, float, tuple, or list.

    """
    if isinstance(size, (int, float, np.floating, np.integer)):
        size = tuple([size] * 2)
    elif isinstance(size, (tuple, list)):
        if len(size) != 2:
            raise ValueError("Expecting a size (x, y) tuple.")
    else:
        raise TypeError("Accepted size type are int, float, list or tuple.")
    if np.any(np.array(size) < 0):
        raise ValueError("Expecting positive 'size' values.")
    return size


def _check_padding(padding: int | float | tuple | list = 0):
    """Check and normalize the padding input.

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
    if padding is None:
        padding = 0
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


def check_extent(extent):
    """
    Validates the extent to ensure it has the correct format and logical consistency.

    Note: this function does not check for the realism of extent values !

    Parameters
    ----------
    extent : list or tuple
        The extent specified as [xmin, xmax, ymin, ymax].

    Returns
    -------
    extent: tuple

    """
    if len(extent) != 4:
        raise ValueError("Extent must contain exactly four elements: [xmin, xmax, ymin, ymax].")
    for v in extent:
        if not isinstance(v, (int, float, np.floating, np.integer)):
            raise ValueError("The extent must be composed by numeric values.")
    if not (extent[0] <= extent[1]):
        raise ValueError("xmin must be less than xmax.")
    if not (extent[2] <= extent[3]):
        raise ValueError("ymin must be less than ymax.")
    return Extent(*extent)


####------------------------------------------------------------------------------------.
#### Planar Extent


def get_extent_around_point(x, y, distance=None, size=None):
    """
    Get the extent around a point.

    Either specify ``distance`` or the wished extent ``size`` (in the unit of the extent).

    Parameters
    ----------
    x : float
        X coordinate of the point.
    y : float
        Y coordinate of the point.
    distance: float
        Distance from the point in each direction.
    size : int, float, tuple, list
        The size of the extent in each direction.
        If ``size`` is a single number, the same size is ensured in all directions.
        If ``size`` is a tuple or list, it must of size 2  and specifying
        the desired size of the extent in the x direction
        and the y direction.

    Returns
    -------
    tuple
        The adjusted extent.

    """
    if distance is not None and size is not None:
        raise ValueError("Either provide the 'distance' or the 'size' of the extent.")
    if distance is None and size is None:
        raise ValueError("Please provide the 'distance' or the 'size' of the extent.")
    if size is not None:
        return adjust_extent(extent=[x, x, y, y], size=size)
    # Calculate new points in the four cardinal directions by the specified distance
    extent = [x - distance, x + distance, y - distance, y + distance]
    return extend_extent(extent, padding=0)


def adjust_extent(extent, size):
    """
    Adjust the extent to have the desired size.

    Parameters
    ----------
    extent : tuple
        A tuple of four values representing the extent.
        The extent format must be ``[xmin, xmax, ymin, ymax]``.
    size : int, float, tuple, list
        The size in degrees of the extent in each direction.
        If ``size`` is a single number, the same size is ensured in all directions.
        If ``size`` is a tuple or list, it must of size 2  and specifying
        the desired size of the extent in the x direction and the y direction.

    Returns
    -------
    tuple
        The adjusted extent.

    """
    # Retrieve desired size
    x_size, y_size = _check_size(size)

    # Retrieve current extent
    extent = Extent(*extent)

    # Center of the current extent
    x_center = (extent.xmax + extent.xmin) / 2
    y_center = (extent.ymax + extent.ymin) / 2

    # Define new min and max xgitudes and yitudes
    xmin = x_center - x_size / 2
    xmax = x_center + x_size / 2
    ymin = y_center - y_size / 2
    ymax = y_center + y_size / 2

    return Extent(xmin, xmax, ymin, ymax)


def merge_extents(list_extent):
    """Return the outer extent of a list of extents."""
    extents = np.vstack(list_extent)
    extent = [
        extents[:, 0].min().item(),
        extents[:, 1].max().item(),
        extents[:, 2].min().item(),
        extents[:, 3].max().item(),
    ]
    return extent


def extend_extent(extent, padding: int | float | tuple | list = 0):
    """Extend the extent by padding in every direction.

    Parameters
    ----------
    extent : tuple
        A tuple of four values representing the extent.
        The extent format must be ``[xmin, xmax, ymin, ymax]``.
    padding : int, float, tuple, list
        The number of degrees to extend the extent in each direction.
        If ``padding`` is a single number, the same padding is applied in all directions.
        If ``padding`` is a tuple or list, it must contain 2 or 4 elements.
        If two values are provided (x, y), they are interpreted as x and y padding, respectively.
        If four values are provided, they directly correspond to padding for each side ``(left, right, top, bottom)``.

    Returns
    -------
    tuple
        The extended extent.

    """
    padding = _check_padding(padding)
    xmin, xmax, ymin, ymax = extent
    xmin = xmin - padding[0]
    xmax = xmax + padding[1]
    ymin = ymin - padding[2]
    ymax = ymax + padding[3]
    return Extent(xmin, xmax, ymin, ymax)


####------------------------------------------------------------------------------------.
#### Geographic Extent


def extend_geographic_extent(extent, padding: int | float | tuple | list = 0):
    """Extend the lat/lon extent by x degrees in every direction.

    Parameters
    ----------
    extent : tuple
        A tuple of four values representing the lat/lon extent.
        The extent format must be ``[xmin, xmax, ymin, ymax]``.
    padding : int, float, tuple, list
        The number of degrees to extend the extent in each direction.
        If ``padding`` is a single number, the same padding is applied in all directions.
        If ``padding`` is a tuple or list, it must contain 2 or 4 elements.
        If two values are provided (x, y), they are interpreted as longitude and latitude padding, respectively.
        If four values are provided, they directly correspond to padding for each side ``(left, right, top, bottom)``.

    Returns
    -------
    tuple
        The extended extent.

    """
    extent = extend_extent(extent, padding)
    xmin = max(extent.xmin, -180)
    xmax = min(extent.xmax, 180)
    ymin = max(extent.ymin, -90)
    ymax = min(extent.ymax, 90)
    return Extent(xmin, xmax, ymin, ymax)


def adjust_geographic_extent(extent, size):
    """
    Adjust the extent to have the desired size.

    Parameters
    ----------
    extent : tuple
        A tuple of four values representing the lat/lon extent.
        The extent format must be ``[xmin, xmax, ymin, ymax]``.
    size : int, float, tuple, list
        The size in degrees of the extent in each direction.
        If ``size`` is a single number, the same size is ensured in all directions.
        If ``size`` is a tuple or list, it must of size 2  and specifying
        the desired size of the extent in the x direction (longitude)
        and the y direction (latitude).

    Returns
    -------
    tuple
        The adjusted extent.

    """
    new_lon_min, new_lon_max, new_lat_min, new_lat_max = adjust_extent(extent, size)
    # Ensure within [-180, 180] longitude extent and of desired size
    if new_lon_min < -180:
        new_lon_max = new_lon_max + (new_lon_min + 180)
        new_lon_min = -180
    if new_lon_max > 180:
        new_lon_min = new_lon_min - (new_lon_max - 180)
        new_lon_max = 180
    # Ensure within [-90, 90] latitude extent and of desired size
    if new_lat_min < -90:
        new_lat_max = new_lat_min + (new_lat_min + 90)
        new_lat_min = -90
    if new_lat_max > 90:
        new_lat_min = new_lat_min - (new_lat_max - 90)
        new_lat_max = 90
    return extend_geographic_extent([new_lon_min, new_lon_max, new_lat_min, new_lat_max], padding=0)


def _is_crossing_dateline(lon: list | np.ndarray):
    """Check if the longitude array is crossing the dateline."""
    lon = np.asarray(lon)
    diff = np.diff(lon)
    return np.any(np.abs(diff) > 180)


def get_geographic_extent_from_xarray(
    xr_obj,
    padding: int | float | tuple | list = 0,
    size: int | float | tuple | list | None = None,
):
    """Get the geographic extent from an xarray object.

    Parameters
    ----------
    xr_obj : xarray.DataArray or xarray.Dataset
        xarray object.
    padding : int, float, tuple, list
        The number of degrees to extend the extent in each direction.
        If ``padding`` is a single number, the same padding is applied in all directions.
        If ``padding`` is a tuple or list, it must contain 2 or 4 elements.
        If two values are provided (x, y), they are interpreted as longitude and latitude padding, respectively.
        If four values are provided, they directly correspond to padding for each side ``(left, right, top, bottom)``.
        The default is ``0``.
    size : int, float, tuple, list
        The desired size in degrees of the extent in each direction.
        If ``size`` is a single number, the same size is enforced in all directions.
        If ``size`` is a tuple or list, it must of size 2 and specify the desired size of
        the extent in the x direction (longitude) and the y direction (latitude).
        The default is ``None``.

    Returns
    -------
    extent : tuple
        A tuple containing the longitude and latitude extent of the xarray object.
        The extent follows the matplotlib/cartopy format ``(xmin, xmax, ymin, ymax)``.

    """
    # TODO: should compute the corners and return based on the sides
    # TODO: check CRS is geographic
    padding = _check_padding(padding=padding)

    lon = xr_obj[xr_obj.gpm.x].to_numpy()
    lat = xr_obj[xr_obj.gpm.y].to_numpy()

    if _is_crossing_dateline(lon):
        raise NotImplementedError(
            "The object cross the dateline. The extent can't be currently defined.",
        )
    extent = Extent(np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat))
    extent = extend_geographic_extent(extent, padding=padding)
    if size is not None:
        extent = adjust_geographic_extent(extent, size=size)
    return extent


def get_geographic_extent_around_point(lon, lat, distance=None, size=None):
    """
    Get the geographic extent around a point.

    Either specify ``distance`` (in meters) or the wished extent ``size`` (in degrees).

    NOTE: this function is not yet designed to define an extent when the area of interest
    would cross the antimeridian or the poles.

    Parameters
    ----------
    lon : float
        Longitude of the point.
    lat : float
        Latitude of the point.
    distance: float
        Distance (in meters) from the point in each direction.
    size : int, float, tuple, list
        The size in degrees of the extent in each direction.
        If ``size`` is a single number, the same size is ensured in all directions.
        If ``size`` is a tuple or list, it must of size 2  and specifying
        the desired size of the extent in the x direction (longitude)
        and the y direction (latitude).

    Returns
    -------
    tuple
        The adjusted extent.

    """
    geod = pyproj.Geod(ellps="WGS84")
    if distance is not None and size is not None:
        raise ValueError("Either provide the 'distance' in meter or the 'size' of the extent in degrees.")
    if distance is None and size is None:
        raise ValueError("Please provide the 'distance' in meter or the 'size' of the extent in degrees.")
    if size is not None:
        return adjust_geographic_extent(extent=[lon, lon, lat, lat], size=size)
    # Define azimuths
    azimuths = [0, 90, 180, 270]  # north, east, south, west
    # Calculate new points in the four cardinal directions by the specified distance
    lons, lats, _ = geod.fwd(np.ones(4) * lon, np.ones(4) * lat, azimuths, np.ones(4) * distance, radians=False)
    extent = [lons.min().item(), lons.max().item(), lats.min().item(), lats.max().item()]
    return extend_geographic_extent(extent, padding=0)


def read_countries_extent_dictionary():
    """Reads a YAML file containing countries extent information and returns it as a dictionary.

    Returns
    -------
    dict
        A dictionary containing countries extent information.

    """
    countries_extent_filepath = os.path.join(
        _root_path,
        "gpm",
        "etc",
        "geospatial",
        "country_extent.yaml",
    )
    return read_yaml(countries_extent_filepath)


def read_continents_extent_dictionary():
    """Read and return a dictionary containing the extents of continents.

    Returns
    -------
    dict
        A dictionary containing the extents of continents.

    """
    continents_extent_filepath = os.path.join(
        _root_path,
        "gpm",
        "etc",
        "geospatial",
        "continent_extent.yaml",
    )
    return read_yaml(continents_extent_filepath)


def get_country_extent(name, padding=0.2):
    """Retrieves the extent of a country.

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
        return extend_geographic_extent(extent, padding=padding)
    # Identify possible match and raise error
    possible_match = difflib.get_close_matches(name, valid_countries, n=1, cutoff=0.6)
    if len(possible_match) == 0:
        raise ValueError("Provide a valid country name.")
    possible_match = possible_match[0]
    raise ValueError(f"No matching country. Maybe are you looking for '{possible_match}'?")


def get_continent_extent(name: str, padding: int | float | tuple | list = 0):
    """Retrieves the extent of a continent.

    Parameters
    ----------
    name : str
        The name of the continent.
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
        A tuple containing the longitude and latitude extent of the continent.

    Raises
    ------
    TypeError
        If the continent name is not provided as a string.
    ValueError
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
        return extend_geographic_extent(extent, padding=padding)
    # Identify possible match and raise error
    possible_match = difflib.get_close_matches(name, valid_continent, n=1, cutoff=0.6)
    if len(possible_match) == 0:
        raise ValueError(f"Provide a valid continent name from {valid_continent}.")
    possible_match = possible_match[0]
    raise ValueError(f"No matching continent. Maybe are you looking for '{possible_match}'?")


####------------------------------------------------------------------------------------.
#### Geographic crop


def crop(xr_obj, extent):
    """Crop a xarray object based on the provided bounding box.

    Parameters
    ----------
    xr_obj : xarray.DataArray or xarray.Dataset
        xarray object.
    extent : list or tuple
        The bounding box over which to crop the xarray object.
        `extent` must follow the matplotlib and cartopy extent conventions:
        extent = [x_min, x_max, y_min, y_max]

    Returns
    -------
    xr_obj : xarray.DataArray or xarray.Dataset
        Cropped xarray object.

    """
    if is_orbit(xr_obj):
        # - Subset only along_track
        list_isel_dicts = get_crop_slices_by_extent(xr_obj, extent)
        if len(list_isel_dicts) > 1:
            raise ValueError(
                "The orbit is crossing the extent multiple times. Use get_crop_slices_by_extent !.",
            )
        return xr_obj.isel(list_isel_dicts[0])
    if is_grid(xr_obj):
        isel_dict = get_crop_slices_by_extent(xr_obj, extent)
        return xr_obj.isel(isel_dict)
    # Otherwise raise informative error
    orbit_dims = ("cross_track", "along_track")
    grid_dims = ("lon", "lat")
    raise ValueError(f"Dataset not recognized. Expecting dimensions {orbit_dims} or {grid_dims}.")


def crop_by_country(xr_obj, name: str):
    """Crop an xarray object based on the specified country name.

    Parameters
    ----------
    xr_obj : xarray.DataArray or xarray.Dataset
        xarray object.
    name : str
        Country name.

    Returns
    -------
    xr_obj : xarray.DataArray or xarray.Dataset
        Cropped xarray object.

    """
    extent = get_country_extent(name)
    return crop(xr_obj=xr_obj, extent=extent)


def crop_by_continent(xr_obj, name: str):
    """Crop an xarray object based on the specified continent name.

    Parameters
    ----------
    xr_obj : xarray.DataArray or xarray.Dataset
        xarray object.
    name : str
        Continent name.

    Returns
    -------
    xr_obj : xarray.DataArray or xarray.Dataset
        Cropped xarray object.

    """
    extent = get_continent_extent(name)
    return crop(xr_obj=xr_obj, extent=extent)


def crop_around_point(xr_obj, lon: float, lat: float, distance=None, size=None):
    """Crop an xarray object around a point.

    Parameters
    ----------
    xr_obj : xarray.DataArray or xarray.Dataset
        xarray object.
    lon : float
        Longitude of the point.
    lat : float
        Latitude of the point.
    distance: float
        Distance (in meters) from the point in each direction.
    size : int, float, tuple, list
        The size in degrees of the extent in each direction.
        If ``size`` is a single number, the same size is ensured in all directions.
        If ``size`` is a tuple or list, it must of size 2  and specifying
        the desired size of the extent in the x direction (longitude)
        and the y direction (latitude).

    Returns
    -------
    xr_obj : xarray.DataArray or xarray.Dataset
        Cropped xarray object.

    """
    extent = get_geographic_extent_around_point(lon=lon, lat=lat, distance=distance, size=size)
    return crop(xr_obj=xr_obj, extent=extent)


@check_is_gpm_object
def get_crop_slices_by_extent(xr_obj, extent):
    """Compute the xarray object slices which are within the specified extent.

    If the input is a GPM Orbit, it returns a list of along-track slices
    If the input is a GPM Grid, it returns a dictionary of the lon/lat slices.

    Parameters
    ----------
    xr_obj : xarray.DataArray or xarray.Dataset
        xarray object.
    extent : list or tuple
        The extent over which to crop the xarray object.
        `extent` must follow the matplotlib and cartopy conventions:
        extent = [x_min, x_max, y_min, y_max]

    """
    # Retrieve spatial coordinates
    x = xr_obj.gpm.x
    y = xr_obj.gpm.y
    # If ORBIT
    if is_orbit(xr_obj):
        xr_obj = xr_obj.transpose("cross_track", "along_track", ...)
        lon = xr_obj[x].to_numpy()
        lat = xr_obj[y].to_numpy()
        idx_row, idx_col = np.where(
            (lon >= extent[0]) & (lon <= extent[1]) & (lat >= extent[2]) & (lat <= extent[3]),
        )
        if idx_col.size == 0:
            raise ValueError("No data inside the provided bounding box.")

        # Retrieve list of along_track slices isel_dict
        list_slices = get_list_slices_from_indices(idx_col)
        return [{"along_track": slc} for slc in list_slices]
    # If GRID
    lon = xr_obj[x].to_numpy()
    lat = xr_obj[y].to_numpy()
    idx_col = np.where((lon >= extent[0]) & (lon <= extent[1]))[0]
    idx_row = np.where((lat >= extent[2]) & (lat <= extent[3]))[0]
    # If no data in the bounding box in current granule, return empty list
    if idx_row.size == 0 or idx_col.size == 0:
        raise ValueError("No data inside the provided bounding box.")
    lat_slices = get_list_slices_from_indices(idx_row)[0]
    lon_slices = get_list_slices_from_indices(idx_col)[0]
    return {x: lon_slices, y: lat_slices}


def get_crop_slices_by_continent(xr_obj, name):
    """Compute the xarray object slices which are within the specified continent.

    If the input is a GPM Orbit, it returns a list of along-track slices.
    If the input is a GPM Grid, it returns a dictionary of the lon/lat slices.

    Parameters
    ----------
    xr_obj : xarray.DataArray or xarray.Dataset
        xarray object.
    name : str
        Continent name.

    """
    extent = get_continent_extent(name)
    return get_crop_slices_by_extent(xr_obj=xr_obj, extent=extent)


def get_crop_slices_by_country(xr_obj, name):
    """Compute the xarray object slices which are within the specified country.

    If the input is a GPM Orbit, it returns a list of along-track slices.
    If the input is a GPM Grid, it returns a dictionary of the lon/lat slices.

    Parameters
    ----------
    xr_obj : xarray.DataArray or xarray.Dataset
        xarray object.
    name : str
        Country name.

    """
    extent = get_country_extent(name)
    return get_crop_slices_by_extent(xr_obj=xr_obj, extent=extent)


def get_crop_slices_around_point(xr_obj, lon: float, lat: float, distance=None, size=None):
    """Compute the xarray object slices which are within the specified distance from a point.

    If the input is a GPM Orbit, it returns a list of along-track slices.
    If the input is a GPM Grid, it returns a dictionary of the lon/lat slices.

    Parameters
    ----------
    xr_obj : xarray.DataArray or xarray.Dataset
        xarray object.
    lon : float
        Longitude of the point.
    lat : float
        Latitude of the point.
    distance: float
        Distance (in meters) from the point in each direction.
    size : int, float, tuple, list
        The size in degrees of the extent in each direction.
        If ``size`` is a single number, the same size is ensured in all directions.
        If ``size`` is a tuple or list, it must of size 2  and specifying
        the desired size of the extent in the x direction (longitude)
        and the y direction (latitude).

    Returns
    -------
    xr_obj : xarray.DataArray or xarray.Dataset
        Cropped xarray object.

    """
    extent = get_geographic_extent_around_point(lon=lon, lat=lat, distance=distance, size=size)
    return get_crop_slices_by_extent(xr_obj=xr_obj, extent=extent)


####------------------------------------------------------------------------------------.
#### Miscellaneous


def unwrap_longitude_degree(x, period=360):
    """Unwrap longitude array."""
    x = np.asarray(x)
    mod = period / 2
    return (x + mod) % (2 * mod) - mod


def get_circle_coordinates_around_point(lon, lat, radius, num_vertices=360):
    """Get the coordinates of a circle with custom radius around a point.

    Parameters
    ----------
    lon : float
        Longitude of the point.
    lat : float
        Latitude of the point.
    radius : float
        Radius (in meters) around the point.
    num_vertices : int, optional
        Number of circle coordinates to return. The default is 360.

    Returns
    -------
    lons :  numpy.ndarray
        Longitude vertices of the circle around the point.
    lats :  numpy.ndarray
        Latitude vertices of the circle around the point.

    """
    geod = pyproj.Geod(ellps="WGS84")

    # Angle between each point in degrees
    angles = np.linspace(0, 360, num_vertices, endpoint=False)

    # Compute the coordinates of the circle's vertices
    lons, lats, _ = geod.fwd(
        np.ones(angles.shape) * lon,
        np.ones(angles.shape) * lat,
        angles,
        np.ones(angles.shape) * radius,
        radians=False,
    )
    return lons, lats


def get_great_circle_arc_endpoints(point, azimuth, distance):
    """Get great circle arc vertices.

    Calculate two points at a given distance from a central point in both the specified
    azimuth direction and its opposite direction along the great circle path.

    Parameters
    ----------
    point : tuple of float
        A tuple representing the middle point (longitude, latitude) of the great circle arc.
    azimuth : float
        The azimuth (in degrees) from the starting point. 0 correspond to the North. 180 to the South.
        The opposite direction will be automatically calculated as (azimuth + 180) % 360.
    distance : float
        The distance (in meters) to the points from the center point.

    Returns
    -------
    start_point : tuple of float
        The point (longitude, latitude) at the specified distance in the given azimuth direction.
    end_point : tuple of float
        The point (longitude, latitude) at the specified distance in the opposite azimuth direction.

    Examples
    --------
    >>> point = (-74.0060, 40.7128)  # New York City
    >>> azimuth = 90  # East
    >>> distance = 100000  # 100 km
    >>> get_great_circle_arc_endpoints(point, azimuth, distance)
    ((-72.54170804504108, 40.65355582184445), (-75.47074533517052, 40.77179828472569))
    """
    # Define the Geod object using the WGS84 ellipsoid (default)
    geod = pyproj.Geod(ellps="WGS84")
    opposite_azimuth = (azimuth + 180) % 360
    # Calculate the destination point
    lon, lat = point
    lon1, lat1, _ = geod.fwd(lon, lat, az=azimuth, dist=distance, radians=False)
    lon2, lat2, _ = geod.fwd(lon, lat, az=opposite_azimuth, dist=distance, radians=False)
    start_point = (lon1, lat1)
    end_point = (lon2, lat2)
    return start_point, end_point


def get_geodesic_line(start_point, end_point, steps, geod=None):
    """Construct a geodesic path between two points.

    This function acts as a wrapper for the geodesic construction available in ``pyproj``.

    Parameters
    ----------
    start_point: tuple
       A longitude-latitude pair designating the start point of the cross section (units are
       degrees east and degrees north).
    end_point: tuple
       A longitude-latitude pair designating the end point of the cross section (units are
       degrees east and degrees north).
    steps: int, optional
       The number of points along the geodesic between the start and the end point
       (including the end points) to use in the cross section.

    Returns
    -------
    numpy.ndarray
        The list of x, y points in the given CRS of length `steps` along the geodesic.

    See Also
    --------
    :py:class:`gpm.utils.manipulations.extract_transect_at_points` and
    :py:class:`gpm.utils.manipulations.extract_transect_between_points`.

    """
    if geod is None:
        geod = pyproj.Geod(ellps="WGS84")

    # Geod.npts only gives points *in between* the start_point and end_point
    vertices = np.concatenate(
        [
            np.array(start_point)[None],
            np.array(geod.npts(start_point[0], start_point[1], end_point[0], end_point[1], steps - 2)),
            np.array(end_point)[None],
        ],
    )
    return vertices
