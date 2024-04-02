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
"""This module contains functions to define and create CF-compliant CRS."""
import warnings

import numpy as np
import xarray as xr
from xarray import Variable

# If the longitude, latitude, vertical or time coordinate is multi-valued,
# varies in only one dimension, and varies independently of other spatiotemporal coordinates,
# it is not permitted to store it as an auxiliary coordinate variable.
# This is both to enhance conformance to COARDS and to facilitate the
# use of generic applications that recognize the NUG convention for coordinate variables.
# An application that is trying to find the latitude coordinate of a variable
# should always look first to see if any of the variable's dimensions correspond to a latitude coordinate variable.
# If the latitude coordinate is not found this way, then the auxiliary coordinate variables
# listed by the coordinates attribute should be checked.
# Note that it is permissible, but optional, to list coordinate variables
# as well as auxiliary coordinate variables in the coordinates attribute
# --> https://github.com/pydata/xarray/issues/6310
# --> https://github.com/pydata/xarray/pull/6366
# --> Coordinates in encoding ? Why ?
# --> Rioxarray save grid_mapping in the encoding !

# --> In https://cfconventions.org/cf-conventions/cf-conventions.html#coordinate-system, search for: coordinates =

# --> https://github.com/corteva/rioxarray/pull/284
# --> https://github.com/opendatacube/datacube-core/issues/1084

# Others work:
# - https://github.com/dcherian/xindexes/blob/main/crsindex.ipynb
# - https://github.com/xarray-contrib/cf-xarray
# - https://cf-xarray.readthedocs.io/en/latest/grid_mappings.html
# - https://cf-xarray.readthedocs.io/en/latest/roadmap.html

# ------------------------------------------------------------------------------.
#### CF-Writers
## coord attributes
# - axis if projection
# - units, standard_name, long_name
# - GEO_exception:
#   --> For geostationary crs save radians instead of x, y ? or meters?
#   --> In _get_projection_coords_attrs, define units specification

## var attributes
# AreaDefinition
# - Only projection coordinates
#   grid_mapping = "crsOSGB"
#   grid_mapping = "crsOSGB: x y"

# - Projection coordinates + WGS84 coordinates
#   coordinates = "lat lon"
#   grid_mapping = "crsOSGB: x y crsWGS84: lat lon"

# - SwathDefinition
#   coordinates = "lat lon"
#   grid_mapping = "crsWGS84"

# ------------------------------------------------------------------------------.
#### CF-Readers
# - If CRS not geographic
#   --> AreaDefinition
#   --> If geostationary,
#       - If x,y units = meter, deal with that
#       - If x,y units = radian (or radians), do as now
# - If CRS is geographic
#   - If dimensional coordinates are 1D --> AreaDefinition
#   - If dimensional coordinates are 2D --> Swath Definition

# - If two CRS specified (projection and crs), returns AreaDefinition

####---------------------------------------------------------------------------.
#### CF-Writers utility


def _get_pyproj_crs_cf_dict(crs):
    """Return the CF-compliant CRS attributes."""
    # Retrieve CF-compliant CRS information
    attrs = crs.to_cf()
    # Add spatial_ref for compatibility with GDAL
    #  - GDAL: Only OGC WKT GEOGCS and PROJCS Projections supported
    #  - PYPROJ write crs_wkt starting with GEOGCRS instead of GEOGCS !
    if "crs_wkt" in attrs:
        spatial_ref = attrs["crs_wkt"]
        spatial_ref = spatial_ref.replace("GEOGCRS", "GEOGCS")
        attrs["spatial_ref"] = spatial_ref
    # Return attributes
    return attrs


def _get_proj_coord_unit(crs, dim):
    """Return the coordinate unit of a projected CRS."""
    axis_info = crs.axis_info[dim]
    if hasattr(axis_info, "unit_conversion_factor"):
        unit_factor = axis_info.unit_conversion_factor
        return f"{unit_factor} metre" if unit_factor != 1 else "metre"
    return None


def _get_obj(ds, inplace=False):
    """Return a dataset copy if inplace=False."""
    if inplace:
        return ds
    return ds.copy(deep=True)


def _get_dataarray_with_spatial_dims(xr_obj):
    # TODO: maybe select variable with spatial dimensions !
    if isinstance(xr_obj, xr.Dataset):
        variables = list(xr_obj.data_vars)
        if len(variables) == 0:
            raise ValueError("The dataset has no variables")
        var = variables[0]
        xr_obj = xr_obj[var]
    return xr_obj


def _get_geographic_coord_attrs():
    """Get coordinates attributes of geographic crs."""
    # X or lon coordinate
    x_coord_attrs = {}
    x_coord_attrs["long_name"] = "longitude"
    x_coord_attrs["standard_name"] = "longitude"
    x_coord_attrs["units"] = "degrees_east"
    x_coord_attrs["coverage_content_type"] = "coordinate"
    # Y or latitude coordinate
    y_coord_attrs = {}
    y_coord_attrs["long_name"] = "latitude"
    y_coord_attrs["standard_name"] = "latitude"
    y_coord_attrs["units"] = "degrees_north"
    y_coord_attrs["coverage_content_type"] = "coordinate"
    return x_coord_attrs, y_coord_attrs


def _get_projection_coords_attrs(crs):
    """Get projection coordinates attributes of projected crs."""
    # Add X metadata
    x_coord_attrs = {}
    x_coord_attrs["axis"] = "X"
    x_coord_attrs["long_name"] = "x coordinate of projection"
    x_coord_attrs["standard_name"] = "projection_x_coordinate"
    x_coord_attrs["coverage_content_type"] = "coordinate"
    units = _get_proj_coord_unit(crs, dim=0)
    if units:
        x_coord_attrs["units"] = units
    # Add Y metadata
    y_coord_attrs = {}
    y_coord_attrs["axis"] = "Y"
    y_coord_attrs["long_name"] = "y coordinate of projection"
    y_coord_attrs["standard_name"] = "projection_y_coordinate"
    y_coord_attrs["coverage_content_type"] = "coordinate"
    units = _get_proj_coord_unit(crs, dim=1)
    if units:
        y_coord_attrs["units"] = units
    return x_coord_attrs, y_coord_attrs


def _get_proj_dim_coords(xr_obj):
    """Determine the spatial 1D dimensions of the `xarray.DataArray`.

    Parameters
    ----------
    xr_obj : xr.Dataset or xr.DataArray

    Returns
    -------
    (x_dim, y_dim) tuple
        Tuple with the name of the spatial dimensions.

    """
    # Check for classical options
    list_options = [("x", "y"), ("lon", "lat"), ("longitude", "latitude")]
    for dims in list_options:
        dim_x = dims[0]
        dim_y = dims[1]
        if (
            dim_x in xr_obj.coords
            and dim_y in xr_obj.coords
            and xr_obj[dim_x].dims == (dim_x,)
            and xr_obj[dim_y].dims == (dim_y,)
        ):
            return dims
    # Otherwise look at available coordinates, and search for CF attributes
    x_dim = None
    y_dim = None
    coords_names = list(xr_obj.coords)
    for coord in coords_names:
        # Select only 1D coordinates with dimension name as the coordinate
        if xr_obj[coord].dims != (coord,):
            continue
        # Retrieve coordinate attribute
        attrs = xr_obj[coord].attrs
        if (attrs.get("axis", "").upper() == "X") or (
            attrs.get("standard_name", "").lower() in ("longitude", "projection_x_coordinate")
        ):
            x_dim = coord
        elif (attrs.get("axis", "").upper() == "Y") or (
            attrs.get("standard_name", "").lower() in ("latitude", "projection_y_coordinate")
        ):
            y_dim = coord
    return x_dim, y_dim


def _get_swath_dim_coords(xr_obj):
    """Determine the spatial 1D dimensions of the `xarray.DataArray`.

    Cases:
    - 2D x/y coordinates with 2 dimensions (along-track-cross-track scan)
    - 1D x/y coordinates with 1 dimensions (along-track nadir scan)

    Parameters
    ----------
    xr_obj : xr.Dataset or xr.DataArray

    Returns
    -------
    (x_dim, y_dim) tuple
        Tuple with the name of the swath coordinates.

    """
    # Check for classical options
    list_options = [("lon", "lat"), ("longitude", "latitude")]
    for coords in list_options:
        coords_x = coords[0]
        coords_y = coords[1]
        if coords_x in xr_obj.coords and coords_y in xr_obj.coords:
            dims_x = list(xr_obj[coords_x].dims)
            dims_y = list(xr_obj[coords_y].dims)
            # 2D swath
            if len(dims_x) == 2 and len(dims_y) == 2 and dims_x == dims_y:
                return coords_x, coords_y
            # Nadir-scan
            if len(dims_x) == 1 and len(dims_y) == 1 and dims_x == dims_y:
                return coords_x, coords_y

    # Otherwise look at available coordinates, and search for CF attributes
    # --> Look if coordinate has  2D dimension
    # --> Look if coordinate has standard_name "longitude" or "latitude"
    coords_x = None
    coords_y = None
    coords_names = list(xr_obj.coords)
    for coord in coords_names:
        # Select only lon/lat swath coordinates with dimension like ('y','x')
        if len(xr_obj[coord].dims) != 2:  # ('y', 'x'), ('cross_track', 'along_track')
            continue
        attrs = xr_obj[coord].attrs
        if attrs.get("standard_name", "INVALID").lower() in ("longitude"):
            coords_x = coord
        elif attrs.get("standard_name", "INVALID").lower() in ("latitude"):
            coords_y = coord
    return coords_x, coords_y


def has_swath_coords(xr_obj):
    lon_coord, lat_coord = _get_swath_dim_coords(xr_obj)
    return bool(lon_coord is not None and lat_coord is not None)


def has_proj_coords(xr_obj):
    x_coord, y_coord = _get_proj_dim_coords(xr_obj)
    return bool(x_coord is not None and y_coord is not None)


def _add_swath_coords_attrs(ds, crs) -> xr.Dataset:
    """Add CF-compliant CRS attributes to the coordinates of a swath.

    Parameters
    ----------
    ds : xarray.Dataset
    crs : :py:class:`~pyproj.crs.CoordinateSystem`
        CRS information to be added to the xr.Dataset

    Returns
    -------
    :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
        Dataset with CF-compliant dimension coordinate attributes

    """
    if not crs.is_geographic:
        raise ValueError("A swath require a geographic CRS.")
    # Get swath coordinates
    lon_coord, lat_coord = _get_swath_dim_coords(ds)
    # Retrieve existing coordinate attributes
    src_lon_coord_attrs = dict(ds[lon_coord].attrs)
    src_lat_coord_attrs = dict(ds[lat_coord].attrs)
    # Drop axis if present (should not be added to swath objects)
    src_lon_coord_attrs.pop("axis", None)
    src_lat_coord_attrs.pop("axis", None)
    # Get geographic coordinate attributes
    lon_coord_attrs, lat_coord_attrs = _get_geographic_coord_attrs()
    # Update attributes
    src_lon_coord_attrs.update(lon_coord_attrs)
    src_lat_coord_attrs.update(lat_coord_attrs)
    # Add updated coordinate attributes
    ds[lon_coord].attrs = src_lon_coord_attrs
    ds[lat_coord].attrs = src_lat_coord_attrs
    return ds


def _add_proj_coords_attrs(ds, crs) -> xr.Dataset:
    """Add CF-compliant attributes to the dimension coordinates of a projected CRS.

    Note: The WGS84 grid is considered a projected CRS here !

    Parameters
    ----------
    ds : xarray.Dataset
    crs : :py:class:`~pyproj.crs.CoordinateSystem`
        CRS information to be added to the xr.Dataset

    Returns
    -------
    :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
        Dataset with CF-compliant dimension coordinate attributes

    """
    # Retrieve CRS information
    is_projected = crs.is_projected
    is_geographic = crs.is_geographic

    # Identify spatial dimension coordinates
    x_dim, y_dim = _get_proj_dim_coords(ds)

    # If available, add attributes
    if x_dim is not None and y_dim is not None:
        # Retrieve existing coordinate attributes
        src_x_coord_attrs = dict(ds[x_dim].attrs)
        src_y_coord_attrs = dict(ds[y_dim].attrs)

        # Attributes for projected CRS
        if is_projected:
            # Get projection coordinate attributes
            x_coord_attrs, y_coord_attrs = _get_projection_coords_attrs(crs)
            # If unit is already present, do not overwrite it !
            # --> Example: for compatibility with GEOS area (metre or radians)
            if "units" in src_x_coord_attrs:
                _ = x_coord_attrs.pop("units", None)
                _ = y_coord_attrs.pop("units", None)
            # Update attributes
            src_x_coord_attrs.update(x_coord_attrs)
            src_y_coord_attrs.update(y_coord_attrs)

        # Attributes for geographic CRS
        elif is_geographic:
            # Get geographic coordinate attributes
            x_coord_attrs, y_coord_attrs = _get_geographic_coord_attrs()
            # Add axis attribute
            x_coord_attrs["axis"] = "X"
            y_coord_attrs["axis"] = "Y"
            # Update attributes
            src_x_coord_attrs.update(x_coord_attrs)
            src_y_coord_attrs.update(y_coord_attrs)

        # Add coordinate attributes
        ds.coords[x_dim].attrs = src_x_coord_attrs
        ds.coords[y_dim].attrs = src_y_coord_attrs

    return ds


def _add_coords_crs_attrs(ds, crs):
    """Add CF-compliant attributes to the CRS dimension coordinates.

    Parameters
    ----------
    ds : xarray.Dataset
    crs : :py:class:`~pyproj.crs.CoordinateSystem`
        CRS information to be added to the xr.Dataset

    Returns
    -------
    :obj:`xarray.Dataset` | :obj:`xarray.DataArray`:
        Dataset with CF-compliant dimension coordinate attributes

    """
    # Projected CRS
    if crs.is_projected:
        ds = _add_proj_coords_attrs(ds, crs)
    # Geographic CRS
    else:
        ds = _add_swath_coords_attrs(ds, crs) if has_swath_coords(ds) else _add_proj_coords_attrs(ds, crs)
    return ds


def _add_crs_coord(ds, crs, grid_mapping_name="spatial_ref"):
    """Add ``name`` coordinate derived from :py:class:`pyproj.crs.CoordinateSystem`.

    Parameters
    ----------
    ds : xarray.Dataset
    crs : :py:class:`~pyproj.crs.CoordinateSystem`
        CRS information to be added to the xr.Dataset
    grid_mapping_name : str
        Name of the grid_mapping coordinate to store the CRS information
        The default is ``spatial_ref``.
        Other common names are ``grid_mapping`` and ``crs``.

    Returns
    -------
    ds : xarray.Dataset
        Dataset including the CRS ``name`` coordinate.

    """
    spatial_ref = Variable((), 0)
    # Retrieve CF-compliant CRS dictionary information
    attrs = _get_pyproj_crs_cf_dict(crs)
    # Add attributes to CRS variable
    spatial_ref.attrs.update(attrs)
    # Add the CRS coordinate to the xr.Dataset
    return ds.assign_coords({grid_mapping_name: spatial_ref})


def _grid_mapping_reference(ds, crs, grid_mapping_name):
    """Define the grid_mapping value to attach to the variables."""
    # Projected CRS
    if crs.is_projected:
        x_dim, y_dim = _get_proj_dim_coords(ds)
        if x_dim is None or y_dim is None:
            warnings.warn("Projection coordinates are not present.", stacklevel=3)
            output = f"{grid_mapping_name}"
        else:
            output = f"{grid_mapping_name}: {x_dim} {y_dim}"
    # Geographic CRS
    elif has_swath_coords(ds):
        lon_coord, lat_coord = _get_swath_dim_coords(ds)
        output = f"{grid_mapping_name}: {lat_coord} {lon_coord}"
    # GRID - 1D x/y with 2 dimensions
    else:
        x_dim, y_dim = _get_proj_dim_coords(ds)
        if x_dim is None or y_dim is None:
            warnings.warn("Projection coordinates are not present.", stacklevel=3)
            output = f"{grid_mapping_name}"
        output = f"{grid_mapping_name}: {x_dim} {y_dim}"
    return output


def _simplify_grid_mapping_value(grid_mapping_value):
    """Simplify grid_mapping value.

    GDAL does not support grid_mapping defined as "crs_wgs84: lat lon"
    If only 1 CRS is specified in such format, it returns "crs_wgs84"
    """
    n_crs = grid_mapping_value.count(":")
    if n_crs == 1:
        grid_mapping_value = grid_mapping_value.split(":")[0]
    return grid_mapping_value


def simplify_grid_mapping_values(ds):
    """Simplify grid_mapping value.

    GDAL does not support grid_mapping defined as "crs_wgs84: lat lon"
    If only 1 CRS is specified in such format, it returns "crs_wgs84"
    """
    keys = list(ds.coords) + list(ds.data_vars)
    for key in keys:
        if "grid_mapping" in ds[key].attrs:
            ds[key].attrs["grid_mapping"] = _simplify_grid_mapping_value(
                ds[key].attrs["grid_mapping"],
            )
    return ds


def _get_spatial_coordinates(xr_obj):
    """Return the spatial coordinates."""
    coords1 = _get_proj_dim_coords(xr_obj)
    coords2 = _get_swath_dim_coords(xr_obj)
    return [coord for coord in (coords1 + coords2) if coord is not None]


def _get_spatial_dims(xr_obj):
    """Return the spatial dimensions."""
    coords = _get_spatial_coordinates(xr_obj)  # can be []
    list_dims = []
    for coord in coords:
        _ = [list_dims.append(dim) for dim in list(xr_obj[coord].dims)]
    return list(set(list_dims))


def _get_variables_with_spatial_dims(ds):
    """Return variables and coordinates depending on spatial dimensions.

    A coordinate with dimension (time, y, x) is selected
    The spatial coordinates (y, x) or (latitude, longitude) are not selected.
    """
    spatial_dims = _get_spatial_dims(ds)  # can be []
    spatial_dims = set(spatial_dims)
    if len(spatial_dims) == 0:
        raise ValueError("No spatial dimension identified in the dataset.")
    variables = list(ds.coords) + list(ds.data_vars)
    list_spatial_variables = [var for var in variables if set(ds[var].dims).issuperset(spatial_dims)]
    # Remove spatial coordinates from the list
    coords = _get_spatial_coordinates(ds)
    return set(list_spatial_variables).difference(coords)


def _add_variables_crs_attrs(ds, crs, grid_mapping_name):
    """Add 'grid_mapping' and 'coordinates' (for swath only) attributes to the variable.

    If a grid_mapping attribute is already existing, add also the new one !
    """
    # Retrieve grid_mapping attributes value
    grid_mapping_value = _grid_mapping_reference(ds, crs=crs, grid_mapping_name=grid_mapping_name)

    # Retrieve variables (and coordinates) requiring the crs attribute
    variables = _get_variables_with_spatial_dims(ds)

    for var in variables:
        grid_mapping = ds[var].attrs.get("grid_mapping", "")
        grid_mapping = grid_mapping_value if grid_mapping == "" else grid_mapping + " " + grid_mapping_value
        ds[var].attrs["grid_mapping"] = grid_mapping

    # Add coordinates attribute if swath data
    # --> If added to attrs, to_netcdf move it to encoding !
    if has_swath_coords(ds):
        lon_coord, lat_coord = _get_swath_dim_coords(ds)
        for var in variables:
            da_coords = ds[var].coords
            if lon_coord in da_coords and lat_coord in da_coords:
                ds[var].encoding["coordinates"] = f"{lat_coord} {lon_coord}"
    return ds


def _get_name_existing_crs_coords(xr_obj):
    """Return a list with the name of CRS coordinates."""
    return [
        coord
        for coord in xr_obj.coords
        if "crs_wkt" in xr_obj[coord].attrs or "grid_mapping_name" in xr_obj[coord].attrs
    ]


def remove_existing_crs_info(ds):
    """Remove existing grid_mapping attributes."""
    for var in ds.data_vars:
        _ = ds[var].attrs.pop("grid_mapping", None)
        _ = ds[var].attrs.pop("coordinates", None)
        _ = ds[var].encoding.pop("coordinates", None)
    crs_coords = _get_name_existing_crs_coords(ds)
    return ds.drop_vars(crs_coords)


def set_dataset_single_crs(ds, crs, grid_mapping_name="spatial_ref", inplace=False):
    """Add CF-compliant CRS information to an xr.Dataset.

    It assumes all dataset variables have same CRS !
    For projected CRS, it expects that the CRS dimension coordinates are specified.
    For swath dataset, it expects that the geographic coordinates are specified.

    Parameters
    ----------
    ds : xarray.Dataset
    crs : :py:class:`~pyproj.crs.CoordinateSystem`
        CRS information to be added to the xr.Dataset
    grid_mapping_name : str
        Name of the grid_mapping coordinate to store the CRS information
        The default is ``spatial_ref``.
        Other common names are ``grid_mapping`` and ``crs``.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with CF-compliant CRS information.

    """
    # Get dataset copy if inplace=False
    ds = _get_obj(ds=ds, inplace=inplace)

    # Add coordinate with CRS information
    ds = _add_crs_coord(ds, crs=crs, grid_mapping_name=grid_mapping_name)

    # Add CF attributes to CRS coordinates
    ds = _add_coords_crs_attrs(ds, crs=crs)

    # Add CF attributes 'grid_mapping' (and 'coordinates' for swath)
    # - To relevant variables and coordinates
    return _add_variables_crs_attrs(ds=ds, crs=crs, grid_mapping_name=grid_mapping_name)


def set_dataset_crs(ds, crs, grid_mapping_name="spatial_ref", inplace=False):
    """Add CF-compliant CRS information to an xr.Dataset.

    It assumes all dataset variables have same CRS !
    For projected CRS, it expects that the CRS dimension coordinates are specified.
    For swath dataset, it expects that the geographic coordinates are specified.

    For projected CRS, if 2D latitude/longitude arrays are specified,
    it assumes they refer to the WGS84 CRS !

    Parameters
    ----------
    ds : xarray.Dataset
    crs : :py:class:`~pyproj.crs.CoordinateSystem`
        CRS information to be added to the xr.Dataset
    grid_mapping_name : str
        Name of the grid_mapping coordinate to store the CRS information
        The default is ``spatial_ref``.
        Other common names are ``grid_mapping`` and ``crs``.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with CF-compliant CRS information.

    """
    import pyproj

    ds = remove_existing_crs_info(ds)
    ds = set_dataset_single_crs(
        ds=ds,
        crs=crs,
        grid_mapping_name=grid_mapping_name,
        inplace=inplace,
    )
    # If CRS is projected and 2D lon/lat are available, also add the WGS84 CRS
    if crs.is_projected and has_swath_coords(ds):
        crs_wgs84 = pyproj.CRS(proj="longlat", ellps="WGS84")
        ds = set_dataset_single_crs(
            ds=ds,
            crs=crs_wgs84,
            grid_mapping_name="crsWGS84",
            inplace=inplace,
        )
    # Simplify grid_mapping if possible
    # - For compatibility with GDAL, if only 1 CRS is specified !
    return simplify_grid_mapping_values(ds)


####---------------------------------------------------------------------------.
#### TODO: Writers
# coord attributes
# --> For geostationary crs save radians instead of x, y ? or meters?
# --> In _get_projection_coords_attrs, define units specification


# def _add_crs_proj_coords(ds, crs, width, height, affine):
#     # https://github.com/corteva/rioxarray/blob/master/rioxarray/rioxarray.py#L118
#     # TODO:
#     # for AreaDefinition. x_coord, y_coord = area_dej.get_proj_coords()
#     # for SwathDefinition: do nothing because expect already present

#     #  _generate_spatial_coords
#     # https://github.com/corteva/rioxarray/blob/master/rioxarray/rioxarray.py#L118
#     pass


####---------------------------------------------------------------------------.
#### CF-Readers


def _get_crs_coordinates(xr_obj):
    """Return a list with the name(s) of the CRS coordinate(s)."""
    crs_attributes = ["grid_mapping_name", "crs_wkt", "spatial_ref"]
    return [coord for coord in xr_obj.coords if np.any(np.isin(crs_attributes, list(xr_obj[coord].attrs)))]


def _get_list_pyproj_crs(xr_obj):
    """Return a list of pyproj specified CRS."""
    import pyproj

    list_crs_names = _get_crs_coordinates(xr_obj)
    if len(list_crs_names) == 0:
        raise ValueError("No CRS coordinate in the dataset.")
    return [pyproj.CRS.from_cf(xr_obj[crs_coord].attrs) for crs_coord in list_crs_names]


def _get_geographic_crs(xr_obj):
    list_crs = _get_list_pyproj_crs(xr_obj)
    list_geographic_crs = [proj_crs for proj_crs in list_crs if proj_crs.is_geographic]
    if len(list_geographic_crs) > 1:
        raise ValueError("More than 1 geographic CRS is specified")
    if len(list_geographic_crs) == 0:
        raise ValueError("No geographic CRS is specified")
    return list_geographic_crs[0]


def get_pyproj_crs(xr_obj):
    """Return :py:class:`pyproj.crs.CoordinateSystem` from CRS coordinate(s).

    If a geographic and projected CRS are present, it returns the projected.

    Parameters
    ----------
    xr_obj : xarray.Dataset or xarray.DataArray

    Returns
    -------
    proj_crs : :py:class:`~pyproj.crs.CoordinateSystem`

    """
    list_crs = _get_list_pyproj_crs(xr_obj)
    # If two crs are provided, select the projected !
    if len(list_crs) == 2:
        list_crs = [proj_crs for proj_crs in list_crs if proj_crs.is_projected]
        if len(list_crs) != 1:
            raise ValueError("A DataArray should have only 1 projected CRS.")
    # Return crs
    return list_crs[0]


def get_pyresample_swath(xr_obj):
    """Get pyresample SwathDefinition from CF-compliant xarray object."""
    from pyresample import SwathDefinition

    if not has_swath_coords(xr_obj):
        raise ValueError("Not a swath object.")

    # Identify name of longitude and latitude coordinates
    lons, lats = _get_swath_dim_coords(xr_obj)

    # Retrieve geographic CRS if available
    try:
        pyproj_crs = _get_geographic_crs(xr_obj)
    except Exception:
        pyproj_crs = None

    # Define SwathDefinition
    # - with xr.DataArray lat/lons
    # - Otherwise fails https://github.com/pytroll/satpy/issues/1434
    # - In old pyresample versions requiring
    #   lons = np.ascontiguousarray(xr_obj[lons].data),
    #   lats = np.ascontiguousarray(xr_obj[lons].data)
    # to avoid 'ndarray is not C-contiguous' when resampling !
    da_lons = xr.DataArray(xr_obj[lons].data, dims=["y", "x"])
    da_lats = xr.DataArray(xr_obj[lats].data, dims=["y", "x"])

    return SwathDefinition(da_lons, da_lats, crs=pyproj_crs)


def _compute_extent(x_coords, y_coords):
    """Compute the extent (x_min, x_max, y_min, y_max) from the pixel centroids in x and y coordinates.

    This function assumes that the spacing between each pixel is uniform.
    """
    # Calculate the pixel size assuming uniform spacing between pixels
    pixel_size_x = (x_coords[-1] - x_coords[0]) / (len(x_coords) - 1)
    pixel_size_y = (y_coords[-1] - y_coords[0]) / (len(y_coords) - 1)

    # Adjust min and max to get the corners of the outer pixels
    x_min, x_max = x_coords[0] - pixel_size_x / 2, x_coords[-1] + pixel_size_x / 2
    y_min, y_max = y_coords[0] - pixel_size_y / 2, y_coords[-1] + pixel_size_y / 2

    return [x_min, x_max, y_min, y_max]


def get_pyresample_projection(xr_obj):
    """Get pyresample AreaDefinition from CF-compliant xarray object."""
    from pyresample import AreaDefinition

    if not has_proj_coords(xr_obj):
        raise ValueError("Not a GRID object.")

    # Identify name of longitude and latitude coordinates
    x, y = _get_proj_dim_coords(xr_obj)

    # Retrieve geographic CRS if available
    pyproj_crs = get_pyproj_crs(xr_obj)

    # Retrieve x and y projection coordinates
    x_coords = xr_obj[x].compute().data
    y_coords = xr_obj[y].compute().data

    # Retrieve shape
    shape = (y_coords.shape[0], x_coords.shape[0])

    # - Derive extent
    extent = _compute_extent(x_coords=x_coords, y_coords=y_coords)

    # Define SwathDefinition
    # area_def = AreaDefinition(crs=pyproj_crs, shape=shape, area_extent=extent)
    return AreaDefinition(
        "GRID_area",
        "GRID_area",
        "",
        pyproj_crs,
        width=shape[1],
        height=shape[0],
        area_extent=extent,
    )


def get_pyresample_area(xr_obj):
    """Define pyresample area from CF-compliant xarray object.

    To be used by the pyresample accessor: ds.pyresample.area
    """
    if has_proj_coords(xr_obj):
        return get_pyresample_projection(xr_obj)
    if has_swath_coords(xr_obj):
        return get_pyresample_swath(xr_obj)
    raise ValueError("Impossible to infer if a SwathDefinition or AreaDefinition.")
