#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:27:48 2023

@author: ghiggi
"""
import warnings
from xarray import Variable
import xarray as xr 
import pyproj 

#------------------------------------------------------------------------------.
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

#------------------------------------------------------------------------------.
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

#------------------------------------------------------------------------------.

def get_pyproj_crs(ds):
    """Return :py:class:`pyproj.crs.CoordinateSystem` from CRS coordinate. 
    
    It search for ``spatial_ref``, ``crs`` and ``grid_mapping`` named coordinate.
    
    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    proj_crs : :py:class:`~pyproj.crs.CoordinateSystem`
    """
    
    if "spatial_ref" in ds:
        proj_crs = pyproj.CRS.from_cf(ds["spatial_ref"].attrs)
    elif "crs" in ds: 
        proj_crs = pyproj.CRS.from_cf(ds["crs"].attrs)
    elif "grid_mapping" in ds: 
        proj_crs = pyproj.CRS.from_cf(ds["grid_mapping"].attrs)
    else: 
        raise ValueError("No CRS coordinate in the dataset.")
    return proj_crs


def _get_pyproj_crs_cf_dict(crs):
    """Return the CF-compliant CRS attributes."""
    # Retrieve CF-compliant CRS information 
    attrs = crs.to_cf()
    # Add spatial_ref for compatibility with GDAL 
    attrs["spatial_ref"] = attrs["crs_wkt"]
    # Return attributes
    return attrs 


def _get_proj_coord_unit(crs, dim):
    "Return the coordinate unit of a projected CRS."
    axis_info = crs.axis_info[dim]
    units = None
    if hasattr(axis_info, "unit_conversion_factor"):
        unit_factor = axis_info.unit_conversion_factor
        if unit_factor != 1:
            units = f"{unit_factor} metre"
        else:
            units = "metre"
        return units
     
    
def _get_obj(ds, inplace=False):
    """Return a dataset copy if inplace=False."""
    if inplace:
        return ds
    else:
        return  ds.copy(deep=True)


def _get_dataarray_with_spatial_dims(xr_obj):
    # TODO: maybe select variable with spatial dimensions ! 
    # --> Exclude dimensions with datetime and integer dtype coordinates?
    if isinstance(xr_obj, xr.Dataset): 
        var = list(xr_obj.data_vars)[0] 
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
    """Determine the spatial 1D dimensions of the `xarray.DataArray`
    
    Parameters
    ----------
    da : :obj:`xarray.DataArray`
    
    Returns
    -------
    (x_dim, y_dim) tuple 
        Tuple with the name of the spatial dimensions.
    """
    # Retrieve DataArray 
    xr_obj = _get_dataarray_with_spatial_dims(xr_obj)
       
    # Check for classical options
    list_options = [("x","y"), ("lon", "lat"), ("longitude", "latitude")]
    for dims in list_options: 
        dim_x = dims[0]
        dim_y = dims[1]
        if dim_x in xr_obj.coords and dim_y in xr_obj.coords:
            if xr_obj[dim_x].dims == (dim_x,) and xr_obj[dim_y].dims == (dim_y,):
                return dims
    # Otherwise look at available coordinates, and search for CF attributes 
    else:
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
                attrs.get("standard_name", "").lower() in ("longitude", "projection_x_coordinate")):
                 x_dim = coord
            elif (attrs.get("axis", "").upper() == "Y") or (
                  attrs.get("standard_name", "").lower() in ("latitude", "projection_y_coordinate")):
                 y_dim = coord
    return x_dim, y_dim 


def _get_swath_dim_coords(xr_obj):
    """Determine the spatial 1D dimensions of the `xarray.DataArray`
    
    Parameters
    ----------
    da : :obj:`xarray.DataArray`
    
    Returns
    -------
    (x_dim, y_dim) tuple 
        Tuple with the name of the swath coordinates.
    """
    # Retrieve DataArray 
    xr_obj = _get_dataarray_with_spatial_dims(xr_obj)
       
    # Check for classical options
    list_options = [("lon", "lat"), ("longitude", "latitude")]
    for dims in list_options: 
        dim_x = dims[0]
        dim_y = dims[1]
        if dim_x in xr_obj.coords and dim_y in xr_obj.coords:
            if len(xr_obj[dim_x].dims) >= 2 and len(xr_obj[dim_y].dims) >= 2:
                return dim_x, dim_y 

    # Otherwise look at available coordinates, and search for CF attributes
    # --> Look if coordinate has  2D dimension
    # --> Look if coordinate has standard_name "longitude" or "latitude"
    else:
        x_dim = None
        y_dim = None 
        coords_names = list(xr_obj.coords)
        for coord in coords_names:
            # Select only lon/lat swath coordinates with dimension like ('y','x')   
            if len(xr_obj[coord].dims) <= 1: # ('y', 'x'), ('cross_track', 'along_track')
                continue
            attrs = xr_obj[coord].attrs
            if attrs.get("standard_name", "").lower() in ("longitude"):
                 x_dim = coord
            elif attrs.get("standard_name", "").lower() in ("latitude"):
                 y_dim = coord
    return x_dim, y_dim 


def has_swath_coords(xr_obj):
    lon_coord, lat_coord = _get_swath_dim_coords(xr_obj)
    if lon_coord is not None and lat_coord is not None: 
        return True
    else: 
        return False 


def has_proj_coords(xr_obj):
    x_coord, y_coord = _get_proj_dim_coords(xr_obj)
    if x_coord is not None and y_coord is not None: 
        return True
    else: 
        return False 


def _add_swath_coords_attrs(ds, crs) -> xr.Dataset:
    """
    Add CF-compliant CRS attributes to the coordinates of a swath.
    
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
    """
    Add CF-compliant attributes to the dimension coordinates of a projected CRS.
    
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
    """
    Add CF-compliant attributes to the CRS dimension coordinates.

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
        if has_swath_coords(ds):
            ds = _add_swath_coords_attrs(ds, crs)
        else: 
            ds = _add_proj_coords_attrs(ds, crs)
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
    ds = ds.assign_coords({grid_mapping_name: spatial_ref})
    return ds


def _grid_mapping_reference(ds, crs, grid_mapping_name):
    """Define the grid_mapping value to attach to the variables."""
    # Projected CRS
    if crs.is_projected: 
        x_dim, y_dim = _get_proj_dim_coords(ds)
        if x_dim is None or y_dim is None: 
            warnings.warn("Projection coordinates are not present.")
            output = f"{grid_mapping_name}"
        else: 
            output = f"{grid_mapping_name}: {x_dim} {y_dim}"
    # Geographic CRS
    else: 
        # If swath
        if has_swath_coords(ds):
            lon_coord, lat_coord = _get_swath_dim_coords(ds)
            output = f"{grid_mapping_name}: {lat_coord} {lon_coord}"
        else: 
            x_dim, y_dim = _get_proj_dim_coords(ds)
            if x_dim is None or y_dim is None: 
                warnings.warn("Projection coordinates are not present.")
                output = f"{grid_mapping_name}"
            output = f"{grid_mapping_name}: {x_dim} {y_dim}"
    return output 


def _add_variables_crs_attrs(ds, crs, grid_mapping_name):
    """Add 'grid_mapping' and 'coordinates' (for swath only) attributes to the variable.
    
    It assumes that all variables need the attributes !
    """
    # TODO: check which variable require attributes ! 
    
    # Add grid_mapping attributes 
    # - If one already existing, attach to it another one ! 
    grid_mapping_value = _grid_mapping_reference(ds, crs=crs, grid_mapping_name=grid_mapping_name)
    for var in ds.data_vars:  
        grid_mapping = ds[var].attrs.get("grid_mapping", "")
        if grid_mapping == "": 
            grid_mapping = grid_mapping_value
        else: 
            grid_mapping = grid_mapping + " " + grid_mapping_value
        ds[var].attrs["grid_mapping"] = grid_mapping
        
    # Add coordinates attribute if swath data
    if has_swath_coords(ds): 
        lon_coord, lat_coord = _get_swath_dim_coords(ds)
        for var in ds.data_vars:
            da_coords = ds[var].coords
            if lon_coord in da_coords and lat_coord in da_coords:
                ds[var].attrs["coordinates"] = f"{lat_coord} {lon_coord}"
    return ds 


def remove_existing_grid_mapping_attrs(ds):
    """Remove existing grid_mapping attributes."""
    for var in ds.data_vars: 
        _ = ds[var].attrs.pop("grid_mapping", None)
    return ds


def set_dataset_single_crs(ds, crs, grid_mapping_name="spatial_ref", inplace=False): 
    """Add CF-compliant CRS information to an xr.Dataset
    
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
    
    # Add CF attributes 'grid_mapping' (and 'coordinates' for swath) to the variables 
    ds = _add_variables_crs_attrs(ds=ds, crs=crs, grid_mapping_name=grid_mapping_name)
    
    return ds
    
    
def set_dataset_crs(ds, crs, grid_mapping_name="spatial_ref", inplace=False):
    """Add CF-compliant CRS information to an xr.Dataset
    
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
    ds = remove_existing_grid_mapping_attrs(ds)
    ds = set_dataset_single_crs(ds=ds, crs=crs, grid_mapping_name=grid_mapping_name, inplace=inplace)
    # If CRS is projected and 2D lon/lat are available, also add the WGS84 CRS
    if crs.is_projected and has_swath_coords(ds):
        crs_wgs84 = pyproj.CRS(proj="longlat", ellps="WGS84")
        ds = set_dataset_single_crs(ds=ds, crs=crs_wgs84, grid_mapping_name="crsWGS84", inplace=inplace)
    return ds 


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

# def _get_ds_area_extent 
# def _get_ds_resolution 
# def _get_ds_shape 