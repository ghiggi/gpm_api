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
"""This module contains tools for quadmesh/footprints computations."""
# -----------------------------------------------------------------.
# API
# get_quadmesh_centroids # gpm.quadmesh_centroids(crs=None)
# get_quadmesh_corners   # gpm.quadmesh_corners(crs=None)
# get_quadmesh_vertices  # gpm.quadmesh_vertices(crs=None, ccw=True)
# get_quadmesh_polygons  # gpm.quadmesh_polygons(crs=None)

# get_footprint_centroids # gpm.footprint_centroids(crs=None)
# get_footprint_vertices  # gpm.footprint_vertices(crs=None)
# get_footprint_polygons  # gpm.footprint_polygons(crs=None)

# Quadmesh vertices (N, M, 4, 2)
# - Output is always 4D array !
# - Order matters and must be checked with both coordinates
# - After CRS conversion it might be necessary to further check vertices order in some edge cases
#   like when the CRS y axis is flipped !

# -----------------------------------------------------------------.
# In future, the code in this file will be added also to pyresample !
# - SwathDefinition --> geographic coordinates --> computation in ECEF (xyz)
# - AreaDefinition --> proj coordinates --> computation in projection space !

# -----------------------------------------------------------------.
# Improvements
# - When inferring 2D coords might require estimate in 4 directions and averaging to
#   be more accurate for very curvilinear grids --> Check what cf_xarray does

# -------------------------------------------------------------------.
import dask.array as da
import numpy as np
import pyproj
from pyproj import Geod

from gpm.utils.decorators import check_is_gpm_object

# -----------------------------------------------------------------.
# TODO
# Check usage of map_overlap for quadmesh functions!
# Adapt wrapper to use crs
# Adapt to use xr_obj.gpm.x, xr_obj.gpm.y, xr_obj.gpm.z, xr_obj.gpm.crs

# ds_gr.gpm.is_orbit ?
# ds_gr.gpm.is_curvilinear
# ds_gr.gpm.crs.is_geographic

####------------------------------------------------------------------------------.
#### Checks


def _check_xy_inputs(x, y):
    """Check coordinates type, values and dimensionality."""
    # Check same array type
    if not isinstance(x, type(y)):
        raise TypeError(f"x is of {type(x)}, y is of {type(y)}")
    # Check valid inputs
    if x.size == 0 or y.size == 0:
        raise ValueError("Please provide x and y coordinates with values.")
    if x.size == 1 and y.size == 1:
        raise ValueError("Coordinates represents a single cell.")
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("Expecting 1D or 2D x and y arrays.")
    # Ensure 1D dimension array (at least)
    x = np.atleast_1d(x)  # TODO: this convert xarray objects to numpy !
    y = np.atleast_1d(y)
    return x, y


def _check_2d_coords(x, y):
    """Check 2D coordinates have same shape."""
    if x.shape != y.shape:
        raise ValueError(f"Arrays must have same shape: {x.shape} vs {y.shape}")
    return x, y


####------------------------------------------------------------------------------.
#### Utilities for vertices


def _remove_consecutive_duplicates(vertices):
    """
    Remove consecutive duplicate vertices from the array.

    Parameters
    ----------
    vertices : np.array
        An array of shape (N, 2) representing the vertices of the polygon.

    Returns
    -------
    np.array
        An array of vertices with consecutive duplicates removed.
    """
    diffs = np.diff(vertices, axis=0)
    unique_indices = np.any(diffs != 0, axis=1)
    unique_vertices = vertices[np.append([True], unique_indices)]
    return unique_vertices


def _find_convex_hull_vertex_index(vertices):
    """
    Find the index of convex hull vertex.

    It selects the index with the smallest X-coordinate.
    If there are several, pick the one with the smallest Y-coordinate.

    Parameters
    ----------
    vertices : np.array
        An array of shape (N, 2) representing the vertices of the polygon.

    Returns
    -------
    int
        Index of the reference vertex.
    """
    min_index = np.lexsort((vertices[:, 1], vertices[:, 0]))[0]
    return min_index


def is_clockwise(vertices):
    """
    Determine if polygon vertices are oriented clockwise or counterclockwise.

    This approach does not work for spherical polygons !

    It select a convex hull polygon vertex and determines the polygon orientation.

    Parameters
    ----------
    vertices : numpy.ndarray
        An array of shape (N, 2) representing the vertices of the polygon.

    Returns
    -------
    bool
        True if the vertices are oriented clockwise, False if counterclockwise.
    """
    # Check input vertices
    vertices = np.asanyarray(vertices)
    vertices = _remove_consecutive_duplicates(vertices)
    # Identify vertex of convex hull
    ref_idx = _find_convex_hull_vertex_index(vertices)

    # Retrieve vertices forming the two arcs
    prev_idx = (ref_idx - 1) % len(vertices)
    next_idx = (ref_idx + 1) % len(vertices)
    a = vertices[prev_idx]
    b = vertices[ref_idx]
    c = vertices[next_idx]

    # Compute determinant
    det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

    # If the determinant is negative, the vertices are oriented clockwise
    return det < 0


####------------------------------------------------------------------------------.
#### Utilities for 2D coordinates array
# - These methods require only 1 coordinate


def _infer_interval_breaks_numpy(coord, axis=0):
    """Infer the outer and inner midpoints of 2D coordinate arrays."""
    # Determine the half-distance between coordinates
    coord = np.asarray(coord)
    deltas = 0.5 * np.diff(coord, axis=axis)
    # Infer outer position of first and last
    first = np.take(coord, [0], axis=axis) - np.take(deltas, [0], axis=axis)
    last = np.take(coord, [-1], axis=axis) + np.take(deltas, [-1], axis=axis)
    # Infer position of internal points
    trim_last = tuple(slice(None, -1) if n == axis else slice(None) for n in range(coord.ndim))
    offsets = coord[trim_last] + deltas
    # Compute the breaks
    breaks = np.concatenate([first, offsets, last], axis=axis)
    return breaks


def _infer_interval_breaks_dask(coord, axis=0):
    """Infer the outer and inner midpoints of 2D coordinate arrays."""
    # Determine the half-distance between coordinates
    deltas = 0.5 * da.diff(coord, axis=axis)
    # Infer outer position of first and last
    first = da.take(coord, [0], axis=axis) - da.take(deltas, [0], axis=axis)
    last = da.take(coord, [-1], axis=axis) + da.take(deltas, [-1], axis=axis)
    # Infer position of internal points
    trim_last = tuple(slice(None, -1) if n == axis else slice(None) for n in range(coord.ndim))
    offsets = coord[trim_last] + deltas
    # Compute the breaks
    breaks = da.concatenate([first, offsets, last], axis=axis)
    return breaks


def infer_interval_breaks(coord, axis=0):
    """Infer the outer and inner midpoints of 2D coordinate arrays.

    An [N,M] array

    The outer points are defined by taking half-distance of the closest centroid.
    The implementation is inspired from :py:class:`xarray.plot.utils._infer_interval_breaks`.

    Parameters
    ----------
    coord : numpy.ndarray or dask.array.Array
        Coordinate array of shape [N,M].
    axis : int, optional
        Axis over which to infer the midpoint.
        axis = 0 infer midpoints along the vertical direction.
        axis = 1 infer midpoints along the horizontal direction.
        The default is 0.

    Returns
    -------
    breaks : numpy.ndarray or dask.array.Array
        If axis = 0, it returns an array of shape [N+1, M]
        If axis = 0, it returns an array of shape [N, M+1]
    """
    if hasattr(coord, "chunks"):
        breaks = _infer_interval_breaks_dask(coord, axis=axis)
    else:
        breaks = _infer_interval_breaks_numpy(coord, axis=axis)
    return breaks


def get_corners_from_centroids(centroids):
    """Get the coordinate corners 2D array from the centroids 2D array.

    The corners are guessed assuming equal spacing on either side of the coordinate.
    """
    # Identify breaks along columns
    breaks = infer_interval_breaks(centroids, axis=1)
    # Identify breaks along rows
    return infer_interval_breaks(breaks, axis=0)


def get_centroids_from_corners(corners):
    """Get the coordinate centroids 2D array from the corners 2D array.

    The centroids are assumed at the midpoint between the corners.
    """
    centroids = (corners[:-1, :-1] + corners[1:, 1:]) / 2
    return centroids


def _from_corners_to_bounds(corners, ccw=True, origin="bottom"):
    """Convert the cells corners coordinate 2D array (N+1,M+1) to the cell coordinate vertices 3D array (N,M,4).

    Counterclockwise and clockwise bounds are defined from the top left corner, assuming
    that the first axis (0) y is directed upward (origin="bottom") and the x axis (1) is directed rightward.

    Counterclockwise: [top_left, bottom_left, bottom_right, top_right]
    Clockwise: [bottom_left, top_left, top_right, bottom_right]

    ATTENTION: The actual ordering should be checked using the joint x and y coordinates after
    the call of this function if the assumption of the y-axis (0) directed upward and the x-axis (1) direct
    rightward does not hold.
    This is the often the case with satellite orbit swaths or when the source coordinates array have been transposed.

    Inspired from https://github.com/xarray-contrib/cf-xarray/blob/main/cf_xarray/helpers.py#L113
    """
    # FIXME: Likely not dask efficient
    # - Use map_overlap and take neighbour values to avoid rechunking?
    top_left = corners[:-1, :-1]
    top_right = corners[:-1, 1:]
    bottom_right = corners[1:, 1:]
    bottom_left = corners[1:, :-1]

    # Retrieve vertices in correct order
    if ccw:  # counterclockwise
        list_vertices = [top_left, bottom_left, bottom_right, top_right]
    else:
        list_vertices = [top_left, top_right, bottom_right, bottom_left]

    if origin != "bottom":
        list_vertices = list_vertices[::-1]
        list_vertices = [list_vertices[i] for i in [2, 3, 0, 1]]
    # Stack
    if hasattr(corners, "chunks"):
        bounds = da.stack(list_vertices, axis=2)
        # Rechunking over the vertices dimension is required !!!
        bounds = bounds.rechunk((bounds.chunks[0], bounds.chunks[1], 4))
    else:
        bounds = np.stack(list_vertices, axis=2)
    return bounds


def _from_bounds_to_corners(bounds, ccw=True):
    """Convert from bounds 3D array (N,M, 4) to corner 2D array (N+1,M+1).

    Assume x axis is rightward an y axis is upward.

    Inspired from https://github.com/xarray-contrib/cf-xarray/blob/main/cf_xarray/helpers.py#L86.
    """
    if not ccw:
        top_left_position = 0
        top_right_position = 1
        bottom_right_position = 2
        bottom_left_position = 3
    else:
        top_left_position = 0
        bottom_left_position = 1
        bottom_right_position = 2
        top_right_position = 3
    # Get corners points except last column and last row
    to_left = bounds[:, :, top_left_position]
    # Get corners last column
    to_right = bounds[:, -1:, top_right_position]
    # Get corners last row (except last column value)
    bot_left = bounds[-1:, :, bottom_left_position]
    # Get left bottom corner value
    bot_right = bounds[-1:, -1:, bottom_right_position]
    # Retrieve corners
    if hasattr(bounds, "chunks"):
        corners = da.block([[to_left, to_right], [bot_left, bot_right]])
    else:
        corners = np.block([[to_left, to_right], [bot_left, bot_right]])
    return corners


####------------------------------------------------------------------------------.
#### Utilities for 1D arrays


def preserve_dimension_order(func):
    """
    A decorator that preserves the original dimensions order of the input arrays in the output.

    The decorator works by squeezing the input arrays, applying the function,
    and then ensuring the original input dimension order.
    """

    def wrapper(x, y, *args, **kwargs):
        # Squeeze arrays to 1D if 2D arrays have singleton dimensions
        x_squeezed = np.atleast_1d(x.squeeze())
        y_squeezed = np.atleast_1d(y.squeeze())
        # Call the decorated function with squeezed inputs
        result = func(x_squeezed, y_squeezed, *args, **kwargs)
        # Reorder dimensions if necessary
        if x.shape != x_squeezed.shape or y.shape != y_squeezed.shape:
            squeezed_axis = np.where(np.array(x.shape) == 1)[0].item()
            if squeezed_axis == 0:
                return result[0].T, result[1].T
        return result

    return wrapper


####------------------------------------------------------------------------------.
#### Utilities for QuadMesh creation
# - These methods requires both x and y coordinates


def get_quadmesh_from_corners(x_corners, y_corners, ccw=True, origin="bottom"):
    # Retrieve QuadMesh bounds # (N, M, 4)
    x_bounds = _from_corners_to_bounds(x_corners, ccw=ccw, origin=origin)
    y_bounds = _from_corners_to_bounds(y_corners, ccw=ccw, origin=origin)
    # Retrieve QuadMesh vertices # (N, M, 4, 2)
    quadmesh = np.stack((x_bounds, y_bounds), axis=3)
    # Determine sample fraction of vertices with clockwise order
    # --> Select in the middle of y axis, a maximum of 5 points across x axis
    y_index = int(quadmesh.shape[0] / 2)
    x_indices = np.unique(np.linspace(0, quadmesh.shape[1] - 1, 5, dtype=int))
    sample_vertices = quadmesh[y_index, x_indices]
    # if hasattr(sample_vertices, "chunks"):
    #     sample_vertices = sample_vertices.compute()
    fraction_clockwise = np.sum([is_clockwise(vertices) for vertices in sample_vertices]) / len(x_indices)
    # Ensure correct order
    if (fraction_clockwise >= 0.5 and ccw) or (fraction_clockwise <= 0.5 and not ccw):
        quadmesh = quadmesh[:, :, ::-1, :]
    return quadmesh


def get_corners_from_quadmesh(quadmesh, ccw=True):
    # - Retrieve QuadMesh bounds # (N, M, 4)
    x_bounds = quadmesh[:, :, :, 0]
    y_bounds = quadmesh[:, :, :, 1]
    # - Retrieve QuadMesh corners # (N+1, M+1)
    x_corners = _from_bounds_to_corners(x_bounds, ccw=ccw)
    y_corners = _from_bounds_to_corners(y_bounds, ccw=ccw)
    # - Retrieve QuadMesh vertices # (N, M, 4, 2)
    return x_corners, y_corners


####------------------------------------------------------------------------------.
#### Geographic Quadmesh of 2D spatial coordinates
# - 2D geographic coordinates (on the sphere)
# - Satellite swath
# - Antimeridian/pole issues
# - Infer corners in X,Y,Z ECEF CRS and backproject


def _predict_forward_point(lon1, lat1, lon2, lat2, geod):
    """
    Predict a point in the forward direction at the same distance as between two given vertices.

    Parameters
    ----------
    lon1, lat1 : Longitude and latitude of the starting vertex.
    lon2, lat2 : Longitude and latitude of the ending vertex.
    geod       : pyproj.Geod object for geodesic calculations.

    Returns
    -------
    (lon3, lat3): Longitude and latitude of the predicted point in the forward direction.
    """
    # Calculate the forward azimuth and distance between the start and end vertices
    fwd_azimuth, _, distance = geod.inv(lon1, lat1, lon2, lat2)

    # Predict the point in the forward direction from the end vertex
    lon3, lat3, _ = geod.fwd(lon2, lat2, fwd_azimuth, distance)

    return lon3, lat3


@preserve_dimension_order
def get_lonlat_corners_from_1d_centroids(lon, lat):
    """Compute lon/lat corners from 1D coordinate array.

    This function is used to define a quadmesh for nadir-looking swath.

    It infer spacing from the available dimension and assume equal spacing on the other dimension.
    """
    # Ensure same size (for GRID slice case where one is scalar and other is array)
    lon = np.atleast_1d(lon)
    lat = np.atleast_1d(lat)
    if len(lon) != len(lat):
        if len(lat) == 1:
            lat = np.ones(lon.shape) * lat
        elif len(lon) == 1:
            lon = np.ones(lat.shape) * lon
        else:
            raise ValueError(f"Shape mismatch between lon and lat arrays: {len(lon)} vs. {len(lat)} ")

    geod = Geod(ellps="WGS84")

    # Define corners between centroids
    ortho_line1_lon = []
    ortho_line1_lat = []
    ortho_line2_lon = []
    ortho_line2_lat = []

    # Process each segment of the original line
    for i in range(len(lon) - 1):
        lon1, lat1 = lon[i], lat[i]
        lon2, lat2 = lon[i + 1], lat[i + 1]

        # Calculate the forward azimuth and distance between consecutive vertices
        fwd_azimuth, _, distance = geod.inv(lon1, lat1, lon2, lat2, radians=False)

        # Calculate the half distance for the orthogonal projection
        half_distance = distance / 2

        # Calculate the orthogonal points from the midpoint of the segment
        mid_lon, mid_lat, _ = geod.fwd(lon1, lat1, fwd_azimuth, half_distance)

        # Orthogonal azimuths, +90 degrees and -90 degrees
        ortho_azimuth1 = (fwd_azimuth + 90) % 360
        ortho_azimuth2 = (fwd_azimuth - 90) % 360

        # Project the orthogonal points
        lon_ortho1, lat_ortho1, _ = geod.fwd(mid_lon, mid_lat, ortho_azimuth1, half_distance)
        lon_ortho2, lat_ortho2, _ = geod.fwd(mid_lon, mid_lat, ortho_azimuth2, half_distance)

        # Append the calculated orthogonal points to the lines
        ortho_line1_lon.append(lon_ortho1)
        ortho_line1_lat.append(lat_ortho1)
        ortho_line2_lon.append(lon_ortho2)
        ortho_line2_lat.append(lat_ortho2)

    # Extend at the extremities to define the corners
    # - Extend the start of ortho_line1
    start_ext_lon1, start_ext_lat1 = _predict_forward_point(
        ortho_line1_lon[1],
        ortho_line1_lat[1],
        ortho_line1_lon[0],
        ortho_line1_lat[0],
        geod=geod,
    )
    # - Extend the end of ortho_line1
    end_ext_lon1, end_ext_lat1 = _predict_forward_point(
        ortho_line1_lon[-2],
        ortho_line1_lat[-2],
        ortho_line1_lon[-1],
        ortho_line1_lat[-1],
        geod=geod,
    )

    # - Extend the start of ortho_line2
    start_ext_lon2, start_ext_lat2 = _predict_forward_point(
        ortho_line2_lon[1],
        ortho_line2_lat[1],
        ortho_line2_lon[0],
        ortho_line2_lat[0],
        geod=geod,
    )
    # - Extend the end of ortho_line2
    end_ext_lon2, end_ext_lat2 = _predict_forward_point(
        ortho_line2_lon[-2],
        ortho_line2_lat[-2],
        ortho_line2_lon[-1],
        ortho_line2_lat[-1],
        geod=geod,
    )

    # Define corners
    lon_corners1 = np.array([start_ext_lon1, *ortho_line1_lon, end_ext_lon1])
    lat_corners1 = np.array([start_ext_lat1, *ortho_line1_lat, end_ext_lat1])

    lat_corners2 = np.array([start_ext_lat2, *ortho_line2_lat, end_ext_lat2])
    lon_corners2 = np.array([start_ext_lon2, *ortho_line2_lon, end_ext_lon2])

    lon_corners = np.vstack((lon_corners1, lon_corners2)).T
    lat_corners = np.vstack((lat_corners1, lat_corners2)).T

    # Define lon lat corners
    return lon_corners, lat_corners


def geocentric_to_geographic(x, y, z, parallel=True):
    from gpm.utils.remapping import reproject_coords

    geocentric_proj = pyproj.Proj(proj="geocent")
    geographic_proj = pyproj.Proj(proj="latlong")
    return reproject_coords(x, y, z, parallel=parallel, src_crs=geocentric_proj.crs, dst_crs=geographic_proj.crs)


def geographic_to_geocentric(lons, lats, z=None, parallel=True):
    from gpm.utils.remapping import reproject_coords

    geocentric_proj = pyproj.Proj(proj="geocent")
    geographic_proj = pyproj.Proj(proj="latlong")
    return reproject_coords(lons, lats, z, parallel=parallel, src_crs=geographic_proj.crs, dst_crs=geocentric_proj.crs)


def get_lonlat_corners_from_centroids(lons, lats, parallel=True):
    """Compute the corners of lon lat 2D pixel centroid coordinate arrays.

    It takes care of cells crossing the antimeridian.
    It compute corners on geocentric cartesian coordinate system (ECEF).
    """
    lons, lats = _check_xy_inputs(lons, lats)
    if lons.squeeze().ndim <= 1 and lats.squeeze().ndim <= 1:
        return get_lonlat_corners_from_1d_centroids(lons, lats)
    if lons.ndim == 2 and lats.ndim == 2:
        lons, lats = _check_2d_coords(lons, lats)
        # Map lons, lats to x, y, z centroids
        x, y, z = geographic_to_geocentric(lons, lats, parallel=parallel)
        x_corners = get_corners_from_centroids(x)
        y_corners = get_corners_from_centroids(y)
        z_corners = get_corners_from_centroids(z)
        # Convert back to lons, lats, height
        lons_corners, lat_corners, _ = geocentric_to_geographic(x_corners, y_corners, z_corners, parallel=parallel)
        return lons_corners, lat_corners
    raise NotImplementedError(f"lons is a {lons.ndim}D array, lats is a {lats.ndim}D array.")


def get_lonlat_quadmesh_vertices(lons, lats, ccw=True, origin="bottom"):  # from partitioning code
    """Convert (x, y) 2D centroid coordinates array to (N, M, 4, 2) QuadMesh vertices.

    The output vertices can be passed directly to a :py:class:`matplotlib.collections.PolyCollection`.
    For plotting with cartopy, the polygon order must be counterclockwise.

    Vertices are defined from the top left corner.
    """
    # - Retrieve QuadMesh corners  # (N+1, M+1)
    x_corners, y_corners = get_lonlat_corners_from_centroids(lons, lats)
    # - Retrieve QuadMesh vertices # (N, M, 4, 2)
    return get_quadmesh_from_corners(x_corners, y_corners, ccw=ccw, origin=origin)


####------------------------------------------------------------------------------.
#### Planar Quadmesh of 2D spatial coordinates
# - 1D coordinates (i.e projection x and y)
# - Grid, Projection Area
# - No antimeridian/pole issue
# - For regular projection corners can be defined from extent + resolution of projection
# --> odc.GeoBox, pyresample.AreaDefinition


def get_projection_centroids(x, y, origin="bottom"):
    """Return 2D centroids arrays."""
    # FIXME: 1D returns always numpy, 2D return as it is currently
    if x.ndim == 1 and y.ndim == 1:
        x, y = np.meshgrid(x, y)
        if origin == "bottom":
            y = y[::-1, :]
        return x, y
    if x.ndim == 2 and y.ndim == 2:
        x, y = _check_2d_coords(x, y)
        return x, y
    raise NotImplementedError(f"x is a {x.ndim}D array, y is a {y.ndim}D array.")


@preserve_dimension_order
def get_projection_corners_from_1d_centroids(x, y, origin="bottom"):
    # Check size of arrays
    is_x_scalar = x.size < 2
    is_y_scalar = y.size < 2
    # Check at least x or y is an array with 2 coordinates
    if is_x_scalar and is_y_scalar:
        raise ValueError("Coordinates represents a single cell. Impossible to infer cell resolution and thus corners.")
    # Infer breaks from coordinate array (if possible)
    if not is_x_scalar:
        x_breaks = infer_interval_breaks(x)
    if not is_y_scalar:
        y_breaks = infer_interval_breaks(y)
    # Otherwise, infer from the other dimension
    if is_x_scalar:
        avg_y_res = np.average(np.diff(y_breaks))
        x_breaks = np.concatenate([x - avg_y_res / 2, x + avg_y_res / 2])
    if is_y_scalar:
        avg_x_res = np.average(np.diff(x_breaks))
        y_breaks = np.concatenate([y - avg_x_res / 2, y + avg_x_res / 2])
    # Default it assumes that y axis is defined upward
    x_corners, y_corners = np.meshgrid(x_breaks, y_breaks)
    if origin == "bottom":
        y_corners = y_corners[::-1, :]
    return x_corners, y_corners


def get_projection_corners_from_centroids(x, y, origin="bottom"):
    """Compute the corners of x and y 2D pixel centroid coordinate arrays."""
    x, y = _check_xy_inputs(x, y)
    # Retrieve 2D corners from 1D centroids
    if x.squeeze().ndim <= 1 and y.squeeze().ndim <= 1:
        return get_projection_corners_from_1d_centroids(x, y, origin=origin)
    # Retrieve 2D corners from 2D centroids
    # - FIXME: case where size 1 one dimension
    if x.ndim == 2 and y.ndim == 2:
        x, y = _check_2d_coords(x, y)
        x_corners = get_corners_from_centroids(x)
        y_corners = get_corners_from_centroids(y)
        return x_corners, y_corners
    raise NotImplementedError(f"x is a {x.ndim}D array, y is a {y.ndim}D array.")


def get_projection_quadmesh_vertices(x, y, ccw=True):  # from partitioning code
    """Convert (x, y) 2D centroid coordinates array to (N, M, 4, 2) QuadMesh vertices.

    The output vertices can be passed directly to a :py:class:`matplotlib.collections.PolyCollection`.
    For plotting with cartopy, the polygon order must be counterclockwise.

    Vertices are defined from the top left corner.
    """
    # - Retrieve QuadMesh corners  # (N+1, M+1)
    x_corners, y_corners = get_projection_corners_from_centroids(x, y)
    # - Retrieve QuadMesh vertices # (N, M, 4, 2)
    return get_quadmesh_from_corners(x_corners, y_corners, ccw=ccw)


####------------------------------------------------------------------------------.
#### Quadmesh Wrappers
# <accessor>: pyresample.area, gpm
# ds.<accessor>.quadmesh_centroids
# ds.<accessor>.quadmesh_corners
# ds.<accessor>.quadmesh_vertices
# ds.<accessor>.quadmesh_polygons
# --> In future these methods could be adapted to return xr.DataArrays


def _try_get_crs(xr_obj):
    try:
        pyproj_crs = xr_obj.gpm.pyproj_crs
    except Exception:
        pyproj_crs = None
    return pyproj_crs


@check_is_gpm_object
def get_quadmesh_centroids(xr_obj, crs=None, origin="bottom"):
    """Return quadmesh x and y centroids of shaope (N,M)."""
    from gpm.utils.remapping import reproject_coords

    # TODO: IMPLEMENT
    # check_valid_crs
    # get_coords/set_index à la xoak/metpy:
    # --> xr_obj.gpm.x, xr_obj.gpm.y, xr_obj.gpm.z, xr_obj.gpm.crs
    # add test for crs-conversion
    x = "lon"
    y = "lat"
    x = xr_obj[x].data
    y = xr_obj[y].data
    src_crs = _try_get_crs(xr_obj)
    if xr_obj.gpm.is_grid:
        x, y = get_projection_centroids(x, y, origin=origin)
    if crs is not None and src_crs is not None:
        x, y, _ = reproject_coords(x, y, src_crs=src_crs, dst_crs=crs)
    return x, y


@check_is_gpm_object
def get_quadmesh_corners(xr_obj, crs=None):
    """Return quadmesh x and y corners of shape (N+1, M+1)."""
    from gpm.utils.remapping import reproject_coords

    # TODO: IMPLEMENT
    # check_valid_crs
    # infert_x_y_coords or xr_obj.gpm.x, xr_obj.gpm.y, xr_obj.gpm.z, xr_obj.gpm.crs
    # add test for crs-conversion
    x = "lon"
    y = "lat"
    src_crs = _try_get_crs(xr_obj)
    x = xr_obj[x].data
    y = xr_obj[y].data
    if crs is not None and src_crs is not None:
        x, y, _ = reproject_coords(x, y, src_crs=src_crs, dst_crs=crs)
    if xr_obj.gpm.is_orbit:  # curvilinear geographic CRS
        return get_lonlat_corners_from_centroids(lons=x, lats=y)
    # xr_obj.gpm.is_grid
    return get_projection_corners_from_centroids(x=x, y=y)


@check_is_gpm_object
def get_quadmesh_vertices(xr_obj, crs=None, ccw=True, origin="bottom"):
    """Return the quadmesh vertices array with shape (N, M, 4, 2).

    Parameters
    ----------
    origin : str
        Origin of the y axis.
        The default is ``bottom``.
    ccw : bool, optional
        If ``True``, vertices are ordered counterclockwise.
        If ``False``, vertices are ordered clockwise.
        The default is ``True``.
    """
    x_corners, y_corners = get_quadmesh_corners(xr_obj, crs=crs)
    vertices = get_quadmesh_from_corners(x_corners, y_corners, ccw=ccw, origin=origin)
    return vertices


@check_is_gpm_object
def get_quadmesh_polygons(xr_obj, crs=None):
    """Return an array with quadmesh shapely polygons."""
    import shapely

    vertices = get_quadmesh_vertices(xr_obj, crs=crs, ccw=True)
    return shapely.polygons(vertices)
