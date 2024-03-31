#!/usr/bin/env python3
# Utility to compute corners and cell vertices from centroid 2D coordinates.
# In future, the code in this file will be added to pyresample !
import dask.array
import numpy as np
import pyproj
from dask.array import map_blocks

### SwathDefinition --> geographic coordinates --> computation in ECEF (xyz)
### AreaDefinition --> proj coordinates --> computation in projection space !

# quadrilateral_corners / quadmesh
# get_lonlat_corners
# get_projection_corners


def _infer_interval_breaks(coord, axis=0):
    """Infer the outer and inner midpoints of 2D coordinate arrays.

    An [N,M] array

    The outer points are defined by taking half-distance of the closest centroid.
    The implementation is inspired from xarray.plot.utils._infer_interval_breaks

    Parameters
    ----------
    coord : np.array
        Coordinate array of shape [N,M].
    axis : int, optional
        Axis over which to infer the midpoint.
        axis = 0 infer midpoints along the vertical direction.
        axis = 1 infer midpoints along the horizontal direction.
        The default is 0.

    Returns
    -------
    breaks : np.array
        If axis = 0, it returns an array of shape [N+1, M]
        If axis = 0, it returns an array of shape [N, M+1]

    """
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
    return np.concatenate([first, offsets, last], axis=axis)


def _get_corners_from_centroids(centroids):
    """Get the coordinate corners 2D array from the centroids 2D array.

    The corners are guessed assuming equal spacing on either side of the coordinate.
    """
    # Identify breaks along columns
    breaks = _infer_interval_breaks(centroids, axis=1)
    # Identify breaks along rows
    return _infer_interval_breaks(breaks, axis=0)


def _do_transform(src_proj, dst_proj, lons, lats, alt):
    """Perform pyproj transformation and stack the results.

    If using pyproj >= 3.1, it employs thread-safe pyproj.transformer.Transformer.
    If using pyproj < 3.1, it employs pyproj.transform.
    Docs: https://pyproj4.github.io/pyproj/stable/advanced_examples.html#multithreading
    """
    if float(pyproj.__version__[0:3]) >= 3.1:
        from pyproj import Transformer

        transformer = Transformer.from_crs(src_proj.crs, dst_proj.crs)
        x, y, z = transformer.transform(lons, lats, alt, radians=False)
    else:
        x, y, z = pyproj.transform(src_proj, dst_proj, lons, lats, alt)
    return np.dstack((x, y, z))


def _geocentric_to_geographic(x, y, z, compute=True):
    """Map from geocentric cartesian to geographic coordinate system."""
    # Ensure dask array
    x = dask.array.asarray(x)
    y = dask.array.asarray(y)
    z = dask.array.asarray(z)
    # Define geocentric cartesian and geographic projection
    geocentric_proj = pyproj.Proj(proj="geocent")
    geographic_proj = pyproj.Proj(proj="latlong")

    # Conversion from geocentric cartesian to geographic coordinate system
    res = map_blocks(
        _do_transform,
        geocentric_proj,
        geographic_proj,
        x,
        y,
        z,
        new_axis=[2],
        chunks=(x.chunks[0], x.chunks[1], 3),
    )
    if compute:
        res = res.compute()
    lons = res[:, :, 0]
    lats = res[:, :, 1]
    return lons, lats


def _geographic_to_geocentric(lons, lats, compute=True):
    """Map from geographic to geocentric cartesian coordinate system."""
    # Ensure dask array
    lons = dask.array.asarray(lons)
    lats = dask.array.asarray(lats)
    # Define geocentric cartesian and geographic projection
    geocentric_proj = pyproj.Proj(proj="geocent")
    geographic_proj = pyproj.Proj(proj="latlong")

    # Conversion from geographic coordinate system to geocentric cartesian
    res = map_blocks(
        _do_transform,
        geographic_proj,
        geocentric_proj,
        lons,
        lats,
        dask.array.zeros_like(lons),  # altitude
        new_axis=[2],
        chunks=(lons.chunks[0], lons.chunks[1], 3),
    )
    if compute:
        res = res.compute()
    x = res[:, :, 0]
    y = res[:, :, 1]
    z = res[:, :, 2]
    return x, y, z


def _get_lonlat_corners(lons, lats):
    """Compute the corners of lon lat 2D pixel centroid coordinate arrays.

    It take care of cells crossing the antimeridian.
    It compute corners on geocentric cartesian coordinate system (ECEF).
    """
    # Map lons, lats to x, y, z centroids
    x, y, z = _geographic_to_geocentric(lons, lats)
    # Compute corners in geocentric cartesian coordinate system
    x_corners = _get_corners_from_centroids(x)
    y_corners = _get_corners_from_centroids(y)
    z_corners = _get_corners_from_centroids(z)
    # Convert back to lons, lats
    lons_corners, lat_corners = _geocentric_to_geographic(x_corners, y_corners, z_corners)
    return lons_corners, lat_corners


def _from_corners_to_bounds(corners, order="counterclockwise"):
    """Convert from corner 2D array (N+1, M+1)  to quadmesh vertices array (N,M, 4).

    Counterclockwise and clockwise bounds are defined from the top left corner.
    Counterclockwise: [top_left, bottom_left, bottom_right, top_right]
    Clockwise: [bottom_left, top_left, top_right, bottom_right]

    Inspired from https://github.com/xarray-contrib/cf-xarray/blob/main/cf_xarray/helpers.py#L113
    """
    top_left = corners[:-1, :-1]
    top_right = corners[:-1, 1:]
    bottom_right = corners[1:, 1:]
    bottom_left = corners[1:, :-1]
    if order == "clockwise":
        list_vertices = [bottom_left, top_left, top_right, bottom_right]
    else:  # counterclockwise
        list_vertices = [top_left, bottom_left, bottom_right, top_right]
    if hasattr(corners, "chunks"):
        bounds = dask.array.stack(list_vertices, axis=2)  # noqa
        # Rechunking over the vertices dimension is required !!!
        bounds = bounds.rechunk((bounds.chunks[0], bounds.chunks[1], 4))
    else:
        bounds = np.stack(list_vertices, axis=2)

    return bounds


def get_quadmesh_vertices(x, y, order="counterclockwise"):
    """Convert (x, y) 2D centroid coordinates array to (N*M, 4, 2) QuadMesh vertices.

    The output vertices can be passed directly to a matplotlib.PolyCollection.
    For plotting with cartopy, the polygon order must be "counterclockwise"

    Vertices are defined from the top left corner.
    """
    # - Retrieve QuadMesh corners (m+1 x n+1)
    x_corners, y_corners = _get_lonlat_corners(x, y)

    # - Retrieve QuadMesh bounds (m*n x 4)
    x_bounds = _from_corners_to_bounds(x_corners, order=order)
    y_bounds = _from_corners_to_bounds(y_corners, order=order)

    # - Retrieve QuadMesh vertices (m*n, 4, 2)
    return np.stack((x_bounds, y_bounds), axis=2)
