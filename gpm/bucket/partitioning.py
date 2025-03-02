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
"""This module implements Spatial Partitioning classes."""
import os
from functools import reduce, wraps

import numpy as np
import pandas as pd

from gpm.bucket.dataframe import (
    check_valid_dataframe,
    df_add_column,
    df_get_column,
    df_is_column_in,
    df_select_valid_rows,
    df_to_pandas,
)
from gpm.utils.geospatial import (
    Extent,
    _check_size,
    check_extent,
    get_continent_extent,
    get_country_extent,
    get_extent_around_point,
    get_geographic_extent_around_point,
)

pd.options.mode.copy_on_write = True

# Future methods:
# to_spherically (geographic)
# to_geopandas [lat_bin, lon_bin, geometry]


def _apply_flatten_arrays(self, func, x, y, **kwargs):
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.ndim == 2 and y.ndim == 2:
        original_shape = x.shape
        x_flat = x.flatten()
        y_flat = y.flatten()
        result = func(self, x_flat, y_flat, **kwargs)
        if isinstance(result, tuple):
            result = tuple(r.reshape(original_shape) for r in result)
        else:  # np.array
            result = result.reshape(original_shape + result.shape[1:])
        return result
    return func(self, x, y, **kwargs)


def flatten_xy_arrays(func):
    @wraps(func)
    def wrapper(self, x, y, **kwargs):
        return _apply_flatten_arrays(self, func=func, x=x, y=y, **kwargs)

    return wrapper


def flatten_indices_arrays(func):
    @wraps(func)
    def wrapper(self, x_indices, y_indices, **kwargs):
        return _apply_flatten_arrays(self, func=func, x=x_indices, y=y_indices, **kwargs)

    return wrapper


def np_broadcast_like(x, shape):
    arr = np.zeros(shape, x.dtype)
    arr[:] = np.expand_dims(x, axis=tuple(range(1, len(shape))))
    return arr


def mask_invalid_indices(flag_value=np.nan):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Extract arguments
            x_indices = kwargs.get("x_indices", args[0] if len(args) > 0 else None)
            y_indices = kwargs.get("y_indices", args[1] if len(args) > 1 else None)
            # Ensure is a 1D numpy array
            x_indices = np.atleast_1d(np.asanyarray(x_indices))
            y_indices = np.atleast_1d(np.asanyarray(y_indices))
            # Determine invalid indices
            invalid_indices = ~np.isfinite(x_indices) | ~np.isfinite(y_indices)
            # Set dummy value for invalid indices
            x_indices[invalid_indices] = 0  # dummy index
            y_indices[invalid_indices] = 0  # dummy index
            # Ensure indices are integers !
            x_indices = x_indices.astype(int)
            y_indices = y_indices.astype(int)
            # Call the original function
            result = func(self, x_indices, y_indices, **kwargs)
            # Apply the mask to the result
            if isinstance(result, tuple):
                masked_result = tuple(np.where(invalid_indices, flag_value, r) for r in result)
            else:  # np.array
                invalid_indices = np_broadcast_like(invalid_indices, result.shape)
                masked_result = np.where(invalid_indices, flag_value, result)
            return masked_result

        return wrapper

    return decorator


def _check_labels_decimals(decimals):
    """Check and normalize the size input.

    This function accepts the number of labels decimals defined as an integer, float, tuple, or list.
    It normalizes the input into a tuple of two elements, each representing the
    desired number of decimals for the x and y partition labels.

    Returns
    -------
    list
        A list of two elements (x_decimals, y_decimals)
    """
    if isinstance(decimals, (int, np.integer)):
        decimals = list([decimals] * 2)
    elif isinstance(decimals, (tuple, list)):
        if len(decimals) != 2:
            raise ValueError("Expecting a decimals (x, y) tuple.")
    else:
        raise TypeError("Accepted decimals type are int, list or tuple.")
    if np.any(np.array(decimals) < 0):
        raise ValueError("Expecting positive 'labels_decimals' values.")
    return list(decimals)


def check_default_levels(levels, default_levels):
    if levels is None:
        levels = default_levels
    if isinstance(levels, str):
        levels = [levels]
    if not isinstance(levels, list):
        raise TypeError("'levels' must be a list specifying the partition names.")
    return levels


def check_partitioning_order(levels, order):
    if set(levels) != set(order):
        raise ValueError(f"Partitions 'order' ({order}) does not match with partition names {levels}.")
    return order


def check_partitioning_flavor(flavor):
    """Validate the flavor argument.

    If ``None``, defaults to "directory".
    """
    if flavor is None:
        flavor = "directory"
    valid_flavors = ["directory", "hive"]
    if flavor not in valid_flavors:
        raise ValueError(f"Invalid partitioning 'flavor '{flavor}'. Valid options are {valid_flavors}.")
    return flavor


def check_valid_x_y(df, x, y):
    """Check if the x and y columns are in the dataframe."""
    if not df_is_column_in(df, column=y):
        raise ValueError(f"y='{y}' is not a column of the dataframe.")
    if not df_is_column_in(df, column=x):
        raise ValueError(f"x='{x}' is not a column of the dataframe.")


def get_array_combinations(x, y):
    """Return all combinations between the two input arrays."""
    # Create the mesh grid
    grid1, grid2 = np.meshgrid(x, y)
    # Stack and reshape the grid arrays to get combinations
    combinations = np.vstack([grid1.ravel(), grid2.ravel()]).T
    return combinations[:, 0], combinations[:, 1]


def get_centroids_from_bounds(bounds):
    """Define partitions centroids from bounds."""
    centroids = (bounds[:-1] + bounds[1:]) / 2
    return centroids


def query_indices(values, bounds):
    """Return the index for the specified coordinates.

    Invalid values (NaN, None) or out of bounds values returns NaN.
    """
    values = np.atleast_1d(np.asanyarray(values)).astype(float)
    return pd.cut(values, bins=bounds, labels=False, include_lowest=True, right=True)


def get_partition_dir_name(partition_name, partition_labels, flavor):
    """Return the directories name of a partition."""
    if flavor == "hive":
        return reduce(np.char.add, [partition_name, "=", partition_labels, os.sep])
    return np.char.add(partition_labels, os.sep)


def get_directories(dict_labels, order, flavor):
    """Return the directory trees of a partitioned dataset."""
    list_dir_names = []
    for partition in order:
        dir_name = get_partition_dir_name(
            partition_name=partition,
            partition_labels=dict_labels[partition],
            flavor=flavor,
        )
        list_dir_names.append(dir_name)
    dir_trees = reduce(np.char.add, list_dir_names)
    dir_trees = np.char.rstrip(dir_trees, os.sep)
    return dir_trees


####-------------------------------------------------------------------------------------------------------------------.
##################################
#### XYPartitioning Utilities ####
##################################
def get_n_decimals(number):
    """Get the number of decimals of a number."""
    number_str = str(number)
    decimal_index = number_str.find(".")

    if decimal_index == -1:
        return 0  # No decimal point found

    # Count the number of characters after the decimal point
    return len(number_str) - decimal_index - 1


def get_bounds(size, vmin, vmax):
    """Define partitions edges."""
    bounds = np.arange(vmin, vmax, size)
    if bounds[-1] != vmax:
        bounds = np.append(bounds, np.array([vmax]))
    return bounds


####-----------------------------------------------------------------------------------------------------------------.
#### Tiles Utilities


def justify_labels(labels, length):
    return np.char.rjust(labels, length, "0")


def get_tile_xy_labels(x_indices, y_indices, origin, n_x, n_y, justify=False):
    """Return the 2D tile labels for the specified x,y indices."""
    x_labels = x_indices.astype(str)
    y_labels = y_indices.astype(str) if origin == "top" else (n_y - 1 - y_indices).astype(str)
    # Optional justify the labels
    if justify:
        x_labels = justify_labels(x_labels, length=len(str(n_x)))
        y_labels = justify_labels(y_labels, length=len(str(n_y)))
    return x_labels, y_labels


def get_tile_id_labels(x_indices, y_indices, origin, direction, n_x, n_y, justify):
    """Return the 1D tile labels for the specified x,y indices."""
    if direction == "x":
        if origin == "top":
            flattened_indices = np.ravel_multi_index((y_indices, x_indices), (n_y, n_x), order="C")
        else:  # origin == "bottom"
            y_indices_flipped = n_y - 1 - y_indices
            flattened_indices = np.ravel_multi_index((y_indices_flipped, x_indices), (n_y, n_x), order="C")
    elif origin == "top":
        flattened_indices = np.ravel_multi_index((y_indices, x_indices), (n_y, n_x), order="F")
    else:  # origin == "bottom"
        y_indices_flipped = n_y - 1 - y_indices
        flattened_indices = np.ravel_multi_index((y_indices_flipped, x_indices), (n_y, n_x), order="F")
    # Conversion to string
    labels = flattened_indices.astype(str)
    # Optional justify the labels
    if justify:
        labels = justify_labels(labels, length=len(str(n_x * n_y)))
    return labels


####-----------------------------------------------------------------------------------------------------------------.
#### Xarray reformatting utility
def _ensure_indices_list(indices):
    if indices is None:
        indices = []
    indices = [indices] if isinstance(indices, str) else list(indices)
    if indices == [None]:  # what is returned by df.index.names if no index !
        indices = []
    return indices


####------------------------------------------------------------------------------------------------------------------.
#### 2D Partitioning Classes


class Base2DPartitioning:
    """
    Handles partitioning of 2D data into rectangular tiles.

    The size of the partitions can varies between and across the x and y directions.

    Parameters
    ----------
    levels : str or list
        Name or names of the partitions.
        If partitioning by 1 level (i.e. by a unique partition id), specify a single partition name.
        If partitioning by 2 or more levels (i.e. by x and y), specify the x, y (z, ...) partition levels names.
    x_bounds : numpy.ndarray
        The partition bounds across the x (horizontal) dimension.
    y_bounds : numpy.ndarray
        The partition bounds across the y (vertical) dimension.
        Please provide the bounds with increasing values order.
        The origin of the partition class indices is the top, left corner.
    order : list
        The order of the partitions when writing multi-level partitions (i.e. x, y) to disk.
        The default, ``None``, corresponds to ``names``.
    flavor : str
        This argument governs the directories names of partitioned datasets.
        The default, ``None``, name the directories with the partitions labels (DirectoryPartitioning).
        The option ``"hive"``, name the directories with the format ``{partition_name}={partition_label}``.
    """

    def __init__(self, x_bounds, y_bounds, levels, flavor=None, order=None):

        self.x_bounds = np.asanyarray(x_bounds)
        self.y_bounds = np.asanyarray(y_bounds)
        self.x_centroids = get_centroids_from_bounds(self.x_bounds)
        self.y_centroids = get_centroids_from_bounds(self.y_bounds)
        # Define partitions names, order and flavour
        self.levels = check_default_levels(levels=levels, default_levels=None)
        if order is None:
            self.order = self.levels
        else:
            self.order = check_partitioning_order(
                levels=self.levels,
                order=order,
            )
        self.flavor = check_partitioning_flavor(flavor)

        # Define info
        self.shape = (len(self.y_centroids), len(self.x_centroids))
        self.n_partitions = self.shape[0] * self.shape[1]
        self.n_levels = len(self.levels)
        self.n_x = self.shape[1]
        self.n_y = self.shape[0]

        # Define private attrs
        self._labels = None
        self._centroids = None
        self._x_coord = "x_c"  # default name for x centroid column for add_centroids
        self._y_coord = "y_c"  # default name for y centroid column for add_centroids

    @flatten_xy_arrays
    def query_indices(self, x, y):
        """Return the 2D partition indices for the specified x,y coordinates."""
        x_indices = query_indices(x, bounds=self.x_bounds)
        y_indices = query_indices(y, bounds=self.y_bounds)
        return x_indices, y_indices

    @flatten_indices_arrays
    @mask_invalid_indices(flag_value="nan")
    def query_labels_by_indices(self, x_indices, y_indices):
        """Return the partition labels as function of the specified 2D partitions indices."""
        return self._custom_labels_function(x_indices=x_indices, y_indices=y_indices)

    def _custom_labels_function(self, x_indices, y_indices):  # noqa
        """Return the partition labels for the specified x,y indices."""
        class_name = self.__class__.name
        raise NotImplementedError(f"'_custom_labels_function' has yet be implemented for subclass {class_name}!")

    @flatten_xy_arrays
    def query_labels(self, x, y):
        """Return the partition labels for the specified x,y coordinates."""
        x_indices, y_indices = self.query_indices(x=x, y=y)
        return self.query_labels_by_indices(x_indices, y_indices)

    @flatten_indices_arrays
    @mask_invalid_indices(flag_value=np.nan)
    def query_centroids_by_indices(self, x_indices, y_indices):
        """Return the partition centroids for the specified x,y indices."""
        x_centroids = self.x_centroids[x_indices]
        y_centroids = self.y_centroids[y_indices]
        return x_centroids, y_centroids

    @flatten_xy_arrays
    def query_centroids(self, x, y):
        """Return the partition centroids for the specified x,y coordinates."""
        x_indices, y_indices = self.query_indices(x=x, y=y)
        return self.query_centroids_by_indices(x_indices, y_indices)

    @property
    def labels(self):
        """Return the labels array of shape (n_y, n_x, n_levels)."""
        if self._labels is None:
            # Retrieve labels combination of all (x,y) indices
            x_indices, y_indices = np.meshgrid(np.arange(self.n_x), np.arange(self.n_y))
            # Retrieve labels
            # - If n_levels >= 2 --> query_labels_by_indices return a tuple !
            labels = self.query_labels_by_indices(x_indices=x_indices, y_indices=y_indices)
            if self.n_levels >= 2:
                labels = np.stack(labels, axis=-1)
            self._labels = labels
        return self._labels

    @property
    def centroids(self):
        """Return the centroids array of shape (n_y, n_x, 2)."""
        if self._centroids is None:
            # Retrieve centroids of all (x,y) indices
            x_indices, y_indices = np.meshgrid(np.arange(self.n_x), np.arange(self.n_y))
            centroids = self.query_centroids_by_indices(x_indices, y_indices)
            centroids = np.stack(centroids, axis=-1)
            self._centroids = centroids
        return self._centroids

    @property
    def bounds(self):
        """Return the partitions bounds."""
        return self.x_bounds, self.y_bounds

    def quadmesh_corners(self, origin="bottom"):
        """Return the quadrilateral mesh corners.

        A quadrilateral mesh is a grid of M by N adjacent quadrilaterals that are defined via a (M+1, N+1)
        grid of vertices.

        The quadrilateral mesh is accepted by :py:class:`matplotlib.pyplot.pcolormesh`,
        :py:class:`matplotlib.collections.QuadMesh` and :py:class:`matplotlib.collections.PolyQuadMesh`.

        Parameters
        ----------
        origin: str
            Origin of the y axis.
            The default is ``bottom``.

        Return
        --------
        (x_corners, y_corners): tuple
            Numpy array of shape (M+1, N+1)
        """
        x_corners, y_corners = np.meshgrid(self.x_bounds, self.y_bounds)
        if origin == "bottom":
            y_corners = y_corners[::-1, :]
        return x_corners, y_corners

    def vertices(self, ccw=True, origin="bottom"):
        """Return the partitions vertices in an array of shape (N, M, 4, 2).

        The output vertices, once the first 2 dimensions are flattened,
        can be passed directly to a :py:class:`matplotlib.collections.PolyCollection`.
        For plotting with cartopy, the polygon order must be counterclockwise ordered.

        Parameters
        ----------
        ccw : bool, optional
            If ``True``, vertices are ordered counterclockwise.
            If ``False``, vertices are ordered clockwise.
            The default is ``True``.
        origin : str
            Origin of the y axis.
            The default is ``bottom``.
        """
        from gpm.utils.area import get_quadmesh_from_corners

        x_corners, y_corners = self.quadmesh_corners(origin=origin)
        vertices = get_quadmesh_from_corners(x_corners, y_corners, ccw=ccw, origin=origin)
        return vertices

    def to_shapely(self):
        """Return an array with shapely polygons."""
        import shapely

        return shapely.polygons(self.vertices(ccw=True))

    @flatten_indices_arrays
    @mask_invalid_indices(flag_value=np.nan)
    def query_vertices_by_indices(self, x_indices, y_indices, ccw=True):
        """Return the partitions vertices in an array of shape (indices, 4, 2)."""
        x_indices = np.atleast_1d(np.asanyarray(x_indices))
        y_indices = np.atleast_1d(np.asanyarray(y_indices))
        x_bnds = (self.x_bounds[x_indices], self.x_bounds[x_indices + 1])
        y_bnds = (self.y_bounds[y_indices], self.y_bounds[y_indices + 1])
        top_left = np.stack((x_bnds[0], y_bnds[1]), axis=1)
        top_right = np.stack((x_bnds[1], y_bnds[1]), axis=1)
        bottom_right = np.stack((x_bnds[1], y_bnds[0]), axis=1)
        bottom_left = np.stack((x_bnds[0], y_bnds[0]), axis=1)
        if ccw:
            list_vertices = [top_left, bottom_left, bottom_right, top_right]
        else:
            list_vertices = [top_left, top_right, bottom_right, bottom_left]
        vertices = np.stack(list_vertices, axis=1)
        return vertices

    @flatten_xy_arrays
    def query_vertices(self, x, y, ccw=True):
        x_indices, y_indices = self.query_indices(x, y)
        return self.query_vertices_by_indices(x_indices, y_indices, ccw=ccw)

    def _get_dict_labels_combo(self, x_indices, y_indices):
        # Retrieve labels combination of all (x,y) indices
        indices = get_array_combinations(x_indices, y_indices)
        # Retrieve corresponding labels
        # If n_levels >= 2 --> self.labels is a tuple
        # If n_levels == 1 --> self.labels is a 1D array
        labels = self.query_labels_by_indices(x_indices=indices[0], y_indices=indices[1])
        dict_labels = {}
        if self.n_levels > 1:
            dict_labels = {self.levels[i]: labels[i] for i in range(0, self.n_levels)}
        else:  # (tile_id)
            dict_labels = {self.levels[0]: labels}
        return dict_labels

    def _directories(self, dict_labels):
        return get_directories(
            dict_labels=dict_labels,
            order=self.order,
            flavor=self.flavor,
        )

    @property
    def directories(self):
        """Return the directory trees."""
        dict_labels = self._get_dict_labels_combo(x_indices=np.arange(0, self.n_x), y_indices=np.arange(0, self.n_y))
        return self._directories(dict_labels=dict_labels)

    def get_partitions_by_extent(self, extent):
        """Return the partitions labels containing data within the extent."""
        extent = check_extent(extent)
        # Define valid query extent (to be aligned with partitioning extent)
        query_extent = [
            max(extent.xmin, self.extent.xmin),
            min(extent.xmax, self.extent.xmax),
            max(extent.ymin, self.extent.ymin),
            min(extent.ymax, self.extent.ymax),
        ]
        query_extent = Extent(*query_extent)
        # Retrieve centroids
        (xmin, xmax), (ymin, ymax) = self.query_centroids(
            x=[query_extent.xmin, query_extent.xmax],
            y=[query_extent.ymin, query_extent.ymax],
        )

        # Retrieve univariate x and y labels within the extent
        x_indices = np.where(np.logical_and(self.x_centroids >= xmin, self.x_centroids <= xmax))[0]
        y_indices = np.where(np.logical_and(self.y_centroids >= ymin, self.y_centroids <= ymax))[0]
        # Retrieve labels corresponding to the combination of all (x,y) indices
        return self._get_dict_labels_combo(x_indices, y_indices)

    def get_partitions_around_point(self, x, y, distance=None, size=None):
        """Return the partition labels with data within the distance/size from a point."""
        extent = get_extent_around_point(x, y, distance=distance, size=size)
        return self.get_partitions_by_extent(extent=extent)

    def directories_by_extent(self, extent):
        """Return the directory trees with data within the specified extent."""
        dict_labels = self.get_partitions_by_extent(extent=extent)
        return self._directories(dict_labels=dict_labels)

    def directories_around_point(self, x, y, distance=None, size=None):
        """Return the directory trees with data within the specified distance from a point."""
        dict_labels = self.get_partitions_around_point(x=x, y=y, distance=distance, size=size)
        return self._directories(dict_labels=dict_labels)

    def add_labels(self, df, x, y, remove_invalid_rows=True):
        """Add partitions labels to the dataframe.

        Parameters
        ----------
        df : pandas.DataFrame, dask.dataframe.DataFrame, polars.DataFrame, pyarrow.Table or polars.LazyFrame
            Dataframe to which add partitions centroids.
        x : str
            Column name with the x coordinate.
        y : str
            Column name with the y coordinate.
        remove_invalid_rows: bool, optional
            Whether to remove dataframe rows for which coordinates are invalid or out of the partitioning extent.
            The default is ``True``.

        Returns
        -------
        df : pandas.DataFrame, dask.dataframe.DataFrame, polars.DataFrame, pyarrow.Table or polars.LazyFrame
            Dataframe with the partitions label(s) column(s).

        """
        check_valid_dataframe(df)
        check_valid_x_y(df, x=x, y=y)
        x_arr = df_get_column(df, column=x)
        y_arr = df_get_column(df, column=y)
        # Retrieve labels
        # - If n_level = 1: array
        # - If n_level = 2: tuple
        labels = self.query_labels(x_arr, y_arr)
        if self.n_levels == 1:
            labels = [labels]
        # Add labels to dataframe
        for partition, values in zip(self.levels, labels, strict=False):
            df = df_add_column(df=df, column=partition, values=values)
        # Check if invalid labels
        invalid_rows = labels[0] == "nan"
        invalid_rows_indices = np.where(invalid_rows)[0]
        if invalid_rows_indices.size > 0:
            if not remove_invalid_rows:
                raise ValueError(f"Invalid labels at rows: {invalid_rows_indices.tolist()}")
            # Remove invalid labels if remove_invalid_rows=True
            df = df_select_valid_rows(df, valid_rows=~invalid_rows)
        return df

    def add_centroids(self, df, x, y, x_coord=None, y_coord=None, remove_invalid_rows=True):
        """Add partitions centroids to the dataframe.

        Parameters
        ----------
        df : pandas.DataFrame, dask.dataframe.DataFrame, polars.DataFrame, pyarrow.Table or polars.LazyFrame
            Dataframe to which add partitions centroids.
        x : str
            Column name with the x coordinate.
        y : str
            Column name with the y coordinate..
        x_coord : str, optional
            Name of the new column with the centroids x  coordinates.
            The default is "x_c".
        y_coord : str, optional
            Name of the new column with the centroids y coordinates.
            The default is "y_c".
        remove_invalid_rows: bool, optional
            Whether to remove dataframe rows for which coordinates are invalid or out of the partitioning extent.
            The default is ``True``.

        Returns
        -------
        df : pandas.DataFrame, dask.dataframe.DataFrame, polars.DataFrame, pyarrow.Table or polars.LazyFrame
            Dataframe with the partitions centroids x and y coordinates columns.

        """
        # Check inputs and retrieve default values
        check_valid_dataframe(df)
        check_valid_x_y(df, x=x, y=y)
        if x_coord is None:
            x_coord = self._x_coord
        if y_coord is None:
            y_coord = self._y_coord
        # Retrieve x and y coordinates arrays
        x_arr = df_get_column(df, column=x)
        y_arr = df_get_column(df, column=y)
        # Retrieve centroids tuple (x, y)
        x_centroids, y_centroids = self.query_centroids(x_arr, y_arr)
        # Add centroids to dataframe
        df = df_add_column(df=df, column=x_coord, values=x_centroids)
        df = df_add_column(df=df, column=y_coord, values=y_centroids)
        # Check if invalid labels
        invalid_rows = np.isnan(x_centroids)
        invalid_rows_indices = np.where(invalid_rows)[0]
        if invalid_rows_indices.size > 0:
            if not remove_invalid_rows:
                raise ValueError(f"Invalid centroids at rows: {invalid_rows_indices.tolist()}")
            # Remove invalid labels if remove_invalid_rows=True
            df = df_select_valid_rows(df, valid_rows=~invalid_rows)
        return df

    def to_xarray(self, df, spatial_coords=None, aux_coords=None):
        """Convert dataframe to spatial xarray Dataset based on partitions centroids.

        This routine assumes that you have grouped and aggregated the dataframe over
        the partition labels or the partition centroids!

        Please add the partition centroids to the dataframe with ``add_centroids`` before calling this method.
        Please specify the partition centroids x and y columns in the ``spatial_coords`` argument.

        Please also specify the presence of auxiliary coordinates (indices) with ``aux_coords``.
        The array cells with coordinates not included in the dataframe will have NaN values.
        """
        # Check inputs
        check_valid_dataframe(df)
        spatial_coords = _ensure_indices_list(spatial_coords)  # [] if None
        aux_coords = _ensure_indices_list(aux_coords)  # [] if None

        # Ensure dataframe is pandas
        df = df_to_pandas(df)

        # Reset dataframe indices if present
        src_indices = df.index.names  # no index returns None
        src_indices = _ensure_indices_list(src_indices)
        if src_indices:
            df = df.reset_index()

        # Check aux_coords are in df (index or column)
        # - If aux_coords were already in the DataFrame index, no need to specify it.
        if aux_coords:
            for coord in aux_coords:
                if coord not in df.columns:
                    raise ValueError(f"Auxiliary coordinate '{coord}' not found in DataFrame columns or index.")
        # Check spatial coords are in df (if specified)
        if spatial_coords:
            for coord in spatial_coords:
                if coord not in df.columns and coord not in src_indices:
                    raise ValueError(f"Spatial coordinate '{coord}' not found in DataFrame columns or index.")
        else:  # tentative guess and raise error if not present
            spatial_coords = [self._x_coord, self._y_coord]
            if self._x_coord not in df.columns or self._y_coord not in df.columns:
                raise ValueError(
                    "Partitiong centroids not found in the dataframe. Please add partitions centroids "
                    "using the 'add_centroids' method and specify the columns in the 'spatial_coords' "
                    "argument of 'to_xarray'.",
                )
        # Finalize auxiliary coords
        possible_coords = np.unique([*spatial_coords, *aux_coords, *src_indices]).tolist()
        possible_aux_coords = set(possible_coords).symmetric_difference(set(spatial_coords))
        aux_coords = possible_aux_coords.difference(set(self.levels))  # exclude also partition names
        coords = list(spatial_coords) + list(aux_coords)

        # Ensure valid coordinates types
        # - Ensure indices are int, float or str (no categorical)
        # - Ensure spatial indices are float
        df = _ensure_valid_coordinates_dtype(df, spatial_coords=spatial_coords, aux_coords=aux_coords)

        # Set coordinates as MultiIndex
        df = df.set_index(coords)

        # Define dictionary of current indices
        dict_indices = {coord: df.index.get_level_values(coord).unique().to_numpy() for coord in coords}

        # Update dictionary with the full x and centroids
        dict_indices[spatial_coords[0]] = self.x_centroids
        dict_indices[spatial_coords[1]] = self.y_centroids

        # Create an empty DataFrame with the MultiIndex
        multi_index = pd.MultiIndex.from_product(
            dict_indices.values(),
            names=dict_indices.keys(),
        )
        empty_df = pd.DataFrame(index=multi_index)

        # Create final dataframe
        df_full = empty_df.join(df, how="left")

        # Reshape to xarray
        ds = df_full.to_xarray()

        return ds


def _ensure_valid_coordinates_dtype(df, spatial_coords, aux_coords):
    for column in spatial_coords:
        df[column] = df[column].astype(float)
    for column in aux_coords:
        if df.dtypes[column].name == "category":
            df[column] = df[column].astype(str)
    return df


class XYPartitioning(Base2DPartitioning):
    """
    Handles partitioning of data into x and y regularly spaced bins.

    Parameters
    ----------
    size : int, float, tuple, list
        The size value(s) of the bins.
        The function interprets the input as follows:
        - int or float: The same size is enforced in both x and y directions.
        - tuple or list: The bin size for the x and y directions.
    extent : list
        The extent for the partitioning specified as ``[xmin, xmax, ymin, ymax]``.
    levels: list, optional
        Names of the x and y partitions.
        The default is ``["xbin", "ybin"]``.
    order : list, optional
        The order of the x and y partitions when writing partitioned datasets.
        The default, ``None``, corresponds to ``levels``.
    flavor : str, optional
        This argument governs the directories names of partitioned datasets.
        The default, ``None``, name the directories with the partitions labels (DirectoryPartitioning).
        The option ``"hive"``, name the directories with the format ``{partition_name}={partition_label}``.
    """

    def __init__(
        self,
        size,
        extent,
        levels=None,
        order=None,
        flavor=None,
        labels_decimals=None,
    ):

        # Check and set extent
        self.extent = check_extent(extent)
        # Check and set partitions size (except maybe last one)
        self.size = _check_size(size)
        # Set partition names
        self.levels = check_default_levels(levels=levels, default_levels=["xbin", "ybin"])
        # Calculate partitions bounds
        x_bounds = get_bounds(size=self.size[0], vmin=self.extent.xmin, vmax=self.extent.xmax)
        y_bounds = get_bounds(size=self.size[1], vmin=self.extent.ymin, vmax=self.extent.ymax)
        # Define options for labels
        if labels_decimals is None:
            labels_decimals = get_n_decimals(self.size[0]) + 1, get_n_decimals(self.size[1]) + 1
        self._labels_decimals = _check_labels_decimals(labels_decimals)
        # Initialize private attributes for labels
        self._xlabels = None
        self._ylabels = None
        # Initialize class
        super().__init__(
            levels=self.levels,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            order=order,
            flavor=flavor,
        )

    # -----------------------------------------------------------------------------------.
    def _custom_labels_function(self, x_indices, y_indices):
        """Return the partition labels as function of the specified 2D partitions indices."""
        x_labels_value = self.x_centroids[x_indices].round(self._labels_decimals[0])
        y_labels_value = self.y_centroids[y_indices].round(self._labels_decimals[1])
        if self._labels_decimals[0] == 0:
            x_labels_value = x_labels_value.astype(int)
        if self._labels_decimals[1] == 0:
            y_labels_value = y_labels_value.astype(int)
        x_labels = x_labels_value.astype(str)
        y_labels = y_labels_value.astype(str)
        return x_labels, y_labels

    def to_dict(self):
        """Return the partitioning settings."""
        dictionary = {
            "partitioning_class": self.__class__.__name__,
            "extent": list(self.extent),
            "size": list(self.size),
            "levels": self.levels,
            "order": self.order,
            "flavor": self.flavor,
            "labels_decimals": list(self._labels_decimals),
        }
        return dictionary

    @property
    def x_labels(self):
        """Return the partition labels across the horizontal dimension."""
        if self._xlabels is None:
            x_labels, _ = self.query_labels_by_indices(
                x_indices=np.arange(0, self.n_x),
                y_indices=np.zeros(self.n_x),
            )
            self._xlabels = x_labels
        return self._xlabels

    @property
    def y_labels(self):
        """Return the partition labels across the vertical dimension."""
        if self._ylabels is None:
            _, y_labels = self.query_labels_by_indices(
                x_indices=np.zeros(self.n_y),
                y_indices=np.arange(0, self.n_y),
            )
            self._ylabels = y_labels
        return self._ylabels


class TilePartitioning(Base2DPartitioning):
    """
    Handles partitioning of data into tiles.

    Parameters
    ----------
    size : int, float, tuple, list
        The size value(s) of the bins.
        The function interprets the input as follows:
        - int or float: The same size is enforced in both x and y directions.
        - tuple or list: The bin size for the x and y directions.
    extent : list
        The extent for the partitioning specified as ``[xmin, xmax, ymin, ymax]``.
    n_levels: int
        The number of tile partitioning levels.
        If ``n_levels=2``, a (x,y) label is assigned to each tile.
        If ``n_levels=1``, a unique id label is assigned to each tile combining the x and y tile indices.
        The ``origin`` and ``direction`` parameters governs its value.
    levels: list, optional
         If ``n_levels>=2``, the first two names must correspond to the x and y partitions.
         The first two levels must
         The default with ``n_levels=1`` is ``["tile"]``.
         The default with ``n_levels=2`` is ``["x", "y"]``.
    origin: str, optional
        The origin of the Y axis. Either ``"bottom"`` or ``"top"``.
        TMS tiles assumes ``origin="top"``.
        Google Maps tiles assumes ``origin="bottom"``.
        The default is ``"bottom"``.
    direction: str, optional
        The direction to follow to define tile ids if ``levels=1`` is specified.
        Valid direction values are "x" and "y".
        ``direction=x`` numbers the tiles rows by rows.
        ``direction=y`` numbers the tiles columns by columns.
    justify: bool, optional
        Whether to justify the labels to ensure having all same number of characters.
        0 is added on the left side of the labels to justify the length.
        THe default is ``False``.
    order : list, optional
        The order of the partitions when writing partitioned datasets.
        The default, ``None``, corresponds to ``levels``.
    flavor : str, optional
        This argument governs the directories names of partitioned datasets.
        The default, ``None``, name the directories with the partitions labels (DirectoryPartitioning).
        The option ``"hive"``, name the directories with the format ``{partition_name}={partition_label}``.
    """

    def __init__(
        self,
        size,
        extent,
        n_levels,
        levels=None,
        origin="bottom",
        direction="x",
        justify=False,
        flavor=None,
        order=None,
    ):
        # Check levels
        if n_levels not in [1, 2]:
            raise ValueError("Invalid value for 'levels'. Must be 1 or 2.")
        default_levels_dict = {1: "tile", 2: ["x", "y"]}
        levels = check_default_levels(levels=levels, default_levels=default_levels_dict[n_levels])
        if len(levels) != n_levels:
            raise ValueError(f"{n_levels} n_levels specified, but {len(levels)} partitions names specified.")
        # Check and set extent
        self.extent = check_extent(extent)
        # Check and set partitions size (except maybe last one)
        self.size = _check_size(size)
        # Calculate partitions bounds
        x_bounds = get_bounds(size=self.size[0], vmin=self.extent.xmin, vmax=self.extent.xmax)
        y_bounds = get_bounds(size=self.size[1], vmin=self.extent.ymin, vmax=self.extent.ymax)
        # Define tiling options
        if direction not in ["x", "y"]:
            raise ValueError("Invalid value for 'direction'. Must be 'x' or 'y'.")
        if origin not in ["top", "bottom"]:
            raise ValueError("Invalid value for 'origin'. Must be 'top' or 'bottom'.")
        self.direction = direction
        self.origin = origin
        self.justify = justify
        # Initialize class
        super().__init__(
            levels=levels,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            order=order,
            flavor=flavor,
        )

    # -----------------------------------------------------------------------------------.
    def _custom_labels_function(self, x_indices, y_indices):
        """Return the partition labels for the specified x,y indices based on the direction, origin, and levels."""
        if self.n_levels == 2:
            return get_tile_xy_labels(
                x_indices,
                y_indices,
                origin=self.origin,
                n_x=self.n_x,
                n_y=self.n_y,
                justify=self.justify,
            )
        # n_levels == 1
        return get_tile_id_labels(
            x_indices,
            y_indices,
            origin=self.origin,
            direction=self.direction,
            n_x=self.n_x,
            n_y=self.n_y,
            justify=self.justify,
        )

    def to_dict(self):
        """Return the partitioning settings."""
        dictionary = {
            "partitioning_class": self.__class__.__name__,
            "extent": list(self.extent),
            "size": list(self.size),
            "n_levels": self.n_levels,
            "levels": self.levels,
            "origin": self.origin,
            "direction": self.direction,
            "justify": self.justify,
            "order": self.order,
            "flavor": self.flavor,
        }
        return dictionary


class LonLatPartitioning(XYPartitioning):
    """Handles geographic partitioning of data based on longitude and latitude bin sizes within a defined extent.

    The last bin size (in lon and lat direction) might not be of size ``size`` !

    Parameters
    ----------
    size : float
        The uniform size for longitude and latitude binning.
        Carefully consider the size of the partitions.
        Earth partitioning by:
        - 1° degree corresponds to 64800 directories (360*180)
        - 5° degree corresponds to 2592 directories (72*36)
        - 10° degree corresponds to 648 directories (36*18)
        - 15° degree corresponds to 288 directories (24*12)
    levels: list, optional
        Names of the longitude and latitude partitions.
        The default is ``["lon_bin", "lat_bin"]``.
    extent : list, optional
        The geographical extent for the partitioning specified as ``[xmin, xmax, ymin, ymax]``.
        Default is the whole Earth: ``[-180, 180, -90, 90]``.
    order : list, optional
        The order of the partitions when writing partitioned datasets.
        The default, ``None``, corresponds to ``levels``.
    flavor : str, optional
        This argument governs the directories names of partitioned datasets.
        The default, `"hive"``, names the directories with the format ``{partition_name}={partition_label}``.
        If ``None``, names the directories with the partitions labels (DirectoryPartitioning).

    Inherits:
    ----------
    XYPartitioning
    """

    def __init__(
        self,
        size,
        extent=[-180, 180, -90, 90],
        levels=None,
        flavor="hive",
        order=None,
        labels_decimals=None,
    ):
        levels = check_default_levels(levels=levels, default_levels=["lon_bin", "lat_bin"])
        super().__init__(
            size=size,
            extent=extent,
            levels=levels,
            order=order,
            flavor=flavor,
            labels_decimals=labels_decimals,
        )
        self._x_coord = "lon_c"  # default name for x centroid column for add_centroids
        self._y_coord = "lat_c"  # default name for y centroid column for add_centroids

    def get_partitions_around_point(self, lon, lat, distance=None, size=None):
        """Return the partition labels with data within the distance/size from a point."""
        extent = get_geographic_extent_around_point(
            lon=lon,
            lat=lat,
            distance=distance,
            size=size,
        )
        return self.get_partitions_by_extent(extent=extent)

    def get_partitions_by_country(self, name, padding=None):
        """Return the partition labels enclosing the specified country."""
        extent = get_country_extent(name=name, padding=padding)
        return self.get_partitions_by_extent(extent=extent)

    def get_partitions_by_continent(self, name, padding=None):
        """Return the partition labels enclosing the specified continent."""
        extent = get_continent_extent(name=name, padding=padding)
        return self.get_partitions_by_extent(extent=extent)

    def directories_by_country(self, name, padding=None):
        """Return the directory trees with data within a country."""
        dict_labels = self.get_partitions_by_country(name=name, padding=padding)
        return self._directories(dict_labels=dict_labels)

    def directories_by_continent(self, name, padding=None):
        """Return the directory trees with data within a continent."""
        dict_labels = self.get_partitions_by_continent(name=name, padding=padding)
        return self._directories(dict_labels=dict_labels)

    def directories_around_point(self, lon, lat, distance=None, size=None):
        """Return the directory trees with data within the distance/size from a point."""
        dict_labels = self.get_partitions_around_point(lon=lon, lat=lat, distance=distance, size=size)
        return self._directories(dict_labels=dict_labels)
