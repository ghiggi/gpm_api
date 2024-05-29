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

import dask.dataframe as dd
import numpy as np
import pandas as pd
import polars as pl

from gpm.utils.geospatial import (
    Extent,
    _check_size,
    check_extent,
    get_continent_extent,
    get_country_extent,
    get_extent_around_point,
    get_geographic_extent_around_point,
)

# Future methods:
# to_shapely
# to_spherically (geographic)
# to_geopandas [lat_bin, lon_bin, geometry]


def check_valid_dataframe(func):
    """Decorator checking if the first argument is a dataframe.

    Accepted dataframes: `pandas.DataFrame`, `dask.DataFrame` or a `polars.DataFrame`.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if 'df' is in kwargs, otherwise assume it is the first positional argument
        df = kwargs.get("df", args[1] if len(args) == 2 else None)
        # Validate the DataFrame
        if not isinstance(df, (pd.DataFrame, dd.DataFrame, pl.DataFrame)):
            raise TypeError("The 'df' argument must be either a pandas.DataFrame or a polars.DataFrame.")
        return func(*args, **kwargs)

    return wrapper


def check_valid_x_y(df, x, y):
    """Check if the x and y columns are in the dataframe."""
    if y not in df:
        raise ValueError(f"y='{y}' is not a column of the dataframe.")
    if x not in df:
        raise ValueError(f"x='{x}' is not a column of the dataframe.")


def _check_partitions(xbin, ybin, partitions):
    if {xbin, ybin} != set(partitions):
        raise ValueError(f"'partitions' ({partitions}) does not match with xbin ({xbin}) and ybin ({ybin}).")
    return partitions


# check_partitioning_flavor


# TODO: get indices and then drop by indices ? Dask?


def ensure_xy_without_nan_values(df, x, y, remove_invalid_rows=True):
    """Ensure valid coordinates in the dataframe."""
    # Remove NaN vales
    if remove_invalid_rows:
        if isinstance(df, pd.DataFrame):
            return df.dropna(subset=[x, y])
        return df.filter(~pl.col(x).is_null() | ~pl.col(y).is_null())

    # Check no NaN values
    if isinstance(df, pd.DataFrame):  # noqa
        indices = df[[x, y]].isna().any(axis=1)
    else:
        indices = df[x].is_null() | df[y].is_null()
    if indices.any():
        rows_indices = np.where(indices)[0].tolist()
        raise ValueError(f"Null values present in columns {x} and {y} at rows: {rows_indices}")
    return df


def _remove_outside_cat_flags(df, column):
    df = df.with_columns(df[column].cast(str).cast(pl.Categorical).alias(column))
    return df


def ensure_valid_partitions(df, xbin, ybin, remove_invalid_rows=True):
    """Ensure valid partitions labels in the dataframe."""
    # Remove NaN values
    if remove_invalid_rows:
        if isinstance(df, pd.DataFrame):
            return df.dropna(subset=[xbin, ybin])
        df = df.filter(~pl.col(xbin).is_in(["outside_right", "outside_left"]))
        df = df.filter(~pl.col(ybin).is_in(["outside_right", "outside_left"]))
        df = df.filter(~pl.col(xbin).is_null() | ~pl.col(ybin).is_null())
        df = _remove_outside_cat_flags(df, column=xbin)
        df = _remove_outside_cat_flags(df, column=ybin)
        return df

    # Check no invalid partitions (NaN or polars outside_right/outside_left)
    if isinstance(df, pd.DataFrame):
        indices = df[[xbin, ybin]].isna().any(axis=1)
    else:
        indices = df[xbin].is_in(["outside_right", "outside_left"]) | df[ybin].is_in(["outside_right", "outside_left"])

    if indices.any():
        rows_indices = np.where(indices)[0].tolist()
        raise ValueError(f"Out of extent x,y coordinates at rows: {rows_indices}")
    # Ensure no more "outside_right"/"outside_left" flag
    if isinstance(df, pl.DataFrame):
        df = _remove_outside_cat_flags(df, column=xbin)
        df = _remove_outside_cat_flags(df, column=ybin)
    return df


def get_partition_dir_name(partition_name, partition_labels, partitioning_flavor):
    """Return the directories name of a partition."""
    if partitioning_flavor == "hive":
        return reduce(np.char.add, [partition_name, "=", partition_labels, os.sep])
    return np.char.add(partition_labels, os.sep)


def get_directories(dict_labels, partitions, partitioning_flavor):
    """Return the directory trees of a partitioned dataset."""
    list_dir_names = []
    for partition in partitions:
        dir_name = get_partition_dir_name(
            partition_name=partition,
            partition_labels=dict_labels[partition],
            partitioning_flavor=partitioning_flavor,
        )
        list_dir_names.append(dir_name)
    dir_trees = reduce(np.char.add, list_dir_names)
    dir_trees = np.char.rstrip(dir_trees, os.sep)
    return dir_trees


def get_array_combinations(x, y):
    """Return all the combinations between the two array."""
    # Create the mesh grid
    grid1, grid2 = np.meshgrid(x, y)
    # Stack and reshape the grid arrays to get combinations
    combinations = np.vstack([grid1.ravel(), grid2.ravel()]).T
    return combinations


def get_n_decimals(number):
    """Get the number of decimals of a number."""
    number_str = str(number)
    decimal_index = number_str.find(".")

    if decimal_index == -1:
        return 0  # No decimal point found

    # Count the number of characters after the decimal point
    return len(number_str) - decimal_index - 1


def get_breaks(size, vmin, vmax):
    """Define partitions edges."""
    breaks = np.arange(vmin, vmax, size)
    if breaks[-1] != vmax:
        breaks = np.append(breaks, np.array([vmax]))
    return breaks


def get_centroids(size, vmin, vmax):
    """Define partitions centroids."""
    breaks = get_breaks(size, vmin=vmin, vmax=vmax)
    centroids = breaks[0:-1] + size / 2
    return centroids


def get_labels(size, vmin, vmax):
    """Define partitions labels (rounding partitions centroids)."""
    n_decimals = get_n_decimals(size)
    centroids = get_centroids(size, vmin, vmax)
    return centroids.round(n_decimals + 1).astype(str)


def get_breaks_and_centroids(size, vmin, vmax):
    """Return the partitions edges and partitions centroids."""
    breaks = get_breaks(size, vmin=vmin, vmax=vmax)
    centroids = get_centroids(size, vmin=vmin, vmax=vmax)
    return breaks, centroids


def get_breaks_and_labels(size, vmin, vmax):
    """Return the partitions edges and partitions labels."""
    breaks = get_breaks(size, vmin=vmin, vmax=vmax)
    labels = get_labels(size, vmin=vmin, vmax=vmax)
    return breaks, labels


def query_labels(values, breaks, labels):
    """Return the partition labels for the specified coordinates.

    Invalid values (NaN, None) or out of breaks values returns NaN.
    """
    values = np.atleast_1d(np.asanyarray(values)).astype(float)
    return pd.cut(values, bins=breaks, labels=labels, include_lowest=True, right=True)


def query_centroids(values, breaks, centroids):
    """Return the partition centroids for the specified coordinates.

    Invalid values (NaN, None) or out of breaks values returns NaN.
    """
    values = np.atleast_1d(np.asanyarray(values)).astype(float)
    return pd.cut(values, bins=breaks, labels=centroids, include_lowest=True, right=True).astype(float)


# TODO: ADD dask function   (maybe function better than cut and breaks?) --> write_bucket test should work
# TODO: ADD tile code
# TODO: add check-flavour --> directory instead of None
# TODO: Add centroids
# TODO: Add readers !
# df_dask[xbin].map_partitions(pd.cut, bins)

# add_polars_xy_centroids_midpoints


# def add_polars_tile_labels(df, size, extent, x, y, tile_id):
#     check_valid_x_y(df, x, y)
#     raise NotImplementedError


# def add_pandas_tile_labels(df, size, extent, x, y, tile_id):
#     check_valid_x_y(df, x, y)
#     raise NotImplementedError


# def add_dask_xy_labels(df, size, extent, x, y, xbin, ybin, remove_invalid_rows=True):
#     """Add partitions labels to a dask DataFrame based on x, y coordinates."""
#     import dask.dataframe as dd
#     # df = dask.dataframe.from_pandas(df, npartitions=2)

#     # Check x,y names
#     check_valid_x_y(df, x, y)

#     # Check/remove rows with NaN x,y columns
#     # df = ensure_xy_without_nan_values(df, x=x, y=y, remove_invalid_rows=remove_invalid_rows)

#     # Retrieve breaks and labels (N and N+1)
#     cut_x_breaks, cut_x_labels = get_breaks_and_labels(size[0], vmin=extent[0], vmax=extent[1])
#     cut_y_breaks, cut_y_labels = get_breaks_and_labels(size[1], vmin=extent[2], vmax=extent[3])

#     # Add partitions labels columns
#     xbin_values = pd.Series(query_labels(df[x].compute().to_numpy(),breaks=cut_x_breaks, labels=cut_x_labels))
#     ybin_values = pd.Series(query_labels(df[y].compute().to_numpy(),breaks=cut_y_breaks, labels=cut_y_labels))
#     df[xbin] = dd.from_pandas(xbin_values, npartitions=df.npartitions)
#     df[ybin] = dd.from_pandas(ybin_values, npartitions=df.npartitions)

#     # # Check/remove rows with invalid partitions (NaN)
#     # df = ensure_valid_partitions(df, xbin=xbin, ybin=ybin, remove_invalid_rows=remove_invalid_rows)
#     return df


def add_pandas_xy_labels(df, size, extent, x, y, xbin, ybin, remove_invalid_rows=True):
    """Add partitions labels to a pandas DataFrame based on x, y coordinates."""
    # Check x,y names
    check_valid_x_y(df, x, y)
    # Check/remove rows with NaN x,y columns
    df = ensure_xy_without_nan_values(df, x=x, y=y, remove_invalid_rows=remove_invalid_rows)
    # Retrieve breaks and labels (N and N+1)
    cut_x_breaks, cut_x_labels = get_breaks_and_labels(size[0], vmin=extent[0], vmax=extent[1])
    cut_y_breaks, cut_y_labels = get_breaks_and_labels(size[1], vmin=extent[2], vmax=extent[3])
    # Add partitions labels columns
    df = df.copy()
    df[xbin] = query_labels(df[x].to_numpy(), breaks=cut_x_breaks, labels=cut_x_labels)
    df[ybin] = query_labels(df[y].to_numpy(), breaks=cut_y_breaks, labels=cut_y_labels)
    # Check/remove rows with invalid partitions (NaN)
    df = ensure_valid_partitions(df, xbin=xbin, ybin=ybin, remove_invalid_rows=remove_invalid_rows)
    return df


def add_polars_xy_labels(df, x, y, size, extent, xbin, ybin, remove_invalid_rows=True):
    """Add partitions to a polars DataFrame based on x, y coordinates."""
    # Check x,y names
    check_valid_x_y(df, x, y)
    # Check/remove rows with null x,y columns
    df = ensure_xy_without_nan_values(df, x=x, y=y, remove_invalid_rows=remove_invalid_rows)
    # Retrieve breaks and labels (N and N+1)
    cut_x_breaks, cut_x_labels = get_breaks_and_labels(size[0], vmin=extent[0], vmax=extent[1])
    cut_y_breaks, cut_y_labels = get_breaks_and_labels(size[1], vmin=extent[2], vmax=extent[3])
    # Add outside labels for polars cut function
    cut_x_labels = ["outside_left", *cut_x_labels, "outside_right"]
    cut_y_labels = ["outside_left", *cut_y_labels, "outside_right"]
    # Deal with left inclusion
    cut_x_breaks[0] = cut_x_breaks[0] - 1e-8
    cut_y_breaks[0] = cut_y_breaks[0] - 1e-8
    # Add partitions columns
    df = df.with_columns(
        pl.col(x).cut(cut_x_breaks, labels=cut_x_labels, left_closed=False).alias(xbin),
        pl.col(y).cut(cut_y_breaks, labels=cut_y_labels, left_closed=False).alias(ybin),
    )
    # Check/remove rows with invalid partitions (out of extent or Null)
    df = ensure_valid_partitions(df, xbin=xbin, ybin=ybin, remove_invalid_rows=remove_invalid_rows)
    return df


def _ensure_indices_list(indices):
    if indices is None:
        indices = []
    indices = [indices] if isinstance(indices, str) else list(indices)
    if indices == [None]:  # what is returned by df.index.names if no index !
        indices = []
    return indices


def _preprocess_dataframe_indices(df, spatial_indices, aux_indices):
    # spatial_indices can be []
    spatial_indices = _ensure_indices_list(spatial_indices)
    aux_indices = _ensure_indices_list(aux_indices)
    # Reset indices if present
    src_indices = df.index.names  # no index returns None
    src_indices = _ensure_indices_list(src_indices)
    if set(spatial_indices).issubset(src_indices):
        df = df.reset_index()

    # Define future dataset indices
    indices = np.unique([*spatial_indices, *aux_indices, *src_indices]).tolist()
    non_spatial_indices = set(indices).symmetric_difference(set(spatial_indices))

    # Ensure indices are float or string and that spatial indices are float
    for idx_name in spatial_indices:
        df[idx_name] = df[idx_name].astype(float)
    for idx_name in non_spatial_indices:
        if df.dtypes[idx_name].name == "category":
            df[idx_name] = df[idx_name].astype(str)
    df = df.set_index(indices)

    # Define dictionary of indices
    dict_indices = {idx_name: df.index.get_level_values(idx_name).unique().to_numpy() for idx_name in indices}
    return df, dict_indices


def df_to_xarray(df, xbin, ybin, size, extent, new_x=None, new_y=None, indices=None):
    """Convert dataframe to xarray Dataset based on specified partitions centroids.

    The partitioning cells not present in the dataframe are set to NaN.
    """
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Ppreprocess dataframe indices
    # - Ensure indices are int, float or str (no categorical)
    # - Ensure the returned dataframe is without index
    # - List the indices which are not the partition centroids
    spatial_indices = [xbin, ybin]
    df, dict_indices = _preprocess_dataframe_indices(df, spatial_indices=spatial_indices, aux_indices=indices)

    # Update dictionary of indices with all possible centroids
    x_centroids = get_centroids(size[0], vmin=extent[0], vmax=extent[1])
    y_centroids = get_centroids(size[1], vmin=extent[2], vmax=extent[3])
    dict_indices[xbin] = x_centroids
    dict_indices[ybin] = y_centroids

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

    # Rename dictionary
    rename_dict = {}
    if new_x is not None:
        rename_dict[xbin] = new_x
    if new_y is not None:
        rename_dict[ybin] = new_y
    ds = ds.rename(rename_dict)
    return ds


class XYPartitioning:
    """
    Handales partitioning of data into x and y bins.

    Parameters
    ----------
    xbin : float
        Identifier for the x bin.
    ybin : float
        Identifier for the y bin.
    size : int, float, tuple, list
        The size value(s) of the bins.
        The function interprets the input as follows:
        - int or float: The same size is enforced in both x and y directions.
        - tuple or list: The bin size for the x and y directions.
    extent : list
        The extent for the partitioning specified as ``[xmin, xmax, ymin, ymax]``.
    partitions : list
        The order of the partitions xbin and ybin (i.e. when writing to disk partitioned datasets)
        The default, ``None``, corresponds to ``[xbin, ybin]``.
    partitioning_flavor : str
        This argument governs the directories names of partitioned datasets.
        The default, ``None``, name the directories with the partitions labels (DirectoryPartitioning).
        The option ``"hive"``, name the directories with the format ``{partition_name}={partition_label}``.
    """

    def __init__(self, xbin, ybin, size, extent, partitioning_flavor=None, partitions=None):
        # Define extent
        self.extent = check_extent(extent)
        # Define bin size
        self.size = _check_size(size)
        # Define bin names
        self.xbin = xbin
        self.ybin = ybin
        # Define partitions and partitioning flavour
        if partitions is None:
            self.partitions = [self.xbin, self.ybin]
        else:
            self.partitions = _check_partitions(xbin=xbin, ybin=ybin, partitions=partitions)
        self.partitioning_flavor = partitioning_flavor
        # Define breaks, centroids and labels
        self.x_breaks = get_breaks(size=self.size[0], vmin=self.extent.xmin, vmax=self.extent.xmax)
        self.y_breaks = get_breaks(size=self.size[1], vmin=self.extent.ymin, vmax=self.extent.ymax)
        self.x_centroids = get_centroids(size=self.size[0], vmin=self.extent.xmin, vmax=self.extent.xmax)
        self.y_centroids = get_centroids(size=self.size[1], vmin=self.extent.ymin, vmax=self.extent.ymax)
        self.x_labels = get_labels(size=self.size[0], vmin=self.extent.xmin, vmax=self.extent.xmax)
        self.y_labels = get_labels(size=self.size[1], vmin=self.extent.ymin, vmax=self.extent.ymax)
        # Define info
        self.shape = (len(self.x_labels), len(self.y_labels))
        self.n_partitions = len(self.x_labels) * len(self.y_labels)
        self.n_levels = len(self.partitions)
        self.n_x = self.shape[0]
        self.n_y = self.shape[1]

    @check_valid_dataframe
    def add_labels(self, df, x, y, remove_invalid_rows=True):
        if isinstance(df, pd.DataFrame):
            return add_pandas_xy_labels(
                df=df,
                x=x,
                y=y,
                size=self.size,
                extent=self.extent,
                xbin=self.xbin,
                ybin=self.ybin,
                remove_invalid_rows=remove_invalid_rows,
            )
        return add_polars_xy_labels(
            df=df,
            x=x,
            y=y,
            size=self.size,
            extent=self.extent,
            xbin=self.xbin,
            ybin=self.ybin,
            remove_invalid_rows=remove_invalid_rows,
        )

    @check_valid_dataframe
    def to_xarray(self, df, new_x=None, new_y=None, indices=None):
        """Convert a dataframe with partitions centroids to a ``xr.Dataset``."""
        return df_to_xarray(
            df=df,
            xbin=self.xbin,
            ybin=self.ybin,
            size=self.size,
            extent=self.extent,
            new_x=new_x,
            new_y=new_y,
            indices=indices,
        )

    def to_dict(self):
        """Return the partitioning settings."""
        dictionary = {
            "name": self.__class__.__name__,
            "extent": list(self.extent),
            "size": list(self.size),
            "xbin": self.xbin,
            "ybin": self.ybin,
            "partitions": self.partitions,
            "partitioning_flavor": self.partitioning_flavor,
        }
        return dictionary

    def _query_x_labels(self, x):
        """Return the x partition labels for the specified x coordinates."""
        return query_labels(x, breaks=self.x_breaks, labels=self.x_labels).astype(str)

    def _query_y_labels(self, y):
        """Return the y partition labels for the specified y coordinates."""
        return query_labels(y, breaks=self.y_breaks, labels=self.y_labels).astype(str)

    def _query_x_centroids(self, x):
        """Return the x partition centroids for the specified x coordinates."""
        return query_centroids(x, breaks=self.x_breaks, centroids=self.x_centroids)

    def _query_y_centroids(self, y):
        """Return the y partition centroids for the specified y coordinates."""
        return query_centroids(y, breaks=self.y_breaks, centroids=self.y_centroids)

    def query_labels(self, x, y):
        """Return the partition labels for the specified x,y coordinates."""
        return self._query_x_labels(x), self._query_y_labels(y)

    def query_centroids(self, x, y):
        """Return the partition centroids for the specified x,y coordinates."""
        return self._query_x_centroids(x), self._query_y_centroids(y)

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
        xmin, xmax = self._query_x_centroids([query_extent.xmin, query_extent.xmax])
        ymin, ymax = self._query_y_centroids([query_extent.ymin, query_extent.ymax])
        # Retrieve univariate x and y labels within the extent
        x_labels = self.x_labels[np.logical_and(self.x_centroids >= xmin, self.x_centroids <= xmax)]
        y_labels = self.y_labels[np.logical_and(self.y_centroids >= ymin, self.y_centroids <= ymax)]
        # Retrieve combination of all (x,y) labels within the extent
        combinations = get_array_combinations(x_labels, y_labels)
        dict_labels = {
            self.xbin: combinations[:, 0],
            self.ybin: combinations[:, 1],
        }
        return dict_labels

    def get_partitions_around_point(self, x, y, distance=None, size=None):
        """Return the partition labels with data within the distance/size from a point."""
        extent = get_extent_around_point(x, y, distance=distance, size=size)
        return self.get_partitions_by_extent(extent=extent)

    def _directories(self, dict_labels):
        return get_directories(
            dict_labels=dict_labels,
            partitions=self.partitions,
            partitioning_flavor=self.partitioning_flavor,
        )

    @property
    def directories(self):
        """Return the directory trees."""
        combinations = get_array_combinations(self.x_labels, self.y_labels)
        dict_labels = {
            self.xbin: combinations[:, 0],
            self.ybin: combinations[:, 1],
        }
        return self._directories(dict_labels=dict_labels)

    def directories_by_extent(self, extent):
        """Return the directory trees with data within the specified extent."""
        dict_labels = self.get_partitions_by_extent(extent=extent)
        return self._directories(dict_labels=dict_labels)

    def directories_around_point(self, x, y, distance=None, size=None):
        """Return the directory trees with data within the specified distance from a point."""
        dict_labels = self.get_partitions_around_point(x=x, y=y, distance=distance, size=size)
        return self._directories(dict_labels=dict_labels)

    def quadmesh(self, origin="bottom"):
        """Return the quadrilateral mesh.

        A quadrilateral mesh is a grid of M by N adjacent quadrilaterals that are defined via a (M+1, N+1)
        grid of vertices.

        The quadrilateral mesh is accepted by `matplotlib.pyplot.pcolormesh`, `matplotlib.collections.QuadMesh`
        `matplotlib.collections.PolyQuadMesh`.

        Parameters
        ----------
        origin: str
            Origin of the y axis.
            The default is ``bottom``.

        Return
        --------
        np.ndarray
            Quadmesh array of shape (M+1, N+1, 2)
        """
        x_corners, y_corners = np.meshgrid(self.x_breaks, self.y_breaks)
        if origin == "bottom":
            y_corners = y_corners[::-1, :]
        return np.stack((x_corners, y_corners), axis=2)


class GeographicPartitioning(XYPartitioning):
    """
    Handles geographic partitioning of data based on longitude and latitude bin sizes within a defined extent.

    The last bin size (in lon and lat direction) might not be of size ``size` !

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
    xbin : str, optional
        Name of the longitude bin, default is 'lon_bin'.
    ybin : str, optional
        Name of the latitude bin, default is 'lat_bin'.
    extent : list, optional
        The geographical extent for the partitioning specified as [xmin, xmax, ymin, ymax].
        Default is the whole earth: [-180, 180, -90, 90].
    partitions : list
        The order of the partitions xbin and ybin (i.e. when writing to disk partitioned datasets)
        The default, ``None``, corresponds to ``[xbin, ybin]``.
    partitioning_flavor : str
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
        xbin="lon_bin",
        ybin="lat_bin",
        extent=[-180, 180, -90, 90],
        partitioning_flavor="hive",
        partitions=None,
    ):
        super().__init__(
            xbin=xbin,
            ybin=ybin,
            size=size,
            extent=extent,
            partitions=partitions,
            partitioning_flavor=partitioning_flavor,
        )

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

    @check_valid_dataframe
    def to_xarray(self, df, new_x="lon", new_y="lat", indices=None):
        """Convert a dataframe with partitions centroids to a ``xr.Dataset``."""
        return df_to_xarray(
            df,
            xbin=self.xbin,
            ybin=self.ybin,
            size=self.size,
            extent=self.extent,
            new_x=new_x,
            new_y=new_y,
            indices=indices,
        )


# class TilePartitioning:
#     """
#     Handles partitioning of data into tiles within a specified extent.

#     Parameters
#     ----------
#     size : float
#         The size of the tiles.
#     extent : list
#         The extent for the partitioning specified as [xmin, xmax, ymin, ymax].
#     tile_id : str, optional
#         Identifier for the tile bin. The default is ``'tile_id'``.


#     """

#     # Define option
#     # 0 / 1 / 10 / 100
#     # 000/ 001/ 010/ 100
#     # origin="upper"

#     def __init__(self, size, extent, tile_id="tile_id", partitioning_flavor=None):
#         self.size = _check_size(size)
#         self.extent = check_extent(extent)
#         self.tile_id = tile_id

#     @property
#     def bins(self):
#         return [self.tile_id]

#     @check_valid_dataframe
#     def add_labels(self, df, x, y):
#         if isinstance(df, pd.DataFrame):
#             return add_pandas_tile_labels(
#                 df,
#                 x=x,
#                 y=x,
#                 size=self.size,
#                 extent=self.extent,
#                 tile_id=self.tile_id,
#             )
#         return add_polars_tile_labels(
#             df,
#             x=x,
#             y=x,
#             size=self.size,
#             extent=self.extent,
#             tile_id=self.tile_id,
#         )
