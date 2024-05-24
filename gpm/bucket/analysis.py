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
"""This module provide utilities for GPM Geographic Bucket Analysis."""

import numpy as np
import pandas as pd
import polars as pl

from gpm.utils.geospatial import _check_size

# in processing.py --> replace, assign_spatial_partitions, get_bin_partition
# assign_spatial_partitions
# get_bin_partition


def get_bin_partition(values, bin_size):
    """Compute the bins partitioning values.

    Parameters
    ----------
    values : float or array-like
        Values.
    bin_size : float
        Bin size.

    Returns
    -------
    Bin value : float or array-like
        DESCRIPTION.

    """
    return bin_size * np.floor(values / bin_size)


# bin_size = 10
# values = np.array([-180,-176,-175, -174, -171, 170, 166])
# get_bin_partition(values, bin_size)


def assign_spatial_partitions(
    df,
    xbin_name,
    ybin_name,
    xbin_size,
    ybin_size,
    x_column="lat",
    y_column="lon",
):
    """Add partitioning bin columns to dataframe.

    Works for both `dask.dataframe.DataFrame` and `pandas.DataFrame`.
    """
    # Remove invalid coordinates
    df = df[~df[x_column].isna()]
    df = df[~df[y_column].isna()]

    # Add spatial partitions columns to dataframe
    partition_columns = {
        xbin_name: get_bin_partition(df[x_column], bin_size=xbin_size),
        ybin_name: get_bin_partition(df[y_column], bin_size=ybin_size),
    }
    return df.assign(**partition_columns)


def _get_bin_edges(vmin, vmax, size):
    """Get bin edges."""
    return np.arange(vmin, vmax + 1e-10, step=size)


def _get_bin_midpoints(vmin, vmax, size):
    """Get bin midpoints."""
    edges = _get_bin_edges(vmin=vmin, vmax=vmax, size=size)
    return edges[:-1] + np.diff(edges) / 2


def create_spatial_bin_empty_df(
    xbin_size=1,
    ybin_size=1,
    xlim=(-180, 180),
    ylim=(-90, 90),
    xbin_name="xbin",
    ybin_name="ybin",
):
    """Create empty spatial bin DataFrame."""
    # Get midpoints
    x_midpoints = _get_bin_midpoints(vmin=xlim[0], vmax=xlim[1], size=xbin_size)
    y_midpoints = _get_bin_midpoints(vmin=ylim[0], vmax=ylim[1], size=ybin_size)

    # Create the MultiIndex from the combination of x and y bins
    multi_index = pd.MultiIndex.from_product(
        [x_midpoints, y_midpoints],
        names=[xbin_name, ybin_name],
    )

    # Create an empty DataFrame with the MultiIndex
    return pd.DataFrame(index=multi_index)


def add_bin_column(df, column, bin_size, vmin, vmax, bin_name, add_midpoint=True):
    # Keep rows within values
    valid_rows = df[column].between(left=vmin, right=vmax, inclusive="both")
    df = df.loc[valid_rows, :]

    # Get bin edges and midpoints
    bin_edges = _get_bin_edges(vmin=vmin, vmax=vmax, size=bin_size)
    bin_midpoints = _get_bin_midpoints(vmin=vmin, vmax=vmax, size=bin_size)

    # Get bin index
    # - 0 is outside to the left of the bins
    # - -1 is outside to the right
    # --> Subtract 1
    bin_idx = np.digitize(df[column], bins=bin_edges, right=False) - 1

    # Add bin index/midpoint values
    if add_midpoint:
        df[bin_name] = bin_midpoints[bin_idx]
    else:
        df[bin_name] = bin_idx
    return df


def get_n_decimals(number):
    number_str = str(number)
    decimal_index = number_str.find(".")

    if decimal_index == -1:
        return 0  # No decimal point found

    # Count the number of characters after the decimal point
    return len(number_str) - decimal_index - 1


def get_lat_bins(bin_spacing):
    n_decimals = get_n_decimals(bin_spacing)
    lat_buckets = np.arange(-90.0, 90.0 + 1e-6, bin_spacing).round(n_decimals)
    if lat_buckets[-1] != 90.0:
        lat_buckets = np.append(lat_buckets, np.array([90.0]))
    return lat_buckets


def get_lon_bins(bin_spacing):
    n_decimals = get_n_decimals(bin_spacing)
    lon_buckets = np.arange(-180.0, 180.0 + 1e-6, bin_spacing).round(n_decimals)
    if lon_buckets[-1] != 180.0:
        lon_buckets = np.append(lon_buckets, np.array([180.0]))
    return lon_buckets


def get_lon_labels(bin_spacing):
    n_decimals = get_n_decimals(bin_spacing)
    lon_buckets = get_lon_bins(bin_spacing)
    lon_labels = lon_buckets[0:-1] + bin_spacing / 2
    return lon_labels.round(n_decimals + 1)


def get_lat_labels(bin_spacing):
    n_decimals = get_n_decimals(bin_spacing)
    lat_buckets = get_lat_bins(bin_spacing)
    lat_labels = lat_buckets[0:-1] + bin_spacing / 2
    return lat_labels.round(n_decimals + 1)


def get_cut_lat_breaks_labels(bin_spacing):
    lat_labels = get_lat_labels(bin_spacing)
    lat_buckets = get_lat_bins(bin_spacing)
    # Define cut labels
    cut_lat_labels = lat_labels.astype(str).tolist()
    cut_lat_labels = ["outside_left", *cut_lat_labels, "outside_right"]
    # Deal with left inclusion
    cut_lat_breaks = lat_buckets
    cut_lat_breaks[0] = cut_lat_breaks[0] - 1e-6
    return cut_lat_breaks, cut_lat_labels


def get_cut_lon_breaks_labels(bin_spacing):
    lon_labels = get_lon_labels(bin_spacing)
    lon_buckets = get_lon_bins(bin_spacing)
    cut_lon_labels = lon_labels.astype(str).tolist()
    cut_lon_labels = ["outside_left", *cut_lon_labels, "outside_right"]
    # - Deal with left inclusion
    cut_lon_breaks = lon_buckets
    cut_lon_breaks[0] = cut_lon_breaks[0] - 1e-6
    return cut_lon_breaks, cut_lon_labels


def add_spatial_bins(
    df,
    x="x",
    y="y",
    xbin_size=1,
    ybin_size=1,
    xlim=(-180, 180),
    ylim=(-90, 90),
    xbin_name="xbin",
    ybin_name="ybin",
    add_bin_midpoint=True,
):
    # Define x bins
    df = add_bin_column(
        df=df,
        column=x,
        bin_size=xbin_size,
        vmin=xlim[0],
        vmax=xlim[1],
        bin_name=xbin_name,
        add_midpoint=add_bin_midpoint,
    )
    # Define y bins
    return add_bin_column(
        df=df,
        column=y,
        bin_size=ybin_size,
        vmin=ylim[0],
        vmax=ylim[1],
        bin_name=ybin_name,
        add_midpoint=add_bin_midpoint,
    )


def pl_add_geographic_bins(
    df,
    xbin_column,
    ybin_column,
    bin_spacing,
    x_column="lon",
    y_column="lat",
):
    cut_lon_breaks, cut_lon_labels = get_cut_lon_breaks_labels(bin_spacing)
    cut_lat_breaks, cut_lat_labels = get_cut_lat_breaks_labels(bin_spacing)
    return df.with_columns(
        pl.col(x_column).cut(cut_lon_breaks, labels=cut_lon_labels).alias(xbin_column),
        pl.col(y_column).cut(cut_lat_breaks, labels=cut_lat_labels).alias(ybin_column),
    )
    # df.filter(pl.col(xbin_column) == "outside_left")
    # df.filter(pl.col(xbin_column) == "outside_right")


def add_geographic_bins(
    df,
    x,
    y,
    xbin,
    ybin,
    size,
    extent,
    add_bin_midpoint=True,
):
    size = _check_size(size)
    if isinstance(df, pd.DataFrame):
        from gpm.bucket.analysis import add_spatial_bins

        df = add_spatial_bins(
            df=df,
            x=x,
            y=y,
            xbin_name=xbin,
            ybin_name=ybin,
            xbin_size=size[0],
            ybin_size=size[1],
            xlim=extent[0:2],
            ylim=extent[0:2],
            add_bin_midpoint=add_bin_midpoint,
        )
    else:
        # TODO: no extent !
        df = pl_add_geographic_bins(
            df=df,
            xbin_column=xbin,
            ybin_column=ybin,
            bin_spacing=size,
            x_column=x,
            y_column=y,
        )
    return df


####----------------------------------------------------------------.
#### Conversion to xarray Dataset


def pl_df_to_xarray(df, xbin_column, ybin_column, bin_spacing):
    df_stats_pd = df.to_pandas()

    df_stats_pd[xbin_column] = df_stats_pd[xbin_column].astype(float)
    df_stats_pd[ybin_column] = df_stats_pd[ybin_column].astype(float)
    df_stats_pd = df_stats_pd.set_index([xbin_column, ybin_column])

    ## Left join to the spatial bin template dataframe
    # Create the MultiIndex from the combination of x and y bins
    lon_labels = get_lon_labels(bin_spacing)
    lat_labels = get_lat_labels(bin_spacing)
    multi_index = pd.MultiIndex.from_product(
        [lon_labels, lat_labels],
        names=[xbin_column, ybin_column],
    )

    # Create an empty DataFrame with the MultiIndex
    empty_df = pd.DataFrame(index=multi_index)

    # Create final dataframe
    df_stats_pd = empty_df.join(df_stats_pd, how="left")

    # Reshape to xarray
    ds = df_stats_pd.to_xarray()
    return ds.rename({xbin_column: "longitude", ybin_column: "latitude"})


def pd_df_to_xarray(df, xbin, ybin, size):
    size = _check_size(size)
    if set(df.index.names) != {xbin, ybin}:
        df[xbin] = df[xbin].astype(float)
        df[ybin] = df[ybin].astype(float)
        df = df.set_index([xbin, ybin])

    # Create an empty DataFrame with the MultiIndex
    lon_labels = get_lon_labels(size[0])
    lat_labels = get_lat_labels(size[1])
    multi_index = pd.MultiIndex.from_product(
        [lon_labels, lat_labels],
        names=[xbin, ybin],
    )
    empty_df = pd.DataFrame(index=multi_index)

    # Create final dataframe
    df_full = empty_df.join(df, how="left")

    # Reshape to xarray
    ds = df_full.to_xarray()
    return ds


def df_to_dataset(df, xbin, ybin, size, extent):
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expecting a pandas or polars DataFrame.")
    size = _check_size(size)
    ds = pd_df_to_xarray(df, xbin=xbin, ybin=ybin, size=size, extent=extent)
    return ds
