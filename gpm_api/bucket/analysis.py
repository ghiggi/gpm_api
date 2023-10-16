#!/usr/bin/env python3
"""
Created on Fri Aug 25 11:18:54 2023

@author: ghiggi
"""

import numpy as np
import pandas as pd
import polars as pl


def get_n_decimals(number):
    number_str = str(number)
    decimal_index = number_str.find(".")

    if decimal_index == -1:
        return 0  # No decimal point found

    # Count the number of characters after the decimal point
    n_decimals = len(number_str) - decimal_index - 1
    return n_decimals


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
    lon_labels = lon_labels.round(n_decimals + 1)
    return lon_labels


def get_lat_labels(bin_spacing):
    n_decimals = get_n_decimals(bin_spacing)
    lat_buckets = get_lat_bins(bin_spacing)
    lat_labels = lat_buckets[0:-1] + bin_spacing / 2
    lat_labels = lat_labels.round(n_decimals + 1)
    return lat_labels


def get_cut_lat_breaks_labels(bin_spacing):
    lat_labels = get_lat_labels(bin_spacing)
    lat_buckets = get_lat_bins(bin_spacing)
    # Define cut labels
    cut_lat_labels = lat_labels.astype(str).tolist()
    cut_lat_labels = ["outside_left"] + cut_lat_labels + ["outside_right"]
    # Deal with left inclusion
    cut_lat_breaks = lat_buckets
    cut_lat_breaks[0] = cut_lat_breaks[0] - 1e-6
    return cut_lat_breaks, cut_lat_labels


def get_cut_lon_breaks_labels(bin_spacing):
    lon_labels = get_lon_labels(bin_spacing)
    lon_buckets = get_lon_bins(bin_spacing)
    cut_lon_labels = lon_labels.astype(str).tolist()
    cut_lon_labels = ["outside_left"] + cut_lon_labels + ["outside_right"]
    # - Deal with left inclusion
    cut_lon_breaks = lon_buckets
    cut_lon_breaks[0] = cut_lon_breaks[0] - 1e-6
    return cut_lon_breaks, cut_lon_labels


def pl_add_geographic_bins(
    df, xbin_column, ybin_column, bin_spacing, x_column="lon", y_column="lat"
):
    cut_lon_breaks, cut_lon_labels = get_cut_lon_breaks_labels(bin_spacing)
    cut_lat_breaks, cut_lat_labels = get_cut_lat_breaks_labels(bin_spacing)
    df = df.with_columns(
        pl.col(x_column).cut(cut_lon_breaks, labels=cut_lon_labels).alias(xbin_column),
        pl.col(y_column).cut(cut_lat_breaks, labels=cut_lat_labels).alias(ybin_column),
    )
    # df.filter(pl.col(xbin_column) == "outside_left")
    # df.filter(pl.col(xbin_column) == "outside_right")
    return df


def pl_df_to_xarray(df, xbin_column, ybin_column, bin_spacing):
    df_stats_pd = df.to_pandas()
    df_stats_pd.dtypes

    df_stats_pd[xbin_column] = df_stats_pd[xbin_column].astype(float)
    df_stats_pd[ybin_column] = df_stats_pd[ybin_column].astype(float)
    df_stats_pd = df_stats_pd.set_index([xbin_column, ybin_column])

    ## Left join to the spatial bin template dataframe
    # Create the MultiIndex from the combination of x and y bins
    lon_labels = get_lon_labels(bin_spacing)
    lat_labels = get_lat_labels(bin_spacing)
    multi_index = pd.MultiIndex.from_product(
        [lon_labels, lat_labels], names=[xbin_column, ybin_column]
    )

    # Create an empty DataFrame with the MultiIndex
    empty_df = pd.DataFrame(index=multi_index)

    empty_df

    # Create final dataframe
    df_stats_pd = empty_df.join(df_stats_pd, how="left")

    # Reshape to xarray
    ds = df_stats_pd.to_xarray()
    ds = ds.rename({xbin_column: "longitude", ybin_column: "latitude"})

    return ds
