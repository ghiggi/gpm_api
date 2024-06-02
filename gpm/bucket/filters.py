import numpy as np
import polars as pl
import pyproj

from gpm.bucket.dataframe import (
    df_add_column,
    df_get_column,
    df_select_valid_rows,
)


def get_geodesic_distance_from_point(lons, lats, lon, lat):
    lons = np.asanyarray(lons)
    lats = np.asanyarray(lats)
    geod = pyproj.Geod(ellps="WGS84")
    _, _, distance = geod.inv(lons, lats, np.ones(lons.shape) * lon, np.ones(lats.shape) * lat, radians=False)
    return distance


def filter_around_point(df, lon, lat, distance):
    # https://stackoverflow.com/questions/76262681/i-need-to-create-a-column-with-the-distance-between-two-coordinates-in-polars
    # Retrieve coordinates
    lons = df_get_column(df, column="lon")
    lats = df_get_column(df, column="lat")
    # Compute geodesic distance
    distances = get_geodesic_distance_from_point(lons=lons, lats=lats, lon=lon, lat=lat)
    valid_indices = distances <= distance
    # Add distance
    df = df_add_column(df, column="distance", values=distances)
    # Select only valid rows
    df = df_select_valid_rows(df, valid_rows=valid_indices)
    return df


def filter_by_extent(df, extent, x="lon", y="lat"):
    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        df = df.filter(
            pl.col(x) >= extent[0],
            pl.col(x) <= extent[1],
            pl.col(y) >= extent[2],
            pl.col(y) <= extent[3],
        )
    else:  # pandas
        idx_valid = (df[x] >= extent[0]) & (df[x] <= extent[1]) & (df[y] >= extent[2]) & (df[y] <= extent[3])
        df = df.loc[idx_valid]
    return df


def apply_spatial_filters(df, filters=None):
    if filters is None:
        filters = {}
    if "extent" in filters:
        df = filter_by_extent(df, extent=filters["extent"], x="lon", y="lat")
    if "point_radius" in filters:
        lon, lat, distance = filters["point_radius"]
        df = filter_around_point(df, lon=lon, lat=lat, distance=distance)
    return df
