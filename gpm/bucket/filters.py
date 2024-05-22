import numpy as np
import pyproj


def get_geodesic_distance_from_point(lons, lats, lon, lat):
    geod = pyproj.Geod(ellps="WGS84")
    _, _, distance = geod.inv(lons, lats, np.ones(lons.shape) * lon, np.ones(lats.shape) * lat, radians=False)
    return distance


def filter_around_point(df, lon, lat, distance):
    distances = get_geodesic_distance_from_point(lons=df["lon"], lats=df["lat"], lon=lon, lat=lat)
    valid_index = distances < distance
    df["distances"] = distances
    df = df[valid_index]
    return df
