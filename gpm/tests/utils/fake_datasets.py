import numpy as np
import pyproj
import xarray as xr


def get_geodesic_path(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    n_points: int,
    offset_distance: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute geodesic path between starting and ending coordinates."""
    geod = pyproj.Geod(ellps="sphere")
    r = geod.inv_intermediate(
        start_lon,
        start_lat,
        end_lon,
        end_lat,
        n_points,
        initial_idx=0,
        terminus_idx=0,
        return_back_azimuth=False,
        flags=pyproj.enums.GeodIntermediateFlag.AZIS_KEEP,
    )

    orthogonal_directions = np.array(r.azis) + 90

    if offset_distance != 0:
        geod.fwd(r.lons, r.lats, orthogonal_directions, [offset_distance] * n_points, inplace=True)

    # Convert into numpy arrays
    lon = np.array(r.lons)
    lat = np.array(r.lats)

    return lon, lat


def get_geodesic_band(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    width: float,
    n_along_track: int,
    n_cross_track: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute coordinates of geodesic band."""
    lon_lines = []
    lat_lines = []
    offsets = np.linspace(-width / 2, width / 2, n_cross_track)

    for offset in offsets:
        lon_line, lat_line = get_geodesic_path(
            start_lon=start_lon,
            start_lat=start_lat,
            end_lon=end_lon,
            end_lat=end_lat,
            n_points=n_along_track,
            offset_distance=offset,
        )
        lon_lines.append(lon_line)
        lat_lines.append(lat_line)

    lon = np.stack(lon_lines)
    lat = np.stack(lat_lines)

    return lon, lat


def get_orbit_dataarray(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    width: float,
    n_along_track: int,
    n_cross_track: int,
) -> xr.DataArray:
    """Create orbit data array on geodesic band."""
    np.random.seed(0)
    cross_track = np.arange(n_cross_track)
    along_track = np.arange(n_along_track)
    data = np.random.rand(n_cross_track, n_along_track)
    granule_id = np.zeros(n_along_track)

    # Coordinates
    lon, lat = get_geodesic_band(
        start_lon=start_lon,
        start_lat=start_lat,
        end_lon=end_lon,
        end_lat=end_lat,
        width=width,
        n_along_track=n_along_track,
        n_cross_track=n_cross_track,
    )

    # Create data array
    da = xr.DataArray(data, coords={"cross_track": cross_track, "along_track": along_track})
    da.coords["lat"] = (("cross_track", "along_track"), lat)
    da.coords["lon"] = (("cross_track", "along_track"), lon)
    da.coords["gpm_granule_id"] = ("along_track", granule_id)

    return da


def get_grid_dataarray(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    n_lon: int,
    n_lat: int,
) -> xr.DataArray:
    """Create grid data array."""
    np.random.seed(0)
    lon = np.linspace(start_lon, end_lon, n_lon)
    lat = np.linspace(start_lat, end_lat, n_lat)
    data = np.random.rand(n_lat, n_lon)

    # Create data array
    return xr.DataArray(data, coords={"lat": lat, "lon": lon})
