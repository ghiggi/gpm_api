import numpy as np
import pyproj
import xarray as xr
from pyproj import Transformer
from xradar.georeference.projection import get_earth_radius

from gpm.utils.manipulations import (
    conversion_factors_degree_to_meter,
)
from gpm.utils.xarray import get_xarray_variable

# Issues:
# - xyz_to_antenna_coordinates SR GR range estimation. Which to choose?


def reproject(x, y, z=None, **kwargs):
    """
    Transform coordinates from a source projection to a target projection.

    Longitude coordinates should be provided as x, latitude as y.

    Parameters
    ----------
    x : numpy.ndarray
        Array of x coordinates.
    y : numpy.ndarray
        Array of y coordinates.
    z : numpy.ndarray, optional
        Array of z coordinates.

    Keyword Arguments
    -----------------
    src_crs : pyproj.CRS
        Source CRS
    dst_crs : pyproj.CRS
        Destination CRS

    Returns
    -------
    trans : tuple of numpy.ndarray or xr.DataArray
        Arrays of reprojected coordinates (X, Y) or (X, Y, Z) depending on input.
    """
    # Retrieve src and dst CRS
    if "src_crs" not in kwargs:
        raise ValueError("'src_crs' argument not specified !")
    if "dst_crs" not in kwargs:
        raise ValueError("'dst_crs' argument not specified !")
    src_crs = kwargs.get("src_crs")
    dst_crs = kwargs.get("dst_crs")
    # Check if input is xarray
    is_xarray_input = isinstance(x, xr.DataArray) & isinstance(y, xr.DataArray)
    # Transform coordinates
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    result = transformer.transform(x, y, z)
    # Return xarray object if input is xarray
    if is_xarray_input:
        dims = x.dims
        inputs = [x, y, z]
        result = tuple([xr.DataArray(result[i], coords=inputs[i].coords, dims=dims) for i in range(0, len(result))])
        # names = ['x','y','z']
        # ds = xr.Dataset({names[i]: xr.DataArray(result[i], dims=dims) for i in range(0, len(result))})
        # # Add crs coordinates
        # ds = add_crs(ds=ds, crs=kwargs.get("dst_crs"))
        # return ds
    return result


def get_crs_center_latitude(crs):
    """
    Retrieve the center latitude of a pyproj CRS.

    Parameters
    ----------
    crs : pyproj.CRS
        The Coordinate Reference System.

    Returns
    -------
    center_latitude : float
        The center latitude of the CRS.
    """
    crs_info = crs.to_dict()
    center_latitude = crs_info.get("lat_0") or crs_info.get("latitude_of_origin")
    if center_latitude is None:
        raise ValueError("Center latitude parameter not found in the CRS.")
    return center_latitude


def get_gr_crs(latitude, longitude, datum="WGS84"):
    proj_crs = pyproj.CRS(
        proj="aeqd",
        datum=datum,
        lon_0=latitude,
        lat_0=longitude,
    )
    return proj_crs


def xyz_to_antenna_coordinates(x, y, z, site_altitude, crs, effective_radius_fraction=None):
    """Returns the ground radar spherical representation (r, theta, phi).

    Parameters
    ----------
    x : array-like
        Cartesian x coordinates.
    y : array-like
        Cartesian y coordinates.
    z : array-like
        Cartesian z coordinates.
    site_altitude : float
        Radar site elevation (in meters, asl).
    crs : pyproj.CRS
        Azimuthal equidistant projection CRS centered on the ground radar site.
        The latitude of the projection center is used to determine the Earth radius.
    effective_radius_fraction : float
        Adjustment factor to account for the refractivity gradient that
        affects radar beam propagation. In principle this is wavelength-
        dependent. The default of 4/3 is a good approximation for most
        weather radar wavelengths.

    Returns
    -------
    r : array-like
        Array containing the radial distances.
    azimuth : array-like
        Array containing the azimuthal angles.
    elevation: array-like
        Array containing the elevation angles.
    """
    if effective_radius_fraction is None:
        effective_radius_fraction = 4.0 / 3.0

    # Get the latitude of the projection center from the CRS
    lat0 = crs.to_dict().get("lat_0", 0.0)

    # Get the approximate Earth radius at the center of the CRS
    earth_radius = get_earth_radius(latitude=lat0, crs=crs)

    # Calculate the effective earth radius
    effective_earth_radius = earth_radius * effective_radius_fraction

    # Calculate radius to radar site
    sr = effective_earth_radius + site_altitude

    # Calculate radius to radar gates
    zr = effective_earth_radius + z

    # Calculate xy-distance
    s = np.sqrt(x**2 + y**2)

    # Calculate Earth's arc angle
    gamma = s / effective_earth_radius

    # Calculate elevation angle
    numerator = np.cos(gamma) - sr / zr
    denominator = np.sin(gamma)
    elevation = np.arctan(numerator / denominator)

    # Calculate radial distance r
    # - Taken from wradlib xyz_to_spherical:
    r = zr * denominator / np.cos(elevation)

    # Warren et al., 2018 (A10) compute r with the follow equation but is wrong
    # r = np.sqrt(zr**2 + sr **2 - 2 * zr * sr * np.cos(gamma))
    # diff = r - r1
    # np.max(diff) # 200 m

    # Calculate azimuth angle
    azimuth = np.rad2deg(np.arctan2(x, y))
    azimuth = np.fmod(azimuth + 360, 360)

    return r, azimuth, np.rad2deg(elevation)


def antenna_to_cartesian(
    r,
    azimuth,
    elevation,
    crs,
    effective_radius_fraction=None,
    site_altitude=0,
):
    r"""Return Cartesian coordinates from antenna coordinates.

    Parameters
    ----------
    r : array-like
        Distances to the center of the radar gates (bins) in meters.
    azimuth : array-like
        Azimuth angle of the radar in degrees.
    elevation : array-like
        Elevation angle of the radar in degrees.
    crs : pyproj.CRS
        Azimuthal equidistant projection CRS centered on the ground radar site.
        The latitude of the projection center is used to determine the Earth radius.
    effective_radius_fraction: float
        Fraction of earth to use for the effective radius. The default is 4/3.
    site_altitude: float
        Altitude above sea level of the radar site (in meters).

    Returns
    -------
    x, y, z : array-like
        Cartesian coordinates in meters from the radar.

    Notes
    -----
    The calculation for Cartesian coordinate is adapted from equations
    2.28(b) and 2.28(c) of Doviak and Zrnić [1] assuming a
    standard atmosphere (4/3 Earth's radius model).

    .. math::
        :nowrap:

        \\begin{gather*}
        z = \\sqrt{r^2+R^2+2*r*R*sin(\\theta_e)} - R
        \\\\
        s = R * arcsin(\\frac{r*cos(\\theta_e)}{R+z})
        \\\\
        x = s * sin(\\theta_a)
        \\\\
        y = s * cos(\\theta_a)
        \\end{gather*}

    Where r is the distance from the radar to the center of the gate,
    :math:`\\theta_a` is the azimuth angle, :math:`\\theta_e` is the
    elevation angle, s is the arc length, and R is the effective radius
    of the earth, taken to be 4/3 the mean radius of earth (6371 km).

    References
    ----------
    .. [1] Doviak and Zrnić, Doppler Radar and Weather Observations, Second
        Edition, 1993, p. 21.
    """
    if effective_radius_fraction is None:
        effective_radius_fraction = 4.0 / 3.0

    # Retrieve Earth radius at radar site (CRS center)
    lat0 = get_crs_center_latitude(crs)
    earth_radius = get_earth_radius(latitude=lat0, crs=crs)

    # Convert the elevation angle from degrees to radians
    theta_e = np.deg2rad(elevation)
    theta_a = np.deg2rad(azimuth)

    # Compute the effective earth radius
    effective_earth_radius = earth_radius * effective_radius_fraction

    # Calculate radius to radar site
    sr = effective_earth_radius + site_altitude

    # Compute height
    z = np.sqrt(r**2 + sr**2 + 2.0 * r * sr * np.sin(theta_e)) - effective_earth_radius

    # Calculate radius to radar gates
    zr = effective_earth_radius + z

    # Compute arc length in m
    # s = effective_earth_radius * np.arctan(r * np.cos(theta_e)/(r * np.sin(theta_e) + sr))
    s = effective_earth_radius * np.arcsin(r * np.cos(theta_e) / zr)

    # Compute x and y
    x = np.sin(theta_a) * s  # s * np.cos(np.pi/2 - theta_a)
    y = np.cos(theta_a) * s  # s * np.sin(np.pi/2 - theta_a)
    return x, y, z


def retrieve_gates_projection_coordinates(ds, dst_crs):
    """Retrieve the radar gates (x,y,z) coordinates in a custom CRS.

    The longitude and latitude coordinates provided in the GPM products correspond to
    the location where the radar gate centroid cross the ellipsoid.

    This routine returns the coordinates array at each radar gate,
    correcting for the satellite parallax.

    It does not account for the curvature of the Earth in the calculations.

    Parameters
    ----------
    ds : xarray.Dataset
        GPM radar dataset.
    dst_crs: pyproj.CRS
         CRS of target projection (assumed to be defined in meters).

    Returns
    -------
    tuple
        Tuple with the (x,y,z) coordinates in specified target CRS.
    """
    # TODO: Provide single frequency
    # ds.isel(nfreq=freq, missing_dims="ignore")

    # Get x,y, z coordinates in GR CRS
    crs_src = pyproj.CRS.from_epsg(4326)  # ds.gpm.crs
    x_sr, y_sr = reproject(x=ds["lon"], y=ds["lat"], src_crs=crs_src, dst_crs=dst_crs)

    # Retrieve local zenith angle
    alpha = get_xarray_variable(ds, variable="localZenithAngle")

    # Compute range distance from ellipsoid to the radar gate
    range_distance_from_ellipsoid = ds.gpm.retrieve("range_distance_from_ellipsoid")

    # Retrieve height of bin
    z_sr = get_xarray_variable(ds, variable="height")

    # Calculate xy-displacement length
    deltas = np.sin(np.deg2rad(alpha)) * range_distance_from_ellipsoid

    # Calculate x,y differences between ground coordinate and center ground coordinate [25th element]
    idx_center = np.where(ds["gpm_cross_track_id"] == 24)[0]
    xdiff = x_sr - x_sr.isel(cross_track=idx_center).squeeze()
    ydiff = y_sr - y_sr.isel(cross_track=idx_center).squeeze()

    # Calculates the xy-angle of the SR scan
    # - Assume xdiff and ydiff being triangles adjacent and opposite
    ang = np.arctan2(ydiff, xdiff)

    # Calculate displacement dx, dy from displacement length
    deltax = deltas * np.cos(ang)
    deltay = deltas * np.sin(ang)

    # Subtract displacement from SR ground coordinates
    x_srp = x_sr - deltax
    y_srp = y_sr - deltay

    # Return parallax-corrected coordinates
    return x_srp, y_srp, z_sr


def retrieve_gates_geographic_coordinates(ds):
    """Retrieve the radar gates (lat, lon, height) coordinates.

    The longitude and latitude coordinates provided in the GPM products correspond to
    the location where the radar gate centroid cross the ellipsoid.

    This routine returns the coordinates array at each radar gate,
    correcting for the satellite parallax.

    It does not account for the curvature of the Earth in the calculations.

    Requires: scLon, scLat, localZenithAngle, height.

    Parameters
    ----------
    ds : xarray.Dataset
        GPM radar dataset.

    Returns
    -------
    tuple
        Tuple with the (lon, lat, height) coordinates.
    """
    # Retrieve required DataArrays
    x1 = ds["lon"]  # at ellipsoid
    y1 = ds["lat"]  # at ellipsoid
    xs = get_xarray_variable(ds, variable="scLon")  # at satellite
    ys = get_xarray_variable(ds, variable="scLat")  # at satellite
    alpha = get_xarray_variable(ds, variable="localZenithAngle")
    height = get_xarray_variable(ds, variable="height")

    # Compute conversion factors deg to meter
    cx, cy = conversion_factors_degree_to_meter(y1)

    # Convert theta from degrees to radians
    tan_theta_rad = np.tan(np.deg2rad(alpha))

    # Calculate the distance 'dist' using the conversion factors
    dist = np.sqrt((cx * (xs - x1)) ** 2 + (cy * (ys - y1)) ** 2)

    # Calculate the delta components
    scale = height / dist * tan_theta_rad
    delta_x = (xs - x1) * scale
    delta_y = (ys - y1) * scale

    # Calculate the target coordinates (xp, yp)
    lon3d = x1 + delta_x
    lat3d = y1 + delta_y

    # Name the DataArray
    lon3d.name = "lon3d"
    lat3d.name = "lat3d"

    # Return the DataArray
    return lon3d, lat3d, height


def convert_s_to_ku_band(ds_gr, bright_band_height):
    """Convert S-band GR reflectivities to Ku-band.

    Does not account for mixed-phase and hail.
    """
    import wradlib as wrl

    with xr.set_options(keep_attrs=True):
        # Initialize Ku-Band DataArray
        da_s = ds_gr["DBZH"].copy()
        da_ku = np.zeros_like(da_s) * np.nan

        # Apply corrections for snow
        da_above_ml_mask = ds_gr["z"] >= bright_band_height
        da_ku = xr.where(
            da_above_ml_mask,
            wrl.util.calculate_polynomial(da_s.copy(), wrl.trafo.SBandToKu.snow),
            da_ku,
        )
        # Apply corrections for rain
        da_below_ml_mask = ds_gr["z"] < bright_band_height
        da_ku = xr.where(
            da_below_ml_mask,
            wrl.util.calculate_polynomial(da_s.copy(), wrl.trafo.SBandToKu.rain),
            da_ku,
        )
        da_ku.name = "DBZH"
    return da_ku
