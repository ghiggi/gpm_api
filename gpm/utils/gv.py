import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr

from gpm.utils.manipulations import (
    conversion_factors_degree_to_meter,
)
from gpm.utils.remapping import reproject_coords
from gpm.utils.timing import print_task_elapsed_time
from gpm.utils.xarray import get_xarray_variable

# Issues:
# - xyz_to_antenna_coordinates SR GR range estimation. Which to choose ?


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
    from xradar.georeference.projection import get_earth_radius

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
    from xradar.georeference.projection import get_earth_radius

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
    x_sr, y_sr, _ = reproject_coords(x=ds["lon"], y=ds["lat"], src_crs=crs_src, dst_crs=dst_crs)

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


def plot_quicklook(ds_sr, ds_gr, min_gr_range, max_gr_range, z_min_threshold_gr, z_min_threshold_sr, cmap="Spectral_r"):

    # Display nearSurface SR reflectivity
    da_sr = ds_sr["zFactorFinal"].gpm.slice_range_at_bin(ds_sr["binClutterFreeBottom"])

    # Mask by specified sensitivity
    mask_gr = ds_gr["DBZH"] > z_min_threshold_gr
    mask_sr = da_sr > z_min_threshold_sr

    # Retrieve SR extent
    sr_extent = ds_sr.gpm.extent()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=120, subplot_kw={"projection": ccrs.PlateCarree()})

    # Plot SR
    da_sr.where(mask_sr).gpm.plot_map(
        ax=ax1,
        cmap=cmap,
        vmin=0,
        vmax=40,
        cbar_kwargs={"label": "SR Surface Reflectivity (dBz)"},
        add_colorbar=True,
    )
    # Plot GR
    ds_gr["DBZH"].where(mask_gr).xradar_dev.plot_map(
        ax=ax2,
        cmap=cmap,
        vmin=0,
        vmax=40,
        cbar_kwargs={"label": "GR Sweep Reflectivity (dBz)", "extend": "both"},
        add_colorbar=True,
    )
    # - Add the SR swath boundary
    da_sr.gpm.plot_swath_lines(ax=ax2)
    # - Restrict the extent to the SR overpass
    ax2.set_extent(sr_extent)
    # - Display GR range distances
    for ax in [ax1, ax2]:
        ds_gr.xradar_dev.plot_range_distance(
            distance=min_gr_range,
            ax=ax,
            add_background=True,
            linestyle="dashed",
            edgecolor="black",
        )
        ds_gr.xradar_dev.plot_range_distance(
            distance=max_gr_range,
            ax=ax,
            add_background=True,
            linestyle="dashed",
            edgecolor="black",
        )
        ds_gr.xradar_dev.plot_range_distance(
            distance=250_000,
            ax=ax,
            add_background=True,
            linestyle="dashed",
            edgecolor="black",
        )
    # - Add GR location
    ax1.scatter(ds_gr["longitude"], ds_gr["latitude"], c="black", marker="X", s=4)
    ax2.scatter(ds_gr["longitude"], ds_gr["latitude"], c="black", marker="X", s=4)
    # - Set title
    ax1.set_title("GPM", fontsize=12, loc="left")
    ax2.set_title("Ground Radar", fontsize=12, loc="left")
    # - Improve layout and display
    plt.tight_layout()
    return fig


@print_task_elapsed_time(prefix="SR/GR Matching")
def volume_matching(
    ds_gr,
    radar_band,
    beamwidth_gr: float = 1.0,
    z_min_threshold_gr=0,
    z_min_threshold_sr=10,
    min_gr_range=0,
    max_gr_range=150_000,
    gr_sensitivity_thresholds=None,
    sr_sensitivity_thresholds=None,
    download_sr=True,
    display_quicklook=True,
):
    """
    Performs the volume matching of GPM Spaceborne Radar (SR) data to Ground Radar (GR).

    Parameters
    ----------
    ds_gr : xr.Dataset
        Xradar Dataset corresponding to a GR sweep.
    radar_band : str
        The GR band. Valid values are "Ku", "S", "C", "X".
    beamwidth_gr : float, optional
        The GR beamwidth. The default is 1.0.
    z_min_threshold_gr : float, optional
        The minimum reflectivity threshold (in dBZ) for GR. The default is 0.
        Values below this threshold are set to np.nan before matching SR/GR gates.
    z_min_threshold_sr : float, optional
        The minimum reflectivity threshold (in dBZ) for SR. The default is 10.
        Values below this threshold are set to np.nan before matching SR/GR gates.
    min_gr_range : float, optional
        The minimum GR range distance (in meters) to select for matching SR/GR gates. The default is 0.
    max_gr_range : float, optional
        The maximum GR range distance (in meters) to select for matching SR/GR gates. The default is 150_000.
    gr_sensitivity_thresholds : list, optional
        The GR sensitivity thresholds to verify NUBF. The default is [6,8,10,12].
    sr_sensitivity_thresholds : list, optional
        The SR sensitivity thresholds to verify NUBF. The default is [10,12,14,16,18].
    download_sr : bool, optional
        Whether to attempt download SR data on-the-fly. The default is True.


    -------
    gdf_match : geopandas.DataFrame
        Dataframe containing the matched SR/GR reflectivities and relevant aggregation statistics.
    ds_sr : xr.Dataset
        The SR dataset matched to GR data.

    """
    import warnings

    warnings.filterwarnings("ignore")
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import pyproj
    import wradlib as wrl
    import xarray as xr
    from shapely import Point

    import gpm
    from gpm.utils.remapping import reproject_coords
    from gpm.utils.zonal_stats import PolyAggregator

    # Define GR beamwidth
    elevation_beamwidth_gr = beamwidth_gr
    azimuth_beamwidth_gr = beamwidth_gr

    # Check radar band
    radar_band = radar_band.upper()
    accepted_radar_bands = ["S", "C", "X", "Ku"]
    if radar_band not in accepted_radar_bands:
        raise ValueError(f"Accepted 'radar_band' are {accepted_radar_bands}.")

    # Define SR and GR sensitivity_thresholds
    if gr_sensitivity_thresholds is None:
        gr_sensitivity_thresholds = [6, 8, 10, 12]
    if sr_sensitivity_thresholds is None:
        sr_sensitivity_thresholds = [10, 12, 14, 16, 18]

    # Define SR CRS
    crs_sr = pyproj.CRS.from_epsg(4326)
    # TODO: derive same from ds_sr.gpm.pyproj_crs

    ####-----------------------------------------------------------------------------.
    #### Preprocess GR
    #### - Set auxiliary info as coordinates
    # - To avoid broadcasting i.e. when masking
    possible_coords = ["sweep_mode", "sweep_number", "prt_mode", "follow_mode", "sweep_fixed_angle"]
    extra_coords = [coord for coord in possible_coords if coord in ds_gr]
    ds_gr = ds_gr.set_coords(extra_coords)

    #### - Georeference the data on a azimuthal_equidistant projection centered on the radar
    ds_gr = ds_gr.xradar.georeference()

    #### - Get the GR CRS
    crs_gr = ds_gr.xradar_dev.pyproj_crs

    #### - Retrieve GR (lon, lat) coordinates
    lon_gr, lat_gr, height_gr = reproject_coords(
        x=ds_gr["x"],
        y=ds_gr["y"],
        z=ds_gr["z"],
        src_crs=crs_gr,
        dst_crs=crs_sr,
    )
    #### - Add lon/lat coordinates to ds_gr
    ds_gr["lon"] = lon_gr
    ds_gr["lat"] = lat_gr
    ds_gr = ds_gr.set_coords(["lon", "lat"])

    #### - Set GR gates with Z < 0 to NaN
    # - Following Morris and Schwaller 2011 recommendation
    ds_gr = ds_gr.where(ds_gr["DBZH"] >= z_min_threshold_sr)

    #### - Retrieve GR extent (in WGS84)
    extent_gr = ds_gr.xradar_dev.extent(max_distance=None, crs=None)

    ####-----------------------------------------------------------------------------.
    #### Retrieve SR data

    # Retrieve GR scan time (assumed to be in UTC)
    gr_min_time = ds_gr["time"].min().to_numpy()
    gr_max_time = ds_gr["time"].max().to_numpy()

    # Define start_time and end_time for searching SR data
    start_time = gr_min_time - np.timedelta64(5, "m")
    end_time = gr_max_time + np.timedelta64(5, "m")

    # Define GPM settings
    product_type = "RS"
    version = 7
    storage = "GES_DISC"

    # Define products
    # - Use TRMM PR before '2014-03-08'
    # - Use GPM DPR after '2014-03-08'
    if start_time >= np.array("2014-03-08 22:09:50", dtype="M8[s]"):
        products = ["2A-DPR", "1B-Ku"]
    else:
        products = ["2A-PR", "1B-PR"]

    # Download SR data (if asked)
    if download_sr:
        for product in products:
            gpm.download(
                product=product,
                product_type=product_type,
                version=version,
                storage=storage,
                start_time=start_time,
                end_time=end_time,
            )

    # Open datasets
    # - TODO: subset variables used !
    ds_sr = gpm.open_dataset(
        product=products[0],
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
        chunks={},  # only read the data needed around GR instead of full granule
    )

    ds_l1b_sr = gpm.open_dataset(
        product=products[1],
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
        chunks={},  # only read the data needed around GR instead of full granule
    )

    # Crop SR datasets to region of interest
    ds_sr = ds_sr.gpm.crop(extent=extent_gr)
    ds_l1b_sr = ds_l1b_sr.gpm.crop(extent=extent_gr)

    # Put SR variables in memory
    ds_sr = ds_sr.compute()
    ds_l1b_sr = ds_l1b_sr.compute()

    # Retrieve important info from L1B
    ds_l1b_sr = ds_l1b_sr.compute()
    beam_width_sr = ds_l1b_sr["crossTrackBeamWidth"]  # alongTrackBeamWidth

    # Select only Ku-band
    if "radar_frequency" in ds_sr.dims:
        ds_sr = ds_sr.sel(radar_frequency="Ku")

    # Compute SR attenuation correction
    ds_sr["zFactorCorrection"] = ds_sr["zFactorFinal"] - ds_sr["zFactorMeasured"]

    ####-----------------------------------------------------------------------------.
    #### Plot Quicklook
    if display_quicklook:
        plot_quicklook(
            ds_sr=ds_sr,
            ds_gr=ds_gr,
            min_gr_range=min_gr_range,
            max_gr_range=max_gr_range,
            z_min_threshold_gr=z_min_threshold_gr,
            z_min_threshold_sr=z_min_threshold_sr,
        )
        plt.show()
    ####-----------------------------------------------------------------------------.
    #### Retrieve SR/GR gate resolution, volume and coordinates
    #### - Retrieve GR gate resolution
    h_res_gr, v_res_gr = ds_gr.xradar_dev.resolution_at_range(
        azimuth_beamwidth=azimuth_beamwidth_gr,
        elevation_beamwidth=elevation_beamwidth_gr,
    )

    #### - Retrieve GR (lon, lat) coordinates
    lon_gr, lat_gr, height_gr = reproject_coords(
        x=ds_gr["x"],
        y=ds_gr["y"],
        z=ds_gr["z"],
        src_crs=crs_gr,
        dst_crs=crs_sr,
    )

    #### - Retrieve SR (x,y,z) coordinates
    x_sr, y_sr, z_sr = retrieve_gates_projection_coordinates(ds_sr, dst_crs=crs_gr)

    #### - Retrieve SR (range, azimuth, elevation) coordinates
    range_sr, azimuth_sr, elevation_sr = xyz_to_antenna_coordinates(
        x=x_sr,
        y=y_sr,
        z=z_sr,
        site_altitude=ds_gr["altitude"],
        crs=crs_gr,
        effective_radius_fraction=None,
    )

    #### - Retrieve SR range distance
    ds_l1b_sr["range_distance_from_satellite"] = ds_l1b_sr.gpm.retrieve("range_distance_from_satellite")
    ds_sr["range_distance_from_satellite"] = ds_l1b_sr.gpm.extract_l2_dataset()["range_distance_from_satellite"]

    #### - Retrieve SR gates volumes
    vol_sr = ds_sr.gpm.retrieve(
        "gate_volume",
        beam_width=beam_width_sr,
        range_distance=ds_sr["range_distance_from_satellite"],
    )

    #### - Retrieve GR gates volumes
    vol_gr = wrl.qual.pulse_volume(
        ds_gr["range"],  # range distance
        h=ds_gr["range"].diff("range").median(),  # range resolution
        theta=elevation_beamwidth_gr,
    )  # beam width
    vol_gr = vol_gr.broadcast_like(ds_gr["DBZH"])
    vol_gr = vol_gr.assign_coords({"lon": ds_gr["lon"], "lat": ds_gr["lat"]})

    #### - Retrieve SR gate resolution
    h_res_sr, v_res_sr = ds_sr.gpm.retrieve(
        "gate_resolution",
        beam_width=beam_width_sr,
        range_distance=ds_sr["range_distance_from_satellite"],
    )

    ####-----------------------------------------------------------------------------.
    #### Retrieve custom SR variables
    #### - Retrieve Bright Band (BB) Ratio
    da_bb_ratio, da_bb_mask = ds_sr.gpm.retrieve("bright_band_ratio", return_bb_mask=True)

    #### - Retrieve Precipitation and Hydrometeors Types
    ds_sr["flagPrecipitationType"] = ds_sr.gpm.retrieve("flagPrecipitationType", method="major_rain_type")
    ds_sr["flagHydroClass"] = ds_sr.gpm.retrieve("flagHydroClass")

    ####-----------------------------------------------------------------------------.
    #### Identify SR gates intersecting GR sweep
    # - Define mask of SR footprints within GR range
    r = np.sqrt(x_sr**2 + y_sr**2)
    # mask_sr_within_gr_range = r < ds_gr["range"].max().item()
    mask_sr_within_gr_range_interval = (r >= min_gr_range) & (r <= max_gr_range)

    # - Define mask of SR footprints within GR elevation beamwidth
    mask_sr_matched_elevation = (elevation_sr >= (ds_gr["sweep_fixed_angle"] - elevation_beamwidth_gr / 2.0)) & (
        elevation_sr <= (ds_gr["sweep_fixed_angle"] + elevation_beamwidth_gr / 2.0)
    )

    # - Define mask of SR footprints matching GR gates
    mask_ppi = mask_sr_matched_elevation & mask_sr_within_gr_range_interval

    ####-----------------------------------------------------------------------------.
    #### Define SR mask
    # Select above minimum SR reflectivity
    # mask_sr_minimum_z = ds_sr["zFactorFinal"] > z_sr_min_threshold

    # Select only 'high quality' data
    # - qualityFlag derived from qualityData.
    # - qualityFlag == 1 indicates low quality retrievals
    # - qualityFlag == 2 indicates bad/missing retrievals
    # mask_sr_quality_data = ds_sr["qualityFlag"] == 0

    # # Select only beams with detected bright band
    # # mask_sr_quality_bb = ds_sr["qualityBB"] == 1
    # mask_sr_quality_precip = ds_sr["qualityTypePrecip"] == 1

    # Select scan with "normal" dataQuality (for entire cross-track scan)
    mask_sr_quality = ds_sr["dataQuality"] == 0

    # Select beams with detected precipitation
    mask_sr_precip = ds_sr["flagPrecip"] > 0

    # Define 3D mask of SR gates matching GR PPI gates
    # - TIP: do not mask eccessively here ... but rather at final stage
    mask_matched_ppi_3d = mask_ppi & mask_sr_precip & mask_sr_quality

    # Define 2D mask of SR beams matching the GR PPI
    mask_matched_ppi_2d = mask_matched_ppi_3d.any(dim="range")

    ####-----------------------------------------------------------------------------.
    #### Convert reflectivities to target band
    # Define retrievals to be used (as function of radar_band)
    dict_retrieval_names = {
        "S": "s_band_cao2013",
        "C": "c_band_tan",
        "X": "x_band_tan",
    }
    # Derive reflectivity of interest
    if radar_band in ["X", "C", "S"]:
        retrieval_name = dict_retrieval_names[radar_band]
        # With attenuation corrected reflectivity
        da_z_final = ds_sr.gpm.retrieve(
            retrieval_name,
            reflectivity="zFactorFinal",
            bb_ratio=da_bb_ratio,
            precip_type=ds_sr["flagPrecipitationType"],
        )
        # With measured reflectivity
        da_z_measured = ds_sr.gpm.retrieve(
            retrieval_name,
            reflectivity="zFactorMeasured",
            bb_ratio=da_bb_ratio,
            precip_type=ds_sr["flagPrecipitationType"],
        )
        # Compute attenuation correction in S-band
        da_z_correction = da_z_final - da_z_measured
    else:
        da_z_final = ds_sr["zFactorFinal"]
        da_z_measured = ds_sr["zFactorMeasured"]
        da_z_correction = da_z_final - da_z_measured

    ####-----------------------------------------------------------------------------.
    #### Aggregate SR radar gates
    # Add variables to SR dataset
    ds_sr["zFactorMeasured_Ku"] = ds_sr["zFactorMeasured"]
    ds_sr["zFactorFinal_Ku"] = ds_sr["zFactorFinal"]
    ds_sr[f"zFactorFinal_{radar_band}"] = da_z_final
    ds_sr[f"zFactorMeasured_{radar_band}"] = da_z_measured
    ds_sr["zFactorCorrection_Ku"] = ds_sr["zFactorCorrection"]
    ds_sr[f"zFactorCorrection_{radar_band}"] = da_z_correction
    ds_sr["hres"] = h_res_sr
    ds_sr["vres"] = v_res_sr
    ds_sr["gate_volume"] = vol_sr
    ds_sr["x"] = x_sr
    ds_sr["y"] = y_sr
    ds_sr["z"] = z_sr

    # Compute path-integrated reflectivities
    ds_sr["zFactorFinal_Ku_cumsum"] = ds_sr["zFactorFinal"].gpm.idecibel.cumsum("range").gpm.decibel
    ds_sr["zFactorFinal_Ku_cumsum"] = ds_sr["zFactorFinal_Ku_cumsum"].where(
        np.isfinite(ds_sr["zFactorFinal_Ku_cumsum"]),
    )
    ds_sr["zFactorMeasured_Ku_cumsum"] = ds_sr["zFactorMeasured"].gpm.idecibel.cumsum("range").gpm.decibel
    ds_sr["zFactorMeasured_Ku_cumsum"] = ds_sr["zFactorMeasured_Ku_cumsum"].where(
        np.isfinite(ds_sr["zFactorMeasured_Ku_cumsum"]),
    )
    ds_sr[f"zFactorFinal_{radar_band}_cumsum"] = (
        ds_sr[f"zFactorFinal_{radar_band}"].gpm.idecibel.cumsum("range").gpm.decibel
    )
    ds_sr[f"zFactorFinal_{radar_band}_cumsum"] = ds_sr[f"zFactorFinal_{radar_band}_cumsum"].where(
        np.isfinite(ds_sr[f"zFactorFinal_{radar_band}_cumsum"]),
    )
    ds_sr[f"zFactorMeasured_{radar_band}_cumsum"] = (
        ds_sr[f"zFactorMeasured_{radar_band}"].gpm.idecibel.cumsum("range").gpm.decibel
    )
    ds_sr[f"zFactorMeasured_{radar_band}_cumsum"] = ds_sr[f"zFactorMeasured_{radar_band}_cumsum"].where(
        np.isfinite(ds_sr[f"zFactorMeasured_{radar_band}_cumsum"]),
    )

    # Add variables to the spaceborne dataset
    z_variables = [
        "zFactorFinal_Ku",
        f"zFactorFinal_{radar_band}",
        "zFactorMeasured_Ku",
        f"zFactorMeasured_{radar_band}",
    ]
    sr_variables = [
        *z_variables,
        "zFactorCorrection_Ku",
        f"zFactorCorrection_{radar_band}",
        "precipRate",
        "airTemperature",
        "zFactorFinal_Ku_cumsum",
        "zFactorMeasured_Ku_cumsum",
        f"zFactorFinal_{radar_band}_cumsum",
        f"zFactorMeasured_{radar_band}_cumsum",
        "hres",
        "vres",
        "gate_volume",
        "x",
        "y",
        "z",
    ]

    # Initialize Dataset where to add aggregated SR gates
    ds_sr_match_ppi = xr.Dataset()

    # Mask SR gates not matching the GR PPI
    ds_sr_ppi = ds_sr[sr_variables].where(mask_matched_ppi_3d)

    # Compute aggregation statistics
    ds_sr_ppi_min = ds_sr_ppi.min("range")
    ds_sr_ppi_max = ds_sr_ppi.max("range")
    ds_sr_ppi_sum = ds_sr_ppi.sum(dim="range", skipna=True)
    ds_sr_ppi_mean = ds_sr_ppi.mean("range")
    ds_sr_ppi_std = ds_sr_ppi.std("range")

    # Aggregate reflectivities (in mm6/mm3)
    for var in z_variables:
        ds_sr_ppi_mean[var] = ds_sr_ppi[var].gpm.idecibel.mean("range").gpm.decibel
        ds_sr_ppi_std[var] = ds_sr_ppi[var].std("range")  # if only 1 value --> with log, std=0 would become -inf

    # Compute counts and fractions above sensitivity thresholds
    ds_sr_match_ppi["SR_counts"] = mask_matched_ppi_3d.sum(dim="range")
    ds_sr_match_ppi["SR_counts_valid"] = (~np.isnan(ds_sr_ppi["zFactorFinal_Ku"])).sum(dim="range")
    for var in z_variables:
        for thr in sr_sensitivity_thresholds:
            fraction = (ds_sr_ppi[var] >= thr).sum(dim="range") / ds_sr_match_ppi["SR_counts"]
            ds_sr_match_ppi[f"SR_{var}_fraction_above_{thr}dBZ"] = fraction

    # Compute fraction of hydrometeor types
    da_hydro_class = ds_sr["flagHydroClass"].where(mask_matched_ppi_3d)
    ds_sr_match_ppi["SR_fraction_no_precip"] = (da_hydro_class == 0).sum(dim="range") / ds_sr_match_ppi["SR_counts"]
    ds_sr_match_ppi["SR_fraction_rain"] = (da_hydro_class == 1).sum(dim="range") / ds_sr_match_ppi["SR_counts"]
    ds_sr_match_ppi["SR_fraction_snow"] = (da_hydro_class == 2).sum(dim="range") / ds_sr_match_ppi["SR_counts"]
    ds_sr_match_ppi["SR_fraction_hail"] = (da_hydro_class == 3).sum(dim="range") / ds_sr_match_ppi["SR_counts"]
    ds_sr_match_ppi["SR_fraction_melting_layer"] = (da_hydro_class == 4).sum(dim="range") / ds_sr_match_ppi["SR_counts"]
    ds_sr_match_ppi["SR_fraction_clutter"] = (da_hydro_class == 5).sum(dim="range") / ds_sr_match_ppi["SR_counts"]
    ds_sr_match_ppi["SR_fraction_below_isotherm"] = (ds_sr_ppi["airTemperature"] >= 273.15).sum(
        dim="range",
    ) / ds_sr_match_ppi["SR_counts"]
    ds_sr_match_ppi["SR_fraction_above_isotherm"] = (ds_sr_ppi["airTemperature"] < 273.15).sum(
        dim="range",
    ) / ds_sr_match_ppi["SR_counts"]

    # Add aggregation statistics
    for var in ds_sr_ppi_mean.data_vars:
        ds_sr_match_ppi[f"SR_{var}_mean"] = ds_sr_ppi_mean[var]
    for var in ds_sr_ppi_std.data_vars:
        ds_sr_match_ppi[f"SR_{var}_std"] = ds_sr_ppi_std[var]
    for var in ds_sr_ppi_sum.data_vars:
        ds_sr_match_ppi[f"SR_{var}_sum"] = ds_sr_ppi_sum[var]
    for var in ds_sr_ppi_max.data_vars:
        ds_sr_match_ppi[f"SR_{var}_max"] = ds_sr_ppi_max[var]
    for var in ds_sr_ppi_min.data_vars:
        ds_sr_match_ppi[f"SR_{var}_min"] = ds_sr_ppi_min[var]
    for var in ds_sr_ppi_min.data_vars:
        ds_sr_match_ppi[f"SR_{var}_range"] = ds_sr_ppi_max[var] - ds_sr_ppi_min[var]

    # Compute coefficient of variation
    for var in z_variables:
        ds_sr_match_ppi[f"SR_{var}_cov"] = ds_sr_match_ppi[f"SR_{var}_std"] / ds_sr_match_ppi[f"SR_{var}_mean"]

    # Add SR L2 variables (useful for final filtering and analysis)
    var_l2 = [
        "flagPrecip",
        "flagPrecipitationType",
        "dataQuality",
        "sunLocalTime",
        "SCorientation",
        "qualityFlag",
        "qualityTypePrecip",
        "qualityBB",
        "pathAtten",
        "piaFinal",
        "reliabFlag",
    ]
    for var in var_l2:
        ds_sr_match_ppi[f"SR_{var}"] = ds_sr[var]

    # Add SR time
    ds_sr_match_ppi["SR_time"] = ds_sr_match_ppi["time"]
    ds_sr_match_ppi = ds_sr_match_ppi.drop("time")

    # Mask SR beams not matching the GR PPI
    ds_sr_match_ppi = ds_sr_match_ppi.where(mask_matched_ppi_2d)

    # Remove unnecessary coordinates
    unecessary_coords = [
        "radar_frequency",
        "sweep_mode",
        "prt_mode",
        "follow_mode",
        "latitude",
        "longitude",
        "altitude",
        "crs_wkt",
        "time",
        "crsWGS84",
        "dataQuality",
        "sunLocalTime",
        "SCorientation",
    ]
    for coord in unecessary_coords:
        if coord in ds_sr_match_ppi:
            ds_sr_match_ppi = ds_sr_match_ppi.drop(coord)
        if coord in mask_matched_ppi_2d.coords:
            mask_matched_ppi_2d = mask_matched_ppi_2d.drop(coord)

    # Stack aggregated dataset to beam dimension index
    ds_sr_stack = ds_sr_match_ppi.stack(sr_beam_index=("along_track", "cross_track"))  # noqa
    da_mask_matched_ppi_stack = mask_matched_ppi_2d.stack(sr_beam_index=("along_track", "cross_track"))  # noqa

    # Drop beams not matching the GR PPI
    ds_sr_match = ds_sr_stack.isel(sr_beam_index=da_mask_matched_ppi_stack)

    ####-----------------------------------------------------------------------------.
    #### Retrieve the SR footprints polygons
    # Retrieve SR footprint polygons (using the footprint radius in AEQD x,y coordinates)
    xy_mean_sr = np.stack([ds_sr_match["SR_x_mean"], ds_sr_match["SR_y_mean"]], axis=-1)
    footprint_radius = ds_sr_match["SR_hres_max"].to_numpy() / 2
    sr_poly = [Point(x, y).buffer(footprint_radius[i]) for i, (x, y) in enumerate(xy_mean_sr)]

    # Create geopandas DataFrame
    df_sr_match = ds_sr_match.to_dataframe()
    gdf_sr = gpd.GeoDataFrame(df_sr_match, crs=crs_gr, geometry=sr_poly)

    # Extract radar gate polygon on the range-azimuth axis
    sr_poly = np.array(gdf_sr.geometry)

    ####-----------------------------------------------------------------------------.
    #### Retrieve the GR gates polygons
    # Add variables to GR dataset
    ds_gr["gate_volume"] = vol_gr
    ds_gr["vres"] = v_res_gr
    ds_gr["hres"] = h_res_gr

    # Add path_integrated_reflectivities
    ds_gr["DBZH_cumsum"] = ds_gr["DBZH"].gpm.idecibel.cumsum("range").gpm.decibel
    ds_gr["DBZH_cumsum"] = ds_gr["DBZH_cumsum"].where(np.isfinite(ds_gr["DBZH_cumsum"]))

    # Mask reflectivites above minimum GR Z threshold
    mask_gr = ds_gr["DBZH"] > z_min_threshold_gr  # important !

    # Subset gates in range distance interval
    ds_gr_subset = ds_gr.sel(range=slice(min_gr_range, max_gr_range)).where(mask_gr)

    # Retrieve geopandas dataframe
    gdf_gr = ds_gr_subset.xradar_dev.to_geopandas()  # here we currently infer the quadmesh using the x,y coordinates

    # Extract radar gate polygon on the range-azimuth axis
    gr_poly = np.array(gdf_gr.geometry)

    ####-----------------------------------------------------------------------------.
    #### Aggregate GR data on SR footprints
    # Define PolyAggregator
    aggregator = PolyAggregator(source_polygons=gr_poly, target_polygons=sr_poly)

    # Aggregate GR reflecitvities and compute statistics
    # - Timestep of acquisition
    time_gr = aggregator.first(values=gdf_gr["time"])

    # - Total number of gates aggregated
    counts = aggregator.counts()
    counts_valid = aggregator.apply(lambda x, weights: np.sum(~np.isnan(x)), values=gdf_gr["DBZH"])  # noqa

    # - Total gate volume
    sum_vol = aggregator.sum(values=gdf_gr["gate_volume"])

    # - Fraction of SR area covered
    fraction_covered_area = aggregator.fraction_covered_area()

    # - Reflectivity statistics
    z_mean = wrl.trafo.decibel(aggregator.average(values=wrl.trafo.idecibel(gdf_gr["DBZH"])))
    z_std = aggregator.std(values=gdf_gr["DBZH"])
    z_max = aggregator.max(values=gdf_gr["DBZH"])
    z_min = aggregator.min(values=gdf_gr["DBZH"])
    z_range = z_max - z_min
    z_cov = z_std / z_mean  # coefficient of variation

    # Create DataFrame with GR matched statistics
    df = pd.DataFrame(
        {
            "GR_Z_mean": z_mean,
            "GR_Z_std": z_std,
            "GR_Z_max": z_max,
            "GR_Z_min": z_min,
            "GR_Z_range": z_range,
            "GR_Z_cov": z_cov,
            "GR_time": time_gr,
            "GR_gate_volume_sum": sum_vol,
            "GR_fraction_covered_area": fraction_covered_area,
            "GR_counts": counts,
            "GR_counts_valid": counts_valid,
        },
        index=gdf_sr.index,
    )
    gdf_gr_match = gpd.GeoDataFrame(df, crs=crs_gr, geometry=aggregator.target_polygons)
    gdf_gr_match.head()

    # Add GR range statistics
    gdf_gr_match["GR_range_max"] = aggregator.max(values=gdf_gr["range"])
    gdf_gr_match["GR_range_mean"] = aggregator.mean(values=gdf_gr["range"])
    gdf_gr_match["GR_range_min"] = aggregator.min(values=gdf_gr["range"])

    # Fraction above sensitivity thresholds
    for thr in sr_sensitivity_thresholds:
        fraction = aggregator.apply(lambda x, weights: np.sum(x > thr), values=gdf_gr["DBZH"]) / counts  # noqa
        gdf_gr_match[f"GR_Z_fraction_above_{thr}dBZ"] = fraction

    # Compute further aggregation statistics
    stats_var = ["vres", "hres", "x", "y", "z", "DBZH_cumsum"]
    for var in stats_var:
        gdf_gr_match[f"GR_{var}_mean"] = aggregator.average(values=gdf_gr[var])
        gdf_gr_match[f"GR_{var}_min"] = aggregator.min(values=gdf_gr[var])
        gdf_gr_match[f"GR_{var}_max"] = aggregator.max(values=gdf_gr[var])
        gdf_gr_match[f"GR_{var}_range"] = gdf_gr_match[f"GR_{var}_max"] - gdf_gr_match[f"GR_{var}_min"]

    # Compute horizontal distance between centroids
    func_dict = {"min": np.min, "mean": np.mean, "max": np.max}
    for suffix, func in func_dict.items():
        arr = np.zeros(aggregator.n_target_polygons) * np.nan
        arr[aggregator.target_intersecting_indices] = np.array(
            [func(dist) for i, dist in aggregator.dict_distances.items()],
        )
        gdf_gr_match[f"distance_horizontal_{suffix}"] = arr

    ####-----------------------------------------------------------------------------.
    #### Create the SR/GR Database
    # Create DataFrame with matched variables
    gdf_match = gdf_gr_match.merge(gdf_sr, on="geometry")

    # Compute ratio SR/GR volume
    gdf_match["VolumeRatio"] = gdf_match["SR_gate_volume_sum"] / gdf_match["GR_gate_volume_sum"]

    # Compute difference in SR/GR gate volume
    gdf_match["VolumeDiff"] = gdf_match["SR_gate_volume_sum"] - gdf_match["GR_gate_volume_sum"]

    # Compute time difference
    gdf_match["time_difference"] = gdf_match["GR_time"] - gdf_match["SR_time"]

    # Compute lower bound and upper bound height
    gdf_match["GR_z_lower_bound"] = gdf_match["GR_z_min"] - gdf_match["GR_vres_mean"] / 2
    gdf_match["GR_z_upper_bound"] = gdf_match["GR_z_max"] + gdf_match["GR_vres_mean"] / 2

    gdf_match["SR_z_lower_bound"] = gdf_match["SR_z_min"] - gdf_match["SR_vres_mean"] / 2
    gdf_match["SR_z_upper_bound"] = gdf_match["SR_z_max"] + gdf_match["SR_vres_mean"] / 2

    ####-----------------------------------------------------------------------------.
    return gdf_match
