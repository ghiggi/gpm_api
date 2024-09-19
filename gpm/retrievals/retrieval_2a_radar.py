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
"""This module contains GPM RADAR 2A products community-based retrievals."""
import numpy as np
import xarray as xr

import gpm
from gpm.checks import check_has_vertical_dim
from gpm.utils.manipulations import (
    get_bright_band_mask,
    get_height_at_temperature,
    get_liquid_phase_mask,
    get_range_axis,
    get_solid_phase_mask,
    get_vertical_datarray_prototype,
)
from gpm.utils.xarray import (
    get_dimensions_without,
    get_xarray_variable,
)

### TODO: requirements Ku, Ka band ...
# check single_frequency
# isel(radar_frequency=freq, missing_dims="ignore")

# Add retrieval decorators specifying variables
# Add retrieval decorators specifying if 2D spatial dimensions required (otherwise assume no)


def get_range_resolution(ds):
    """Return the range bin size."""
    ds = ds.isel(cross_track=0, along_track=0, radar_frequency=0, range=slice(-2, None), missing_dims="ignore")
    try:
        range_resolution = retrieve_range_resolution(ds).to_numpy()[0]
    except Exception:
        product = ds.attrs.get("gpm_api_product")
        scan_mode = ds.attrs.get("ScanMode")
        # TODO: Make more accurate and depending on scan mode and satellite !
        if product in gpm.available_products(product_levels="2A", product_categories="RADAR"):
            range_resolution = 125 if scan_mode in ["FS", "NS", "MS"] else 250
        else:
            raise ValueError("Expecting a radar product.")
    return range_resolution


def get_nrange(ds):
    """Return the number of radar gates in L2 products."""
    product = ds.attrs.get("gpm_api_product")
    scan_mode = ds.attrs.get("ScanMode")
    # TODO: Make more accurate and depending on scan mode and satellite !
    if product in gpm.available_products(product_levels="2A", product_categories="RADAR"):
        if scan_mode in ["FS", "NS", "MS"]:
            return 176
        return 88
    raise ValueError("Expecting a radar product.")


def retrieve_range_resolution(ds):
    """Retrieve range resolution."""
    # Retrieve required DataArrays
    check_has_vertical_dim(ds)
    range_bin = get_vertical_datarray_prototype(ds, fill_value=1) * ds["range"]  # start at 1
    height = get_xarray_variable(ds, variable="height")
    alpha = get_xarray_variable(ds, variable="localZenithAngle")
    ellipsoidBinOffset = get_xarray_variable(ds, variable="ellipsoidBinOffset")
    n_gates = get_nrange(ds)
    range_distance_from_ellipsoid = height / np.cos(np.deg2rad(alpha))
    range_distance_from_gate_at_ellipsoid = range_distance_from_ellipsoid - ellipsoidBinOffset
    range_resolution = range_distance_from_gate_at_ellipsoid / (n_gates - range_bin)
    return range_resolution


def retrieve_range_distance_from_ellipsoid(ds):
    """Retrieve distance from the ellipsoid along the radar beam.

    Accurate range resolution is provided by 'rangeBinSize' in the L1B product.
    Requires: ellipsoidBinOffset, height, localZenithAngle.
    """
    # ---------------------------------------------------------------------------.
    # Alternative code

    # height = get_xarray_variable(ds, variable="height")
    # alpha = get_xarray_variable(ds, variable="localZenithAngle")
    # ellipsoidBinOffset = get_xarray_variable(ds, variable="ellipsoidBinOffset")
    # range_distance_from_ellipsoid = height / np.cos(np.deg2rad(alpha))

    # import matplotlib.pyplot as plt
    # alpha = get_xarray_variable(ds, variable="localZenithAngle")
    # z_sr1 = np.cos(np.deg2rad(alpha)) * range_distance_from_ellipsoid
    # diff = ds["height"] - z_sr1
    # _ = plt.hist(diff.data.flatten()) # [-60, 60]
    # ---------------------------------------------------------------------------.

    # Retrieve required DataArrays
    check_has_vertical_dim(ds)
    range_bin = get_vertical_datarray_prototype(ds, fill_value=1) * ds["range"]  # start at 1
    ellipsoidBinOffset = get_xarray_variable(ds, variable="ellipsoidBinOffset")
    range_resolution = get_range_resolution(ds)

    # Compute range_distance_from_ellipsoid
    n_gates = get_nrange(ds)
    range_distance_from_ellipsoid = ellipsoidBinOffset + (n_gates - range_bin) * range_resolution

    # Name the DataArray
    range_distance_from_ellipsoid.name = "range_distance_from_ellipsoid"
    return range_distance_from_ellipsoid


def retrieve_range_distance_from_satellite(ds, scan_angle=None, earth_radius=6371000):
    """Retrieve range distance in meters (at bin center) from the satellite.

    This is a rough approximation took from wradlib dist_from_orbit function.

    Parameters
    ----------
    ds : xarray.Dataset
        GPM radar dataset.
    scan_angle : xarray.DataArray, optional
        Cross-track scan angle in degree.
        If not specified, assumes ``np.abs(np.linspace(-17.04, 17.04, 49))`` for Ku.
    earth_radius: float, optional
        Earth radius.

    Returns
    -------
    da : xarray.DataArray
        Range distances from the satellite.

    """
    # TODO: check is dataset, single frequency

    # Retrieve local zenith angle
    alpha = get_xarray_variable(ds, variable="localZenithAngle")

    # Retrieve satellite altitude
    if "dprAlt" in ds:
        sat_alt = get_xarray_variable(ds, variable="dprAlt")  # only available in 2A-DPR !
    else:
        sat_alt = get_xarray_variable(ds, variable="scAlt")

    # Retrieve distance from ellipsoid toward the satellite along the radar beam
    range_distance_from_ellipsoid = ds.gpm.retrieve("range_distance_from_ellipsoid")

    # Define off-nadir scan angle
    # --> TODO: Add scan_angle from dataset: 17 for Ku, 8.5 for Ka in MS?
    if scan_angle is None:
        scan_angle_template = np.abs(np.linspace(-17.04, 17.04, 49))  # assuming Ku swath of size 49
        scan_angle = xr.DataArray(scan_angle_template[ds["gpm_cross_track_id"].data], dims="cross_track")
    scan_angle = np.abs(scan_angle)  # ensure absolute values

    # Compute radial distance (along the radar beam ) from satellite to the Earth surface
    # - WRADLIB: https://github.com/wradlib/wradlib/blob/main/wradlib/georef/satellite.py#L203C6-L203C28
    range_distance_at_ellipsoid = (
        (earth_radius + sat_alt) * np.cos(np.radians(alpha - scan_angle)) - earth_radius
    ) / np.cos(np.radians(alpha))

    # Compute range distance to the satellite
    range_distance_from_satellite = range_distance_at_ellipsoid - range_distance_from_ellipsoid
    return range_distance_from_satellite


# def retrieve_range_distance_from_satellite(ds):
#     """Retrieve range distance in L2 RADAR Products.

#     Not possible because scRangeEllipsoid is not provided !
#     """
#     # rangeBinSize :
#     da_vertical = ds[get_vertical_variables(ds)[0]]
#     range_bin = xr.ones_like(da_vertical) * ds["range"] # start at 1 !
#     rangeBinSize = 125 # To estimate based on scan_mode !
#     binEllipsoid_2A = get_xarray_variable(ds, variable="binEllipsoid_2A")
#     ellipsoidBinOffset = get_xarray_variable(ds, variable="ellipsoidBinOffset")
#     scRangeEllipsoid = get_xarray_variable(ds, variable="scRangeEllipsoid") # MISSING

#     # Compute range distance
#     index_from_ellipsoid = binEllipsoid_2A - range_bin  # 0 at ellipsoid, increasing above
#     range_distance_from_ellipsoid = index_from_ellipsoid * rangeBinSize
#     range_distance_at_ellispoid = scRangeEllipsoid - ellipsoidBinOffset
#     range_distance = range_distance_at_ellispoid - range_distance_from_ellipsoid

#     # Name the DataArray
#     range_distance.name = "range_distance"
#     return range_distance


def retrieve_gate_volume(ds, beam_width=None, range_distance=None, scan_angle=None):
    r"""Calculates the sampling volume of the radar beam.

    We assume a cone frustum which has the volume :math:`V=(\\pi/3) \\cdot h \\cdot (R^2 + R \\cdot r + r^2)`.
    R and r are the radii of the two frustum surface circles.
    Assuming that the pulse width is small compared to the range, we get
    :math:`R=r= \\tan ( 0.5 \\cdot \\theta \\cdot \\pi/180 ) \\cdot range`,
    with theta being the aperture angle (beam width).
    Thus, the radar gate volume simply becomes the volume of a cylinder with
    :math:`V=\\pi \\cdot h \\cdot range^2 \\cdot \\tan(0.5 \\cdot \\theta \\cdot \\pi/180)^2`

    Parameters
    ----------
    ds: xarray.Dataset
        GPM radar dataset.
    beam_width : float
        The cross-track aperture angle of the radar beam [degree].
        The accurate cross-track aperture angle is provided by 'crossTrackBeamWidth' in the L1B product.
    range_distance: xarray.DataArray
        DataArray with range distance from each satellite at each radar gate.
        If not specified, is computed using a rough approximation.
    scan_angle: xarray.DataArray
        Used to estimate range_distance if not specified.
        Cross-track scan angle in degree.
        If not specified, assumes ``np.abs(np.linspace(-17.04, 17.04, 49))`` for Ku.

    Returns
    -------
    xarray.DataArray
        Volume of radar gates in cubic meters.
    """
    if beam_width is None:
        beam_width = 0.71
    if range_distance is None:
        range_distance = retrieve_range_distance_from_satellite(ds, scan_angle=scan_angle)  # earth radius
    range_resolution = get_range_resolution(ds)
    # cross_section_radius = range_distance * np.tan(np.radians(beam_width / 2.0))
    # cross_section_area = np.pi * (range_distance**2) * (np.tan(np.radians(beam_width / 2.0)) ** 2)
    # gate_volume = range_resolution * cross_section_area
    gate_volume = np.pi * range_resolution * (range_distance**2) * (np.tan(np.radians(beam_width / 2.0)) ** 2)
    gate_volume.name = "gate_volume"
    gate_volume.attrs["units"] = "m3"
    return gate_volume


def retrieve_gate_resolution(ds, beam_width, range_distance=None, scan_angle=None):
    """
    Retrieve the horizontal and vertical 'resolution' of each radar gates.

    The gate horizontal resolution is computed by projecting the radius of the radar gate
    onto the horizontal plane, averaging the projected radii in the along-track and cross-track directions.

    The gate vertical resolution is computed by projecting the radar gate range resolution onto the vertical plane.

    Parameters
    ----------
    ds: xarray.Dataset
        GPM radar dataset.
    beam_width : float
        The cross-track aperture angle of the radar beam [degree].
        The accurate cross-track aperture angle is provided by 'crossTrackBeamWidth' in the L1B product.
    range_distance: xarray.DataArray
        DataArray with range distance from each satellite at each radar gate.
        If not specified, is computed using a rough approximation.
    scan_angle: xarray.DataArray
        Used to estimate range_distance if not specified.
        Cross-track scan angle in degree.
        If not specified, assumes ``np.abs(np.linspace(-17.04, 17.04, 49))`` for Ku.

    Returns
    -------
    tuple
        Tuple with (h_res, v_res).

    """
    # Retrieve local zenith angle
    alpha = get_xarray_variable(ds, variable="localZenithAngle")
    # Retrieve beamwidth
    if beam_width is None:
        beam_width = 0.71
    # Retrieve range resolution
    range_resolution = get_range_resolution(ds)
    # Retrieve range distance
    if range_distance is None:
        range_distance = retrieve_range_distance_from_satellite(ds, scan_angle=scan_angle)  # earth radius

    # Horizontal gate spacing
    # --> TODO: should use crossBeamWidth and alongTrackBeamWidth instead?
    h_res = (1 + np.cos(np.radians(alpha))) * range_distance * np.tan(np.deg2rad(beam_width / 2.0))

    # Vertical gate spacing
    # - wradlib, Crisologo and Warren et al., 2018 (Eq. A5) use " range_resolution / np.cos(alpha) "
    # - Should be * ! And then it match with GPM product height difference !
    v_res = range_resolution * np.cos(np.deg2rad(alpha))

    # Add units and name
    h_res.name = "h_res"
    h_res.attrs["units"] = "m"
    v_res.name = "v_res"
    v_res.attrs["units"] = "m"
    return h_res, v_res


def retrieve_dfrMeasured(ds):
    """Retrieve measured DFR."""
    da_z = get_xarray_variable(ds, variable="zFactorMeasured")
    da_dfr = da_z.sel(radar_frequency="Ku") - da_z.sel(radar_frequency="Ka")
    da_dfr.name = "dfrMeasured"
    return da_dfr


def retrieve_dfrFinal(ds):
    """Retrieve final DFR."""
    da_z = get_xarray_variable(ds, variable="zFactorFinal")
    da_dfr = da_z.sel(radar_frequency="Ku") - da_z.sel(radar_frequency="Ka")
    da_dfr.name = "dfrFinal"
    return da_dfr


def retrieve_dfrFinalNearSurface(ds):
    """Retrieve final DFR near the surface."""
    da_z = get_xarray_variable(ds, variable="zFactorFinalNearSurface")
    da_dfr = da_z.sel(radar_frequency="Ku") - da_z.sel(radar_frequency="Ka")
    da_dfr.name = "dfrFinalNearSurface"
    return da_dfr


def retrieve_heightClutterFreeBottom(ds):
    """Retrieve clutter height."""
    da = ds.gpm.get_height_at_bin(bins="binClutterFreeBottom")
    da.name = "heightClutterFreeBottom"
    return da


def retrieve_heightRealSurfaceKa(ds):
    """Retrieve height of real surface at Ka band."""
    da = ds.gpm.get_height_at_bin(bins=ds["binRealSurface"].sel({"radar_frequency": "Ka"}))
    da.name = "heightRealSurfaceKa"
    return da


def retrieve_heightRealSurfaceKu(ds):
    """Retrieve height of real surface at Ku band."""
    da = ds.gpm.get_height_at_bin(bins=ds["binRealSurface"].sel({"radar_frequency": "Ku"}))
    da.name = "heightRealSurfaceKu"
    return da


def retrieve_flagPrecipitationType(xr_obj, method="major_rain_type"):
    """Retrieve major rain type from the 2A-<RADAR> typePrecip variable."""
    da = get_xarray_variable(xr_obj, variable="typePrecip")
    available_methods = ["major_rain_type"]
    if method not in available_methods:
        raise NotImplementedError(f"Implemented methods are {available_methods}")
    # Decode typePrecip
    # if method == "major_rain_type"
    da = da / 10000000
    da = da.astype(int)
    da.attrs["flag_values"] = [0, 1, 2, 3]
    da.attrs["flag_meanings"] = ["no rain", "stratiform", "convective", "other"]
    da.attrs["description"] = "Precipitation type"
    da.name = "flagPrecipitationType"
    return da


def retrieve_bright_band_ratio(ds, return_bb_mask=True):
    """Returns the Bright Band (BB) Ratio.

    A BB ratio of < 0 indicates that a bin is located below the melting layer (ML).
    A BB ratio of > 0 indicates that a bin is located above the ML.
    A BB ratio with values in between 0 and 1 indicates that the radar is inside the ML.

    This function has been ported as it is from wradlib.

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    xarray.DataArray
        BB Ratio (3D) and BB (2D) mask.
    """
    quality = get_xarray_variable(ds, variable="qualityBB")
    height = get_xarray_variable(ds, variable="height")
    bb_height = get_xarray_variable(ds, variable="heightBB")
    bb_width = get_xarray_variable(ds, variable="widthBB")

    # Identify column with BB
    has_bb = (bb_height > 0) & (bb_width > 0) & (quality == 1)
    has_bb.name = "qualityBB"

    # Set columns without BB to np.nan
    bb_height_m = bb_height.where(has_bb)
    bb_width_m = bb_width.where(has_bb)

    # Get median bright band height and width
    # - Need to compute because dask.array.nanmedian does not support reductions with axis=None (dim=None)
    # - https://github.com/dask/dask/pull/5684/files
    bb_height_m = bb_height_m.compute().median(skipna=True)
    bb_width_m = bb_width_m.compute().median(skipna=True)

    # Estimate the bottom/top melting layer height
    zmlt = bb_height_m + bb_width_m / 2.0
    zmlb = bb_height_m - bb_width_m / 2.0

    # Get Bright Band (BB) Ratio (3D DataArray)
    bb_ratio = (height - zmlb) / (zmlt - zmlb)

    if return_bb_mask:
        return bb_ratio, has_bb
    return bb_ratio


def retrieve_flagHydroClass(ds, reflectivity="zFactorFinal", bb_ratio=None, precip_type=None):
    """Classify reflectivity profile into rain, snow, hail and melting layer."""
    # Retrieve precip type and BB ratio
    if precip_type is None:
        precip_type = ds.gpm.retrieve("flagPrecipitationType", method="major_rain_type")

    if bb_ratio is None:
        bb_ratio = ds.gpm.retrieve("bright_band_ratio", return_bb_mask=False)

    # Retrieve clutter mask
    da_bin_clutter_free = get_xarray_variable(ds, variable="binClutterFreeBottom")
    da_mask_clutter = ds["range"] > da_bin_clutter_free
    da_mask_clutter = da_mask_clutter.transpose(..., "range")

    # Retrieve Ku reflectivity
    da_z = get_xarray_variable(ds, variable=reflectivity)

    # Initialize DataArray for hydrometeor class
    da_class = get_vertical_datarray_prototype(ds, fill_value=0)

    # Infer masks for melting layer
    da_above_ml_mask = bb_ratio >= 1  # above ML mask
    da_below_ml_mask = bb_ratio <= 0  # below ML mask
    da_ml_mask = (bb_ratio > 0) & (bb_ratio < 1)  # ML mask

    # Assign class to 3D array
    # - Below melting layer
    da_class = xr.where(
        da_below_ml_mask & precip_type.isin([1, 2, 3]),
        1,  # rain
        da_class,
    )
    # - Above melting layer
    da_class = xr.where(
        da_above_ml_mask & precip_type.isin([1, 3]),
        2,  # snow
        da_class,
    )
    da_class = xr.where(
        da_above_ml_mask & precip_type == 2,
        3,  # hail
        da_class,
    )
    # - Melting layer
    da_class = xr.where(
        da_ml_mask & precip_type.isin([1]),
        4,  # melting layer
        da_class,
    )
    # - Mask DataArray where precipitating
    da_class = da_class.where(da_z > 8, 0)

    # - Clutter
    da_class = xr.where(
        da_mask_clutter,
        5,  # clutter
        da_class,
    )

    # Add attributes
    da_class.attrs["flag_values"] = [0, 1, 2, 3, 4, 5]
    da_class.attrs["flag_meanings"] = ["no precipitation", "rain", "snow", "hail", "melting layer", "clutter"]
    da_class.name = "flagHydroClass"
    return da_class


def retrieve_s_band_cao2013(ds, reflectivity="zFactorFinal", bb_ratio=None, precip_type=None):
    r"""Retrieve S-band reflectivity from Ku-band reflectivity.

    Cao et al., 2013 provides coefficients to compute DFR (S-Ku) using the following polynomial:

    .. math::
        \text{DFR (S-Ku)} = a_0 + a_1 Z(\text{Ku})^1 + a_2 Z(\text{Ku})^2 + a_3 Z(\text{Ku})^3 + a_4 Z(\text{Ku})^4

    S-band reflectivity is derived by summing Ku-reflectivity to DFR (S-Ku).

    The Bright Band Ratio and the Precipitation Type are used to discriminate between
    rain, snow, hail, mixed phase gates, and use the adequate coefficients.

    The method :
    - uses snow coefficients if precipitation type is assumed to be stratiform (precip_type=1)
    - uses hail coefficients if precipitation type is assumed to be convective (precip_type=2)
    - assumes no mixed-phase (either snow or hail) above the melting layer
    - assumes no mixed-phase (either rain or full hail) below the melting layer

    Please ensure precip_type to have values 0, 1 or 2. precip type = 0 means no rain.

    Parameters
    ----------
    ds : xarray.Dataset
        GPM Radar dataset.
    reflectivity: xarray.DataArray or str
        3D reflectivity array or dataset variable name. The default is "zFactorFinal".
    bb_ratio : xarray.DataArray, optional
        Bright Band Ratio. If not specified, ``gpm.retrieve("bright_band_ratio")`` is called.
    precip_type : xarray.DataArray, optional
        DataArray indicating the precipitation class.
        Please ensure ``precip_type`` to have values 0, 1 or 2. ``precip type = 0`` means no rain.

    Returns
    -------
    xarray.DataArray
        S band reflectivity.
    """
    import wradlib as wrl

    # Retrieve precip type and BB ratio
    if precip_type is None:
        precip_type = ds.gpm.retrieve("flagPrecipitationType", method="major_rain_type")

    if bb_ratio is None:
        bb_ratio, bb_mask = ds.gpm.retrieve("bright_band_ratio", return_bb_mask=True)
        # bb_ratio = bb_ratio.where(bb_mask)

    # Retrieve Ku reflectivity
    da_z_ku = get_xarray_variable(ds, variable=reflectivity)

    # Infer masks for melting layer
    da_above_ml_mask = bb_ratio >= 1  # above ML mask
    da_below_ml_mask = bb_ratio <= 0  # below ML mask
    da_ml_mask = (bb_ratio > 0) & (bb_ratio < 1)  # ML mask

    # Define index from bottom to top of ML varying from 0 to 10
    # --> This is used to select the appropriate coefficients columns
    ind = xr.where(da_ml_mask, np.round(bb_ratio * 10), 0).astype("int")

    # Retrieve coefficients for reflectivity conversion from Cao et al., 2013
    a_s = wrl.trafo.KuBandToS.snow  # (5, 11) Columns represents transition from 100 % rain to 100% snow
    a_h = wrl.trafo.KuBandToS.hail  # (5, 11) Columns represents transition from 100 % rain to 100% hail

    # Initialize DataArray for coefficients
    ndegree = a_s.shape[0]
    da_coeff = xr.zeros_like(da_z_ku.expand_dims(dim={"degree": ndegree})) * np.nan

    # Assign coefficients to 3D array
    # - Above melting layer
    da_coeff = xr.where(
        da_above_ml_mask & precip_type == 1,
        a_s[:, 10],
        da_coeff,
    )
    da_coeff = xr.where(
        da_above_ml_mask & precip_type == 2,
        a_h[:, 10],
        da_coeff,
    )
    # - Below melting layer
    da_coeff = xr.where(
        da_below_ml_mask & precip_type == 1,
        a_s[:, 0],
        da_coeff,
    )
    da_coeff = xr.where(
        da_below_ml_mask & precip_type == 2,
        a_h[:, 0],
        da_coeff,
    )
    # - Inside the melting layer
    da_coeff = xr.where(
        da_ml_mask & precip_type == 1,
        xr.DataArray(a_s[:, ind], dims=["degree", *ind.dims]),
        da_coeff,
    )
    da_coeff = xr.where(
        da_ml_mask & precip_type == 2,
        xr.DataArray(a_h[:, ind], dims=["degree", *ind.dims]),
        da_coeff,
    )

    # Compute DFR S-Ku
    dfr_s_ku = (xr.concat([da_z_ku**i for i in range(ndegree)], dim="degree") * da_coeff).sum(dim="degree")

    # Compute S band
    da_z_s = da_z_ku + dfr_s_ku
    da_z_s.name = da_z_ku.name
    return da_z_s


def retrieve_c_band_tan(ds, reflectivity="zFactorFinal", bb_ratio=None, precip_type=None):
    """Retrieve C-band reflectivity from Ku-band reflectivity."""
    # TODO: reference? tan year?
    da_z_ku = get_xarray_variable(ds, variable=reflectivity)
    da_z_s = retrieve_s_band_cao2013(ds=ds, bb_ratio=bb_ratio, precip_type=precip_type)
    da_dfr_s_ku = da_z_s - da_z_ku
    da_z_x = da_z_ku + da_dfr_s_ku * 0.53
    return da_z_x.where(da_z_ku >= 0)


def retrieve_x_band_tan(ds, reflectivity="zFactorFinal", bb_ratio=None, precip_type=None):
    """Retrieve X-band reflectivity from Ku-band reflectivity."""
    da_z_ku = get_xarray_variable(ds, variable=reflectivity)
    da_z_s = retrieve_s_band_cao2013(ds=ds, bb_ratio=bb_ratio, precip_type=precip_type)
    da_dfr_s_ku = da_z_s - da_z_ku
    da_z_x = da_z_ku + da_dfr_s_ku * 0.32
    return da_z_x.where(da_z_ku >= 0)


def retrieve_REFC(
    ds,
    variable="zFactorFinal",
    radar_frequency="Ku",
    mask_bright_band=False,
    mask_solid_phase=False,
    mask_liquid_phase=False,
):
    """Retrieve the vertical maximum radar reflectivity in the column.

    Also called: Composite REFlectivity
    """
    if mask_solid_phase and mask_liquid_phase:
        raise ValueError("Either specify 'mask_solid_phase' or 'mask_liquid_phase'.")
    # Retrieve required DataArrays
    da = get_xarray_variable(ds, variable=variable).squeeze()
    if "radar_frequency" in da.dims:
        da = da.sel({"radar_frequency": radar_frequency})
    # Mask bright band region
    if mask_bright_band:
        da_bright_band = get_bright_band_mask(ds)
        da = da.where(~da_bright_band)
    # Mask ice phase region
    if mask_solid_phase:
        da_mask = get_solid_phase_mask(ds)
        da = da.where(da_mask)
    # Mask liquid phase region
    if mask_liquid_phase:
        da_mask = get_liquid_phase_mask(ds)
        da = da.where(da_mask)
    # Compute maximum
    da_max = da.max(dim="range")
    # Add attributes
    if mask_solid_phase:
        da_max.name = "REFC_liquid"
    elif mask_liquid_phase:
        da_max.name = "REFC_solid"
    else:
        da_max.name = "REFC"
    da_max.attrs["units"] = "dBZ"
    return da_max


def retrieve_REFCH(ds, variable="zFactorFinal", radar_frequency="Ku"):
    """Retrieve the height at which the maximum radar reflectivity is observed.

    Also called: Composite REFlectivity Height
    """
    # Retrieve required DataArrays
    da = get_xarray_variable(ds, variable=variable).squeeze()
    if "radar_frequency" in da.dims:
        da = da.sel({"radar_frequency": radar_frequency})
    da_height = ds["height"]

    # Compute maximum reflectivity height
    # - Need to use a fillValue because argmax fails if all nan along column
    # - Need to compute argmax becose not possible to isel with 1D array with dask
    da_all_na = np.isnan(da).all("range")
    da_argmax = da.fillna(-10).argmax(dim="range", skipna=True)
    da_argmax = da_argmax.compute()
    da_max_height = da_height.isel({"range": da_argmax})
    da_max_height = da_max_height.where(~da_all_na)

    # Add attributes
    da_max_height.name = "REFCH"
    da_max_height.attrs["description"] = "Composite REFlectivity Height "
    da_max_height.attrs["units"] = "dBZ"
    return da_max_height


def retrieve_EchoDepth(
    ds,
    threshold,
    variable="zFactorFinal",
    radar_frequency="Ku",
    min_threshold=0,
    mask_liquid_phase=False,
):
    """Retrieve Echo Depth with reflectivity above xx dBZ.

    Common thresholds are 18, 30, 50, 60 dbZ.
    """
    # Ensure thresholds is a list
    # if isinstance(threshold, (int, float)):
    #     threshold = [threshold]

    # Retrieve required DataArrays
    da = get_xarray_variable(ds, variable=variable).squeeze()
    if "radar_frequency" in da.dims:
        da = da.sel({"radar_frequency": radar_frequency})
    da_height = ds["height"].copy()

    # Mask height bin where not raining
    da_mask_3d_rain = da > min_threshold
    da_height = da_height.where(da_mask_3d_rain)

    # Mask heights where Z is not above threshold
    # da_threshold = xr.DataArray(threshold, coords={"threshold": threshold}, dims="threshold")
    # da_mask_3d = da > da_threshold
    da_mask_3d = da > threshold
    da_height_masked = da_height.where(da_mask_3d)

    # Mask liquid phase
    if mask_liquid_phase:
        da_liquid_mask = get_liquid_phase_mask(ds)
        da_height_masked = da_height_masked.where(~da_liquid_mask)

    # Retrieve min and max echo height
    da_max_height = da_height_masked.max(dim="range")
    da_min_height = da_height_masked.min(dim="range")

    # OLD MASKING
    # if mask_liquid_phase:
    #     da_isnan = np.isnan(da_min_height)
    #     da_height_melting = ds["heightZeroDeg"]
    #     da_height_melting = da_height_melting.where(~da_isnan)
    #     # If max is below the 0 °C isotherm --> set to nan
    #     da_max_height = da_max_height.where(da_max_height > da_height_melting)
    #     # If min below the 0 °C isoterm --> set the isotherm height
    #     da_min_height = da_min_height.where(da_min_height > da_height_melting, da_height_melting)

    # Compute depth
    da_depth = da_max_height - da_min_height

    # Add attributes
    da_depth.name = f"EchoDepth{threshold}dBZ"
    da_depth.attrs["units"] = "m"
    return da_depth


def retrieve_EchoTopHeight(
    ds,
    threshold,
    variable="zFactorFinal",
    radar_frequency="Ku",
    min_threshold=0,
):
    """Retrieve Echo Top Height (maximum altitude) for a particular reflectivity threshold.

    Common thresholds are 18, 30, 50, 60 dbZ.
    References: Delobbe and Holleman, 2006; Stefan and Barbu, 2018
    """
    # Retrieve required DataArrays
    da = get_xarray_variable(ds, variable=variable).squeeze()
    if "radar_frequency" in da.dims:
        da = da.sel({"radar_frequency": radar_frequency})
    da_height = ds["height"].copy()

    # Mask height bin where not raining
    da_mask_3d_rain = da > min_threshold
    da_height = da_height.where(da_mask_3d_rain)

    # Mask heights where Z is not above threshold
    da_mask_3d = da > threshold
    da_height_masked = da_height.where(da_mask_3d)

    # Retrieve max echo top height
    da_max_height = da_height_masked.max(dim="range")

    # Add attributes
    da_max_height.name = f"ETH{threshold}dBZ"
    da_max_height.attrs["units"] = "m"
    return da_max_height


def retrieve_VIL(ds, variable="zFactorFinal", radar_frequency="Ku"):
    """Compute Vertically Integrated Liquid indicator.

    Represents the total amount of rain that would fall if all the liquid water
    in a column inside a rain cloud (usually a thunderstorm) would be
    brought to the surface.

    Reference:
        Greene, D.R., and R.A. Clark, 1972:
        Vertically integrated liquid water - A new analysis tool.
        Mon. Wea. Rev., 100, 548-552.
        Amburn and Wolf (1997)
    """
    da = get_xarray_variable(ds, variable=variable).squeeze()
    if "radar_frequency" in da.dims:
        da = da.sel({"radar_frequency": radar_frequency})
    da_height = get_xarray_variable(ds, variable="height")
    da_mask = np.isnan(da).all(dim="range")

    # Compute the thickness of each level (difference between adjacent heights)
    thickness_arr = -da_height.diff(dim="range").data

    # Compute average Z between range bins [in mm^6/m-3]
    da_z = 10 ** (da / 10)  # Takes 2.5 seconds per granule
    n_ranges = len(da["range"])
    z_below = da_z.isel({"range": slice(0, n_ranges - 1)}).data
    z_above = da_z.isel({"range": slice(1, n_ranges)}).data
    z_avg_arr = (z_below + z_above) / 2

    # Clip reflectivity values at 56 dBZ
    vmax = 10 ** (56 / 10)
    z_avg_arr = z_avg_arr.clip(max=vmax)

    # Compute VIL profile
    thickness_arr = np.broadcast_to(thickness_arr, z_avg_arr.shape)
    vil_profile_arr = (z_avg_arr ** (4 / 7)) * thickness_arr  # Takes 3.8 s seconds per granule

    # Compute VIL
    range_axis = get_range_axis(da)
    dims = get_dimensions_without(da, dims="range")
    scale_factor = 3.44 * 10**-6
    vil_profile_arr[np.isnan(vil_profile_arr)] = 0  # because numpy.sum does not remove nan
    vil_arr = scale_factor * vil_profile_arr.sum(axis=range_axis)  # DataArray.sum is very slow !
    da_vil = xr.DataArray(vil_arr, dims=dims)

    # Mask where input profile is all Nan
    da_vil = da_vil.where(~da_mask)

    # Add attributes
    da_vil.name = "VIL"
    da_vil.attrs["description"] = "Radar-derived estimate of liquid water in a vertical column"
    da_vil.attrs["units"] = "kg/m2"
    return da_vil


def retrieve_VILD(
    ds,
    variable="zFactorFinal",
    radar_frequency="Ku",
    threshold=18,
    use_echo_top=False,
):
    """Compute Vertically Integrated Liquid Density.

    VIL Density = VIL/Echo Top Height

    By default, the Echo Top Height (or Echo Depth) is computed for 18 dBZ.
    More info at https://www.weather.gov/lmk/vil_density

    """
    da_vil = retrieve_VIL(ds, variable=variable, radar_frequency="Ku")
    if use_echo_top:
        da_e = retrieve_EchoTopHeight(
            ds,
            threshold=threshold,
            variable=variable,
            radar_frequency=radar_frequency,
            min_threshold=0,
        )
    else:
        da_e = retrieve_EchoDepth(
            ds,
            threshold=threshold,
            variable=variable,
            radar_frequency=radar_frequency,
            min_threshold=0,
        )
    da_vild = da_vil / da_e * 1000
    # Add attributes
    da_vild.name = "VILD"
    da_vild.attrs["description"] = "VIL Density"
    da_vild.attrs["units"] = "g/m3"
    return da_vild


def _get_weights(da, lower_threshold, upper_threshold):
    """Compute weights using a linear weighting function."""
    da_mask_nan = np.isnan(da)
    da_mask_lower = da < lower_threshold
    da_mask_upper = da > upper_threshold
    da_weights = (da - lower_threshold) / (upper_threshold - lower_threshold)
    da_weights = da_weights.where(~da_mask_lower, 0)
    da_weights = da_weights.where(~da_mask_upper, 1)
    da_weights = da_weights.where(~da_mask_nan)
    da_weights.name = "weights"
    return da_weights


def retrieve_HailKineticEnergy(
    ds,
    variable="zFactorFinal",
    radar_frequency="Ku",
    lower_threshold=40,
    upper_threshold=50,
):
    """Compute Hail Kinetic Energy.

    Lower and upper reflectivity thresholds are used to retain only
    higher reflectivities typically associated with hail and filtering out
    most of the lower reflectivities typically associated with liquid water.
    """
    # Retrieve required DataArrays
    da_z = get_xarray_variable(ds, variable=variable).squeeze()
    if "radar_frequency" in da_z.dims:
        da_z = da_z.sel({"radar_frequency": radar_frequency})

    # Compute W(Z)
    # - Used to define a transition zone between rain and hail reflectivities
    da_z_weighted = _get_weights(
        da_z,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )
    # Compute Hail Kinetic Energy
    scale_factor = 5 * (10**-6)
    da_e = scale_factor * 10 ** (0.084 * da_z) * da_z_weighted  # J/m2

    da_e.name = "HailKineticEnergy"
    da_e.attrs["description"] = "Hail kinetic energy"
    da_e.attrs["units"] = "J/m2/s"
    return da_e


def retrieve_SHI(
    ds,
    variable="zFactorFinal",
    radar_frequency="Ku",
    lower_z_threshold=40,
    upper_z_threshold=50,
):
    """Retrieve the Severe Hail Index (SHI).

    SHI is used to compute the Probability of Severe Hail (POSH) and Maximum Estimated Size of Hail (MESH).
    SHI applies a thermally weighted vertical integration of reflectivity from the melting level
    to the top of the storm, neglecting any reflectivity less than 40 dBZ,
    thereby attempting to capture only the ice content of a storm.

    Reference: Witt et al., 1998

    Parameters
    ----------
    ds : xarray.Dataset
        GPM L2 RADAR Dataset.
    variable : str, optional
        Reflectivity field. The default is "zFactorFinal".
    radar_frequency : str, optional
        Radar frequency. The default is "Ku".
    lower_z_threshold : int or  float, optional
        Lower reflectivity threshold. The default is 40 dBZ.
    upper_z_threshold : int or  float, optional
        Upper reflectivity threshold. The default is 50 dBZ.

    Returns
    -------
    da_shi : xarray.DataArray
        Severe Hail Index (SHI)

    """
    # Retrieve required DataArrays
    da_z = get_xarray_variable(ds, variable=variable).squeeze()
    if "radar_frequency" in da_z.dims:
        da_z = da_z.sel({"radar_frequency": radar_frequency})
    da_t = get_xarray_variable(ds, variable="airTemperature").squeeze()
    da_height = get_xarray_variable(ds, variable="height").squeeze()
    da_mask = np.isnan(da_z).all(dim="range")

    # Compute W(T)
    # - Used to define a transition zone between rain and hail reflectivities
    # - Hail growth only occurs at temperatures < 0°C
    # - Most growth for severe hail occurs at temperatures near -20°C or colder
    da_height_zero_deg = get_height_at_temperature(
        da_height=da_height,
        da_temperature=da_t,
        temperature=273.15,
    )  # 2.5 s per granule
    da_height_minus_20_deg = get_height_at_temperature(
        da_height=da_height,
        da_temperature=da_t,
        temperature=273.15 - 20,
    )  # 2.5 s per granule
    da_t_weighted = _get_weights(
        da_height,
        lower_threshold=da_height_zero_deg,
        upper_threshold=da_height_minus_20_deg,
    )  # 14 s per granule

    # Compute HailKineticEnergy
    da_e = retrieve_HailKineticEnergy(
        ds,
        variable=variable,
        radar_frequency=radar_frequency,
        lower_threshold=lower_z_threshold,
        upper_threshold=upper_z_threshold,
    )  # 12 s per granule

    # Define thickness array (difference between adjacent heights)
    da_thickness = -da_height.diff(dim="range")
    da_thickness = xr.concat([da_thickness, da_thickness.isel({"range": -1})], dim="range")
    da_thickness["range"] = da_e["range"]

    # Compute SHI
    da_shi_profile = da_e * da_t_weighted * da_thickness  # 3.45 s per granule
    da_shi_profile = da_shi_profile.where(da_height > da_height_zero_deg)  # 4.5 s per granule
    da_shi = 0.1 * da_shi_profile.sum("range")  # 4.3 s (< 1s with numpy.sum !)

    # Mask where input profile is all Nan
    da_shi = da_shi.where(~da_mask)

    # Add attributes
    da_shi.name = "SHI"
    da_shi.attrs["description"] = "Severe Hail Index "
    da_shi.attrs["units"] = "J/m/s"
    return da_shi


def retrieve_MESH(ds, scale_factor=2):
    """Retrieve the Maximum Estimated Size of Hail (MESH).

    Also known as the Maximum Expected Hail Size (MEHS).

    The “size” in MESH refers to the maximum diameter (in mm) of a hailstone.
    It's an indicator that transforms SHI into hail size by fitting SHI to a
    chosen percentile of maximum observed hail size (using a power-law)
    """
    da_shi = retrieve_SHI(ds)
    da_mesh = 2.54 * da_shi**0.5 * scale_factor
    # Add attributes
    da_mesh.name = "MESH"
    da_mesh.attrs["description"] = "Maximum Estimated Size of Hail"
    da_mesh.attrs["units"] = "mm"
    return da_mesh


def retrieve_POSH(ds):
    """The Probability of Severe Hail (POSH).

    The probability of 0.75-inch diameter hail occurring.

    When SHI = WT, POSH = 50%.
    Output probabilities are rounded off to the nearest 10%, to avoid conveying an unrealistic degree of precision.
    """
    # Retrieve zero-degree height
    da_height_0 = get_xarray_variable(ds, variable="heightZeroDeg").squeeze()
    # Retrieve warning threshold
    da_wt = 57.5 * da_height_0 - 121
    da_wt = da_wt.clip(min=20)
    # Retrieve SHI
    da_shi = retrieve_SHI(ds)
    mask_below_0 = da_shi <= 0
    da_shi = da_shi.where(~mask_below_0)
    # Retrieve POSH
    da_posh = 29 * np.log(da_shi / da_wt) + 50
    da_posh = da_posh.where(~mask_below_0, 0)
    da_posh = da_shi.clip(min=0, max=1).round(1) * 100
    # Add attributes
    da_posh.name = "POSH"
    da_posh.attrs["description"] = "Probability of Severe Hail"
    da_posh.attrs["units"] = "%"
    return da_posh


def retrieve_POH(ds, method="Foote2005"):
    """The Probability of Hail (POH) at the surface.

    Based on EchoDepth45dBZ above melting layer.

    No hail if EchoDepth45dBZ above melting layer < 1.65 km.
    100% hail if EchoDepth45dBZ above melting layer > 5.5 / 5.8 km.
    to 100% (hail; Δz > 5.5 km)

    Reference:
        - Foote et al., 2005. Hail metrics using conventional radar.

    Output probabilities are rounded off to the nearest 10%, to avoid
      conveying an unrealistic degree of precision.
    """
    # TODO: add utility to set 0 where rainy area (instead of nan value)
    variable = "zFactorFinal"
    radar_frequency = "Ku"
    # Compute POH
    da_echo_depth_45_solid = retrieve_EchoDepth(
        ds,
        threshold=45,
        variable=variable,
        radar_frequency=radar_frequency,
        mask_liquid_phase=True,
    )
    if method == "Foote2005":
        da_echo_depth_45_solid = da_echo_depth_45_solid / 1000
        da_poh = (
            -1.20231
            + 1.00184 * da_echo_depth_45_solid
            - 0.17018 * da_echo_depth_45_solid * da_echo_depth_45_solid
            + 0.01086 * da_echo_depth_45_solid * da_echo_depth_45_solid * da_echo_depth_45_solid
        )
        da_poh = da_poh.clip(0, 1).round(1) * 100
    else:
        raise NotImplementedError(f"Method {method} is not yet implemented.")

    # Add attributes
    da_poh.name = "POH"
    da_poh.attrs["description"] = "Probability of Hail"
    da_poh.attrs["units"] = "%"

    return da_poh


def retrieve_MESHS(ds):
    """The Maximum Expected Severe Hail Size at the surface.

    Based on EchoTop45dBZ.

    No hail if EchoDepth45dBZ above melting layer < 1.65 km.
    100% hail if EchoDepth45dBZ above melting layer > 5.5 / 5.8 km.
    """
    variable = "zFactorFinal"
    radar_frequency = "Ku"
    h0 = get_xarray_variable(ds, variable="heightZeroDeg")

    # Compute MESHS
    et_50_2cm = 1.5 * h0 + 1700
    et_50_4cm = 1.7824 * h0 + 2544.1
    et_50_6cm = 1.933 * h0 + 4040

    et50 = retrieve_EchoTopHeight(
        ds,
        threshold=45,  # C --> Ku-band
        variable=variable,
        radar_frequency=radar_frequency,
    )
    meshs4 = 4 + ((2 * (et50 - et_50_4cm)) / (et_50_6cm - et_50_4cm))
    meshs2 = 2 + (2 * (et50 - et_50_2cm) / (et_50_4cm - et_50_2cm))
    mask_between_2_4 = np.logical_and(et50 > et_50_2cm, et50 < et_50_4cm)
    mask_above_4 = et50 > et_50_4cm
    meshs2 = meshs2.where(mask_between_2_4, 0)
    meshs4 = meshs4.where(mask_above_4, 0)
    da_meshs = meshs2 + meshs4

    # Add attributes
    da_meshs.name = "MESHS"
    da_meshs.attrs["description"] = "Maximum Expected Severe Hail Size "
    da_meshs.attrs["units"] = "cm"

    return da_meshs
