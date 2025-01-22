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
"""This module contains GPM RADAR L1B products community-based retrievals."""
import datetime

import numpy as np
import xarray as xr

import gpm
from gpm.checks import check_has_vertical_dim
from gpm.utils.manipulations import (
    conversion_factors_degree_to_meter,
    get_vertical_datarray_prototype,
)
from gpm.utils.xarray import get_xarray_variable


def retrieve_range_distance_from_satellite(ds, mask_below_ellipsoid=False, first_option=True):
    """Retrieve range distance from the satellite (at bin center).

    If ``first_option=True``, requires: scRangeEllipsoid, ellipsoidBinOffset, rangeBinSize, binEllipsoid
    If ``first_option=False``, requires: startBinRange, rangeBinSize (binEllipsoid)

    """
    # Retrieve required DataArrays
    check_has_vertical_dim(ds)
    range_bin = get_vertical_datarray_prototype(ds, fill_value=1) * ds["range"]  # 'range' start at 1 !

    # Requires: scRangeEllipsoid, ellipsoidBinOffset, rangeBinSize, binEllipsoid
    if first_option:
        ellipsoidBinOffset = get_xarray_variable(ds, variable="ellipsoidBinOffset")
        rangeBinSize = get_xarray_variable(ds, variable="rangeBinSize")
        binEllipsoid = get_xarray_variable(ds, variable="binEllipsoid")
        # Compute range distance
        range_distance_at_ellipsoid = ds["scRangeEllipsoid"] - ellipsoidBinOffset  # at ellispoid bin center !
        range_index_from_ellipsoid = binEllipsoid - range_bin  # 0 at ellipsoid, increasing above, decreasing below
        range_distance_from_satellite = range_distance_at_ellipsoid - (range_index_from_ellipsoid * rangeBinSize)

    # Requires: startBinRange, rangeBinSize
    else:
        startBinRange = get_xarray_variable(ds, variable="startBinRange")
        rangeBinSize = get_xarray_variable(ds, variable="rangeBinSize")

        # Compute range distance
        range_distance_from_satellite = startBinRange + (range_bin - 1) * rangeBinSize
        if mask_below_ellipsoid:
            binEllipsoid = get_xarray_variable(ds, variable="binEllipsoid")

    # Mask below ellipsoid
    if mask_below_ellipsoid:
        range_distance_from_satellite = range_distance_from_satellite.where(range_bin <= binEllipsoid)

    # Name the DataArray
    range_distance_from_satellite.name = "range_distance_from_satellite"
    return range_distance_from_satellite


def retrieve_range_distance_from_ellipsoid(ds, mask_below_ellipsoid=False):
    """Retrieve range distance from the ellipsoid.

    Requires: ellipsoidBinOffset, rangeBinSize, binEllipsoid
    """
    # Retrieve required DataArrays
    check_has_vertical_dim(ds)
    range_bin = get_vertical_datarray_prototype(ds, fill_value=1) * ds["range"]  # start at 1 !

    binEllipsoid = get_xarray_variable(ds, variable="binEllipsoid")
    rangeBinSize = get_xarray_variable(ds, variable="rangeBinSize")
    ellipsoidBinOffset = get_xarray_variable(ds, variable="ellipsoidBinOffset")

    # Compute range_distance_from_ellipsoid
    # index_from_ellipsoid = binEllipsoid - range_bin  # 0 at ellipsoid, increasing above, decreasing below
    # range_distance_from_ellipsoid = index_from_ellipsoid * rangeBinSize + ellipsoidBinOffset
    range_distance_from_ellipsoid = (binEllipsoid - range_bin) * rangeBinSize + ellipsoidBinOffset

    # Mask below ellipsoid
    if mask_below_ellipsoid:
        range_distance_from_ellipsoid = range_distance_from_ellipsoid.where(range_bin <= binEllipsoid)

    # Name the DataArray
    range_distance_from_ellipsoid.name = "range_distance_from_ellipsoid"
    return range_distance_from_ellipsoid


def retrieve_height(ds, mask_below_ellipsoid=False):
    """Retrieve (normal) height from the the ellipsoid.

    From GPM DPR ATBD Level 2.
    Requires: scLocalZenith, ellipsoidBinOffset, rangeBinSize, binEllipsoid
    """
    # Retrieve required DataArrays
    check_has_vertical_dim(ds)
    range_bin = get_vertical_datarray_prototype(ds, fill_value=1) * ds["range"]  # start at 1 !

    binEllipsoid = get_xarray_variable(ds, variable="binEllipsoid")
    rangeBinSize = get_xarray_variable(ds, variable="rangeBinSize")
    ellipsoidBinOffset = get_xarray_variable(ds, variable="ellipsoidBinOffset")
    scLocalZenith = get_xarray_variable(ds, variable="scLocalZenith")

    # Compute height
    # index_from_ellipsoid = binEllipsoid - range_bin  # 0 at ellipsoid, increasing above, decreasing below
    # range_distance_from_ellipsoid = index_from_ellipsoid * rangeBinSize  # at bin center
    # height =  (range_distance_from_ellipsoid + ellipsoidBinOffset) * np.cos(np.deg2rad(scLocalZenith))
    height = ((binEllipsoid - range_bin) * rangeBinSize + ellipsoidBinOffset) * np.cos(np.deg2rad(scLocalZenith))

    # Mask below ellipsoid
    if mask_below_ellipsoid:
        height = height.where(range_bin <= binEllipsoid)

    # Name the DataArray
    height.name = "height"
    return height


def retrieve_sampling_type(ds):
    """Retrieve sampling type: low vs high resolution."""
    check_has_vertical_dim(ds)
    da_vertical = get_vertical_datarray_prototype(ds, fill_value=1)
    da_sampling_type_0 = da_vertical * 1
    da_sampling_type_1 = da_vertical * 2
    echoLowResBinNumber = get_xarray_variable(ds, variable="echoLowResBinNumber")
    echoHighResBinNumber = get_xarray_variable(ds, variable="echoHighResBinNumber")
    bin_HighResBinNumber = echoLowResBinNumber + echoHighResBinNumber

    da_sampling_type_0 = da_sampling_type_0.where(ds["range"] <= echoLowResBinNumber, 0)
    da_sampling_type_1 = da_sampling_type_1.where(
        np.logical_and(ds["range"] > echoLowResBinNumber, ds["range"] <= bin_HighResBinNumber),
        0,
    )
    da_sampling_type = da_sampling_type_0 + da_sampling_type_1
    da_sampling_type = da_sampling_type.where(da_sampling_type >= 1, np.nan) - 1
    da_sampling_type.name = "sampling_type"
    da_sampling_type.attrs = {"flag_values": [0, 1], "flag_meanings": ["low_resolution", "high_resolution"]}
    return da_sampling_type


def retrieve_ellipsoidBinOffset(ds, binEllipsoid_variable="binEllipsoid"):
    """Retrieve distance offset from ellipsoid to the center of the ellipsoid bin.

    This can only be used in L1B format.
    """
    # Retrieve required DataArrays
    binEllipsoid = get_xarray_variable(ds, variable=binEllipsoid_variable)
    rangeBinSize = get_xarray_variable(ds, variable="rangeBinSize")
    startBinRange = get_xarray_variable(ds, variable="startBinRange")  # distance between sensor and start data
    scRangeEllipsoid = get_xarray_variable(ds, variable="scRangeEllipsoid")  # distance between sensor and the ellipsoid

    # Compute ellipsoidBinOffset
    ellipsoidBinOffset = scRangeEllipsoid - (startBinRange + (binEllipsoid - 1) * rangeBinSize)

    # Name the DataArray
    ellipsoidBinOffset.name = "ellipsoidBinOffset"
    return ellipsoidBinOffset


def retrieve_scRangeDEM(ds):
    """Retrieve range distance from statellite to DEM.

    Requires: scRangeEllipsoid, DEMHmean, scLocalZenith
    """
    # Retrieve required DataArrays
    scRangeEllipsoid = get_xarray_variable(ds, variable="scRangeEllipsoid")
    DEMHmean = get_xarray_variable(ds, variable="DEMHmean")
    scLocalZenith = get_xarray_variable(ds, variable="scLocalZenith")

    # Compute scRangeDEM
    scRangeDEM = scRangeEllipsoid - DEMHmean * (1 / np.cos(np.deg2rad(scLocalZenith)))

    # Name the DataArray
    scRangeDEM.name = "scRangeDEM"
    return scRangeDEM


def retrieve_geolocation_3d(ds):
    """Retrieve the lat, lon, height 3D array.

    Requires: scLon, scLat, scLocalZenith, height.
    """
    # Retrieve required DataArrays
    height = ds["height"] if "height" in ds else retrieve_height(ds)

    x1 = ds["lon"]  # at ellipsoid
    y1 = ds["lat"]  # at ellipsoid
    xs = get_xarray_variable(ds, variable="scLon")  # at satellite
    ys = get_xarray_variable(ds, variable="scLat")  # at satellite
    scLocalZenith = get_xarray_variable(ds, variable="scLocalZenith")

    # Compute conversion factors deg to meter
    cx, cy = conversion_factors_degree_to_meter(y1)

    # Convert theta from degrees to radians
    tan_theta_rad = np.tan(np.deg2rad(scLocalZenith))

    # Calculate the distance 'dist' using the conversion factors
    dist = np.sqrt((cx * (xs - x1)) ** 2 + (cy * (ys - y1)) ** 2)

    # Calculate the delta components
    scale = height / dist * tan_theta_rad
    delta_x = (xs - x1) * scale
    delta_y = (ys - y1) * scale

    # Calculate the target coordinates (xp, yp)
    lon_3d = x1 + delta_x
    lat_3d = y1 + delta_y

    # Name the DataArray
    lon_3d.name = "lon3d"
    lat_3d.name = "lat_3d"

    # Return the DataArray
    return lon_3d, lat_3d, height


def retrieve_Psignal(ds, echoPower_variable="echoPower"):
    """Retrieve the signal echo power in milliWatt (received echo - noise echo)."""
    # Retrieve required DataArrays
    echoPower = get_xarray_variable(ds, variable=echoPower_variable)
    noisePower = get_xarray_variable(ds, variable="noisePower")
    # Compute Psignal
    Psignal = echoPower.gpm.idecibel - noisePower.gpm.idecibel
    Psignal = Psignal.where(Psignal > 0)  #  Psignal negative set as np.nan
    # Name the DataArray
    Psignal.name = "Psignal"
    Psignal.attrs["units"] = "mW"
    return Psignal


def retrieve_signalPower(ds, echoPower_variable="echoPower"):
    """Retrieve the signal power in dBm (received echo - noise echo).

    0 dBm means the measured power level is exactly one milliwatt.
    The higher the power, the more energy is returned.
    (Echo)Power in dBm is always between -20 and -120 dBm for GPM radar.
    """
    # Retrieve required DataArrays
    Psignal = retrieve_Psignal(ds, echoPower_variable=echoPower_variable)
    # Compute echoSignalPower
    echoSignalPower = Psignal.gpm.decibel
    # Name the DataArray
    echoSignalPower.name = "echoSignalPower"
    echoSignalPower.attrs["units"] = "dBm"
    return echoSignalPower


def get_dielectric_constant(ds, dielectric_constant=None):
    """Return the dielectric constant."""
    if dielectric_constant is not None:
        return dielectric_constant
    # Retrieve default constant
    # - Defaults are taken from L2 products
    # - L1B DielectricFactor<Band> value is -9999.9
    product = ds.attrs.get("gpm_api_product")
    if "Ka" in product:
        value = ds.attrs.get("DielectricFactorKa", -9999.9)
        if value == -9999.9:
            value = 0.8989
    elif "Ku" in product or "DPR" in product or "PR" in product:
        value = ds.attrs.get("DielectricFactorKu", -9999.9)
        if value == -9999.9:
            value = 0.9255
    else:
        raise ValueError("Expecting a radar product.")
    return value


def get_radar_wavelength(ds):
    """Return the radar wavelength."""
    default_dict = {
        "PR": 0.02173,
        "KuPR": 0.022044,
        "KaPR": 0.008433,
    }
    eqvWavelength = ds.attrs.get("eqvWavelength", None)
    if eqvWavelength is None:
        product = ds.attrs.get("gpm_api_product")
        if "Ka" in product:
            eqvWavelength = default_dict["KaPR"]
        elif "Ku" in product or "DPR" in product:
            eqvWavelength = default_dict["KuPR"]
        elif "PR" in product:
            eqvWavelength = default_dict["PR"]
        else:
            raise ValueError("Expecting a radar product.")
    return eqvWavelength


def retrieve_zMeasured(ds, signal_variable, dielectric_constant=None):
    """Return the reflectivity (in dBz) of a power signal (in dBm)."""
    # Retrieve required DataArrays
    range_distance_from_satellite = get_xarray_variable(ds, variable="range_distance_from_satellite")
    txAntGain = get_xarray_variable(ds, variable="txAntGain")  # dB
    rxAntGain = get_xarray_variable(ds, variable="rxAntGain")  # dB
    transPulseWidth = get_xarray_variable(ds, variable="transPulseWidth")  # s
    radarTransPower = get_xarray_variable(ds, variable="radarTransPower")  # dBm
    crossTrackBeamWidth = get_xarray_variable(ds, variable="crossTrackBeamWidth")  # degrees
    alongTrackBeamWidth = get_xarray_variable(ds, variable="alongTrackBeamWidth")  # degrees
    signalPower = get_xarray_variable(ds, variable=signal_variable)  # dBm

    # Retrieve constants
    dielectric_constant = get_dielectric_constant(ds, dielectric_constant)  # already squared !
    eqvWavelength = get_radar_wavelength(ds)
    speed_light = 299_792_458  # m/s

    # Compute reflectivity
    Psignal = signalPower.gpm.idecibel
    Ptransmitted = radarTransPower.gpm.idecibel
    txAntGain = txAntGain.gpm.idecibel
    rxAntGain = rxAntGain.gpm.idecibel
    constant = 2**10 * 10**18 * np.log(2) / (np.pi**3 * speed_light) * eqvWavelength**2
    z = (
        constant
        * range_distance_from_satellite**2
        * Psignal
        / (
            txAntGain
            * rxAntGain
            * np.deg2rad(crossTrackBeamWidth)
            * np.deg2rad(alongTrackBeamWidth)
            * transPulseWidth
            *
            # dielectric_constant ** 2  *
            dielectric_constant
            * Ptransmitted
        )
    )

    # Convert back to decibel
    zMeasured = z.gpm.decibel

    # Name the DataArray
    zMeasured.name = "zMeasured"
    zMeasured.attrs["units"] = "dBz"
    return zMeasured


def retrieve_sigmaMeasured(ds, signal_variable="signalPower", local_zenith_angle="scLocalZenith"):
    """Return the Normalized Radar Cross Section (NRCS) (in dB) of a power signal (in dBm).

    scLocalZenith in L1B product.
    LocalZenithAngle in L2 products.
    """
    # Retrieve required DataArrays
    range_distance_from_satellite = get_xarray_variable(ds, variable="range_distance_from_satellite")  # m
    txAntGain = get_xarray_variable(ds, variable="txAntGain")  # dB
    rxAntGain = get_xarray_variable(ds, variable="rxAntGain")  # dB
    receivedPulseWidth = get_xarray_variable(ds, variable="receivedPulseWidth")  # s
    radarTransPower = get_xarray_variable(ds, variable="radarTransPower")  # dBm
    crossTrackBeamWidth = get_xarray_variable(ds, variable="crossTrackBeamWidth")  # degrees
    alongTrackBeamWidth = get_xarray_variable(ds, variable="alongTrackBeamWidth")  # degrees
    lza = get_xarray_variable(ds, variable=local_zenith_angle)  # degree
    signalPower = get_xarray_variable(ds, variable=signal_variable)  # dBm

    # Retrieve constants
    eqvWavelength = get_radar_wavelength(ds)
    speed_light = 299_792_458  # m/s
    lza_rad = np.deg2rad(lza)

    # Convert from decibels
    Psignal = signalPower.gpm.idecibel
    Ptransmitted = radarTransPower.gpm.idecibel
    txAntGain = txAntGain.gpm.idecibel
    rxAntGain = rxAntGain.gpm.idecibel

    # Compute Normalized Radar Cross Section
    constant = 512 * np.pi**2 * np.log(2) / eqvWavelength**2
    theta_bp = 1 / (np.sqrt((np.deg2rad(crossTrackBeamWidth) ** -2) + (np.deg2rad(alongTrackBeamWidth) ** -2)))
    theta_p = 1 / (2 / (speed_light * receivedPulseWidth) * range_distance_from_satellite * np.tan(lza_rad))

    sigma = (
        constant
        * Psignal
        * range_distance_from_satellite**2
        * np.cos(lza_rad)
        / (txAntGain * rxAntGain * Ptransmitted * theta_p * theta_bp)
    )

    # Convert back to decibel
    sigmaMeasured = sigma.gpm.decibel

    # Name the DataArray
    sigmaMeasured.name = "sigmaMeasured"
    sigmaMeasured.attrs["units"] = "dB"
    return sigmaMeasured


def retrieve_snRatio(ds, echo_variable="echoPower", noise_variable="noisePower"):
    """Return the Signal to Noise Ratio in dB."""
    echoPower = get_xarray_variable(ds, variable=echo_variable)  # dBm
    noisePower = get_xarray_variable(ds, variable=noise_variable)  # dBm

    # Compute SNR
    ratio = echoPower.gpm.idecibel / noisePower.gpm.idecibel  # [mW/mW]
    snRatio = ratio.gpm.decibel

    # Name the DataArray
    snRatio.name = "snRatio"
    snRatio.attrs["units"] = "dB"
    return snRatio


def resample_hs_to_fs(ds_hs, smooth=True, except_vars=["echoCount", "sampling_type"]):
    """Resample HS using LUT.

    This function takes care of updating the bin variables and `startBinRange``,
    ``rangeBinSize`` and ``ellipsoidBinOffset``.

    Parameters
    ----------
    ds_hs : xarray.Dataset
        L1B or L2 Ka-band Dataset with HS scan mode.
    smooth : bool, optional
        If ``smooth=True``, applies averaging as presented in the L2 DPR ATBD.
        Note that if ``smooth=True``, the vertical variables and height/distance coordinates
        will have a duplicated values at range index 0 and 1!.
        The default is True.
    except_vars : list, optional
        List of vertical variables to not be smoothed (i.e. count variables).. The default is ["echoCount"].

    Returns
    -------
    ds_r : xarray.Dataset
        L1B or L2 Ka-band Dataset with resampled range gates (as in FS/NS/MS scan modes).
    """
    from gpm.utils.manipulations import get_vertical_coords_and_vars

    # Find range index (using LUT)
    n_along_track = ds_hs.sizes["along_track"]
    n_cross_track = ds_hs.sizes["cross_track"]
    n_range = ds_hs.sizes["range"] * 2
    da = xr.DataArray(
        np.ones((n_cross_track, n_along_track, n_range)),
        dims=("cross_track", "along_track", "range"),
        coords={"range": np.arange(1, n_range + 1)},
    )
    range_lut = xr.DataArray(np.repeat(np.arange(1, len(ds_hs["range"]) + 1), 2), dims="range")
    range_indexing = da * range_lut
    range_indexing = range_indexing.astype(int)

    # Reindex (using nearest approach)
    # ds_output = ds_ka_vertical.isel(range=range_indexing-1) # THIS DOES NOT WORK !
    ds_r = ds_hs.sel(range=range_indexing)

    # Update range and gpm_range_id
    new_range_size = len(ds_hs["range"]) * 2
    start_range = ds_hs["range"].data[0]
    ds_r = ds_r.assign_coords(
        {
            "range": ("range", np.arange(start_range, new_range_size + 1)),
            "gpm_range_id": ("range", np.arange(start_range - 1, new_range_size)),
        },
    )

    # Correct bin variables !
    for var in ds_r.gpm.bin_variables:
        ds_r[var] = ds_r[var] * 2  # - 2

    # Update variables (used by L1B retrievals)
    # - rangeBinSize (for retrieve_range_distance_from_satellite and reflectivity retrievals)
    if "rangeBinSize" in ds_r:
        ds_r["rangeBinSize"] = ds_r["rangeBinSize"] / 2

    # # - echoLowResBinNumber and echoHighResBinNumber (for sampling_type)
    for var in ["echoLowResBinNumber", "echoHighResBinNumber"]:
        if var in ds_r:
            ds_r[var] = ds_r[var] * 2

    # Correct startBinRange
    if "startBinRange" in ds_r:
        da_rangeBinSize = get_xarray_variable(ds_r, variable="rangeBinSize")
        ds_r["startBinRange"] = ds_r["startBinRange"] - da_rangeBinSize / 2  # half of the new

    # Correct ellipsoid_bin_offset
    if "ellipsoidBinOffset" in ds_r:
        da_rangeBinSize = get_xarray_variable(ds_r, variable="rangeBinSize")
        ds_r["ellipsoidBinOffset"] = ds_r["ellipsoidBinOffset"] - da_rangeBinSize / 2  # half of the new

    # Smooth out vertical continouus variables and coordinates
    # --> Should not smooth integers/categories like "EchoCount"
    # --> Roll on range and then reset natives !
    if smooth:
        ds_r = ds_r.transpose(..., "range")
        for var in get_vertical_coords_and_vars(ds_r):
            if var not in except_vars:
                da = ds_r[var]
                src_data = da.data.copy()
                da_smooth = (da + da.shift(range=1)) / 2
                # Replicate first value (that would be otherwise NaN)
                # - This cause unrealistic values for height/distance variables
                da_smooth.data[:, :, 0] = src_data[:, :, 0]
                ds_r[var].data = da_smooth.data
    return ds_r


def open_dataset_1b_ka_fs(
    start_time,
    end_time,
    variables=None,
    groups=None,
    product_type="RS",
    chunks={},
    verbose=False,
    parallel=False,
    l2_format=True,
    **kwargs,
):
    """Open 1B-Ka dataset in FS scan_mode format in either L1B or L2 format.

    It expects start_time after the GPM DPR scan pattern change occurred the 8 May 2018.
    (Over)Resample HS on MS (using LUT/range distance).
    The L2 FS format has 176 bins.

    Notes
    -----
    - 1B-Ka HS has a zig-zag pattern after scan change in 2018 !
    - 1B-Ka HS has 1 scan more than 1B-Ka MS (when queried by time)
    - 1B-Ka HS have range resolution of 250 m (130 bins)
    - 1B-Ka MS have range resolution of 125 m (260 bins)
    - 2A-Ka HS have range resolution of 125 m (88 bins)
    - 2A-Ka MS have range resolution of 125 m (176 bins)

    """
    from gpm.io.checks import check_time

    dt = gpm.open_datatree(
        product="1B-Ka",
        product_type=product_type,
        version=7,
        variables=variables,
        groups=groups,
        # Search also a bit around to
        start_time=check_time(start_time) - datetime.timedelta(seconds=2),
        end_time=check_time(end_time) + datetime.timedelta(seconds=2),
        chunks=chunks,
        decode_cf=True,
        parallel=parallel,
        verbose=verbose,
        **kwargs,
    )
    ds_l1_ka_ms = dt["MS"].to_dataset().compute()
    ds_l1_ka_hs = dt["HS"].to_dataset().compute()

    # Ensure matched scans
    idx_hs_start = np.where(ds_l1_ka_ms["gpm_id"].data[0] == ds_l1_ka_hs["gpm_id"].data)[0].item()
    ds_l1_ka_hs = ds_l1_ka_hs.isel(along_track=slice(idx_hs_start - 1, idx_hs_start + len(ds_l1_ka_ms["gpm_id"])))

    # Retrieve sampling_type (before L2 extraction) if 'range' dimension available
    dataset_with_range = "range" in ds_l1_ka_hs
    if dataset_with_range:
        ds_l1_ka_ms["sampling_type"] = ds_l1_ka_ms.gpm.retrieve("sampling_type")
        ds_l1_ka_hs["sampling_type"] = ds_l1_ka_hs.gpm.retrieve("sampling_type")

    # Split 1B-Ka HS scans in 1_12 and 37_49 scan angles
    ds_l1_ka_hs_37_49 = ds_l1_ka_hs.isel(cross_track=slice(0, 12))
    ds_l1_ka_hs_1_12 = ds_l1_ka_hs.isel(cross_track=slice(12, 24))

    ds_l1_ka_hs_37_49 = ds_l1_ka_hs_37_49.isel(along_track=slice(1, None))
    ds_l1_ka_hs_1_12 = ds_l1_ka_hs_1_12.isel(along_track=slice(0, -1))

    # Extract L2 dataset if asked, resample and concatenate
    if dataset_with_range:
        if l2_format:
            ds_l2_ka_ms = ds_l1_ka_ms.gpm.extract_l2_dataset()
            ds_l2_ka_hs_1_12 = ds_l1_ka_hs_1_12.gpm.extract_l2_dataset()
            ds_l2_ka_hs_37_49 = ds_l1_ka_hs_37_49.gpm.extract_l2_dataset()

            ds_l2_ka_hs_1_12_r = resample_hs_to_fs(ds_hs=ds_l2_ka_hs_1_12)
            ds_l2_ka_hs_37_49_r = resample_hs_to_fs(ds_hs=ds_l2_ka_hs_37_49)
            ds = xr.concat((ds_l2_ka_hs_1_12_r, ds_l2_ka_ms, ds_l2_ka_hs_37_49_r), dim="cross_track")
        else:
            ds_l1_ka_hs_1_12_r = resample_hs_to_fs(ds_hs=ds_l1_ka_hs_1_12)
            ds_l1_ka_hs_37_49_r = resample_hs_to_fs(ds_hs=ds_l1_ka_hs_37_49)
            ds = xr.concat((ds_l1_ka_hs_1_12_r, ds_l1_ka_ms, ds_l1_ka_hs_37_49_r), dim="cross_track")
    # Othweise concatenate only
    else:
        ds = xr.concat((ds_l1_ka_hs_1_12, ds_l1_ka_ms, ds_l1_ka_hs_37_49), dim="cross_track")
    # Finalize dataset coordinates and attributes
    ds = ds.assign_coords(
        {
            "gpm_along_track_id": ("along_track", ds_l1_ka_ms["gpm_along_track_id"].data),
            "gpm_id": ("along_track", ds_l1_ka_ms["gpm_id"].data),
            "gpm_cross_track_id": ("cross_track", np.arange(0, len(ds["cross_track"]))),
            "time": ("along_track", ds_l1_ka_ms["time"].data),
        },
    )
    ds.attrs["ScanMode"] = "FS"
    return ds
