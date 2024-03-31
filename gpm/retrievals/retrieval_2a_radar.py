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
"""This module contains GPM RADAR 2A products community-based retrivals."""
import numpy as np
import xarray as xr

from gpm.utils.manipulations import (
    get_bright_band_mask,
    get_dims_without,
    get_height_at_temperature,
    get_liquid_phase_mask,
    get_range_axis,
    get_solid_phase_mask,
    get_variable_dataarray,
)

### TODO: requirements Ku, Ka band ...


def retrieve_dfrMeasured(ds):
    """Retrieve measured DFR."""
    da_z = ds["zFactorMeasured"]
    da_dfr = da_z.sel(radar_frequency="Ku") - da_z.sel(radar_frequency="Ka")
    da_dfr.name = "dfrMeasured"
    return da_dfr


def retrieve_dfrFinal(ds):
    """Retrieve final DFR."""
    da_z = ds["zFactorFinal"]
    da_dfr = da_z.sel(radar_frequency="Ku") - da_z.sel(radar_frequency="Ka")
    da_dfr.name = "dfrFinal"
    return da_dfr


def retrieve_dfrFinalNearSurface(ds):
    """Retrieve final DFR near the surface."""
    da_z = ds["zFactorFinalNearSurface"]
    da_dfr = da_z.sel(radar_frequency="Ku") - da_z.sel(radar_frequency="Ka")
    da_dfr.name = "dfrFinalNearSurface"
    return da_dfr


def retrieve_heightClutterFreeBottom(ds):
    """Retrieve clutter height."""
    da = ds.gpm.get_height_at_bin(bin="binClutterFreeBottom")
    da.name = "heightClutterFreeBottom"
    return da


def retrieve_heightRealSurfaceKa(ds):
    """Retrieve height of real surface at Ka band."""
    da = ds.gpm.get_height_at_bin(bin=ds["binRealSurface"].sel({"radar_frequency": "Ka"}))
    da.name = "heightRealSurfaceKa"
    return da


def retrieve_heightRealSurfaceKu(ds):
    """Retrieve height of real surface at Ku band."""
    da = ds.gpm.get_height_at_bin(bin=ds["binRealSurface"].sel({"radar_frequency": "Ku"}))
    da.name = "heightRealSurfaceKu"
    return da


def retrieve_precipitationType(xr_obj, method="major_rain_type"):
    """Retrieve major rain type from the 2A-<RADAR> typePrecip variable."""
    da = get_variable_dataarray(xr_obj, variable="typePrecip")
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
    return da


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
    da = get_variable_dataarray(ds, variable=variable)
    if len(da["radar_frequency"].data) != 1:
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
    da = get_variable_dataarray(ds, variable=variable)
    if len(da["radar_frequency"].data) != 1:
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
    # Retrieve required DataArrays
    da = get_variable_dataarray(ds, variable=variable)
    if len(da["radar_frequency"].data) != 1:
        da = da.sel({"radar_frequency": radar_frequency})
    da_height = ds["height"].copy()
    # Mask height bin where not raining
    da_mask_3d_rain = da > min_threshold
    da_height = da_height.where(da_mask_3d_rain)

    # Mask heights where Z is not above threshold
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
    da = get_variable_dataarray(ds, variable=variable)
    if len(da["radar_frequency"].data) != 1:
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
    da = ds[variable].sel({"radar_frequency": radar_frequency})
    heights_arr = np.asanyarray(da["range"].data)
    da_mask = np.isnan(da).all(dim="range")

    # Compute the thickness of each level (difference between adjacent heights)
    thickness_arr = np.diff(heights_arr)

    # Compute average Z between range bins [in mm^6/m-3]
    da_z = 10 ** (da / 10)  # Takes 2.5 seconds per granule
    n_ranges = len(da["range"])
    z_below = da_z.isel(range=slice(0, n_ranges - 1)).data
    z_above = da_z.isel(range=slice(1, n_ranges)).data
    z_avg_arr = (z_below + z_above) / 2

    # Clip reflectivity values at 56 dBZ
    vmax = 10 ** (56 / 10)
    z_avg_arr = z_avg_arr.clip(max=vmax)

    # Compute VIL profile
    thickness_arr = np.broadcast_to(thickness_arr, z_avg_arr.shape)
    vil_profile_arr = (z_avg_arr ** (4 / 7)) * thickness_arr  # Takes 3.8 s seconds per granule

    # Compute VIL
    range_axis = get_range_axis(da)
    dims = get_dims_without(da, dims=["range"])
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
    use_echo_top=True,
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
    da_z = ds[variable].sel({"radar_frequency": radar_frequency})

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
    ds : TYPE
        DESCRIPTION.
    variable : str, optional
        Reflectivity field. The default is "zFactorFinal".
    radar_frequency : str, optional
        Radar frequency. The default is "Ku".
    lower_z_threshold : (int, float), optional
        Lower reflectivity threshold. The default is 40 dBZ.
    upper_z_threshold : (int, float), optional
        Upper reflectivity threshold. The default is 50 dBZ.

    Returns
    -------
    da_shi : xr.DataArray
        Severe Hail Index (SHI)

    """
    # Retrieve required DataArrays
    da_z = ds[variable].sel({"radar_frequency": radar_frequency})
    da_t = ds["airTemperature"]
    da_height = ds["height"]
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
    heights_arr = np.asanyarray(ds["range"].data)
    thickness_arr = np.diff(heights_arr)
    thickness_arr = np.append(thickness_arr, thickness_arr[-1])
    thickness_arr = np.broadcast_to(thickness_arr, da_e.shape)

    # Compute SHI
    da_shi_profile = da_e * da_t_weighted * thickness_arr  # 3.45 s per granule
    da_shi_profile = da_shi_profile.where(da_height > da_height_zero_deg)  # 4.5 s per granule
    da_shi = 0.1 * da_shi_profile.sum("range")  # 4.3 s (< 1s with numpy.sum !)

    # Mask where input profile is all Nan
    da_shi = da_shi.where(~da_mask)

    # Add attributes
    da_shi.name = "SHI"
    da_shi.attrs["description"] = "Severe Hail Index "
    da_shi.attrs["units"] = "J/m/s"
    return da_shi


def retrieve_MESH(ds):
    """Retrieve the Maximum Estimated Size of Hail (MESH).

    Also known as the Maximum Expected Hail Size (MEHS).

    The “size” in MESH refers to the maximum diameter (in mm) of a hailstone.
    It's an indicator that transforms SHI into hail size by fitting SHI to a
    chosen percentile of maximum observed hail size (using a power-law)
    """
    da_shi = retrieve_SHI(ds)
    da_mesh = 2.54 * da_shi**0.5
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
    da_height_0 = ds["heightZeroDeg"]
    # Retrieve warning threshold
    da_wt = 57.5 * da_height_0 - 121
    # Retrieve SHI
    da_shi = retrieve_SHI(ds)
    # Retrieve POSH
    da_posh = 29 * np.log(da_shi / da_wt) + 50
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
    da_echo_depth_45_solid = retrieve_EchoDepth(
        ds,
        threshold=45,
        variable=variable,
        radar_frequency=radar_frequency,
        min_threshold=0,
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
