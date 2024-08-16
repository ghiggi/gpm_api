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
"""This module contains functions to decode GPM DPR, PR, Ka and Ku products."""
import xarray as xr

from gpm.dataset.decoding.utils import (
    add_decoded_flag,
    ceil_dataarray,
    is_dataarray_decoded,
    remap_numeric_array,
)


def decode_landSurfaceType(da):
    """Decode the 2A-<RADAR> variable landSurfaceType."""
    da = da.where(da >= 0)  # < 0 set to np.nan
    da = da / 100
    da = ceil_dataarray(da)
    value_dict = {
        0: "Ocean",
        1: "Land",
        2: "Coast",
        3: "Inland Water",
    }
    da.attrs["flag_values"] = list(value_dict)
    da.attrs["flag_meanings"] = list(value_dict.values())
    da.attrs["description"] = "Land Surface Type"
    return da


def decode_snowIceCover(da):
    """Decode the 2A-<RADAR> variable snowIceCover."""
    da = da.where(da >= 0)  # -99 is set to np.nan
    value_dict = {
        0: "Open water",
        1: "Snow-free land",
        2: "Snow-covered land",
        3: "Sea ice",
    }
    da.attrs["flag_values"] = list(value_dict)
    da.attrs["flag_meanings"] = list(value_dict.values())
    da.attrs["description"] = "Snow/Ice Cover"
    return da


def decode_phase(da):
    """Decode the 2A-<RADAR> variable phase."""
    da = da / 100
    da = ceil_dataarray(da)
    da = da.where(da >= 0)  # < 0 set to np.nan
    da.attrs["flag_values"] = [0, 1, 2]
    da.attrs["flag_meanings"] = ["solid", "mixed_phase", "liquid"]
    da.attrs["description"] = "Precipitation Phase State"
    return da


def decode_phaseNearSurface(da):
    """Decode the 2A-<RADAR> variable phaseNearSurface."""
    da = da / 100
    da = ceil_dataarray(da)
    da = da.where(da >= 0)  # < 0 set to np.nan
    da.attrs["flag_values"] = [0, 1, 2]
    da.attrs["flag_meanings"] = ["solid", "mixed_phase", "liquid"]
    da.attrs["description"] = "Precipitation phase state near the surface"
    return da


def decode_flagPrecip(da):
    """Decode the 2A-<RADAR> variable flagPrecip."""
    if da.attrs["gpm_api_product"] == "2A-DPR":
        # V7
        value_dict = {
            0: "not detected by both Ku and Ka",
            1: "detected by Ka only",
            2: "detected by Ka only",
            10: "detected by Ku only",
            11: "detected by both Ku and Ka",
            12: "detected by both Ku and Ka",
            20: "detected by both Ku and Ka",
            21: "detected by both Ku and Ka",
            22: "detected by both Ku and Ka",
        }
        # V6
        # value_dict = {
        #     0: "not detected by both Ku and Ka",
        #     1: "detected by Ka only",
        #     10: "detected by Ku only",
        #     11: "detected by both Ku and Ka",
        # }
        da.attrs["flag_values"] = list(value_dict)
        da.attrs["flag_meanings"] = list(value_dict.values())
    else:
        da.attrs["flag_values"] = [0, 1]
        da.attrs["flag_meanings"] = ["not detected", "detected"]
    da.attrs["description"] = "Flag for precipitation detection"
    return da


def decode_qualityFlag(da):
    """Decode the 2A-<RADAR> variable qualityFlag."""
    da = da.where(da >= 0)  # < 0 (-99) set to np.nan
    value_dict = {
        0: "High quality. No issues.",
        1: "Low quality (warnings during retrieval)",
        2: "Bad (errors during retrieval)",
    }
    da.attrs["flag_values"] = list(value_dict)
    da.attrs["flag_meanings"] = list(value_dict.values())
    da.attrs["description"] = "Flag for precipitation detection"
    return da


def decode_qualityTypePrecip(da):
    """Decode the 2A-<RADAR> variable qualityTypePrecip."""
    da = da.where(da > 0, 0)
    da.attrs["flag_values"] = [0, 1]
    da.attrs["flag_meanings"] = ["no rain", "good"]
    da.attrs["description"] = "Quality of the precipitation type"
    return da


def decode_qualityBB(da):
    """Decode the 2A-<RADAR> variable qualityBB."""
    da = da.where(da >= 0)  # < -1111 and -9999 set to np.nan
    da.attrs["flag_values"] = [0, 1]
    da.attrs["flag_meanings"] = ["BB not detected", "BB detected"]
    da.attrs["description"] = "Quality of Bright-Band Detection"
    return da


def decode_reliabFlag(da):
    """Decode the 2A-<RADAR> variable reliabFlag."""
    da = da.where(da >= 0)
    da.attrs["flag_values"] = [1, 2, 3, 4]
    da.attrs["flag_meanings"] = ["reliable", "marginally reliable", "unreliable", "lower-bound"]
    da.attrs["description"] = "Reliability flag for the effective PIA estimate (pathAtten)"
    return da


def decode_flagShallowRain(da):
    """Decode the 2A-<RADAR> variable flagShallowRain."""
    da = da.where(da >= 0)  # -11111 is set to np.nan
    remapping_dict = {0: 0, 10: 1, 11: 2, 20: 3, 21: 4}
    da.data = remap_numeric_array(da.data, remapping_dict)
    value_dict = {
        0: "No shallow rain",
        1: "Shallow isolated (maybe)",
        2: "Shallow isolated (certain)",
        3: "Shallow non-isolated (maybe)",
        4: "Shallow non-isolated (certain)",
    }
    da.attrs["flag_values"] = list(value_dict)
    da.attrs["flag_meanings"] = list(value_dict.values())
    da.attrs["description"] = "Type of shallow rain"
    return da


def decode_flagHeavyIcePrecip(da):
    """Decode the 2A-<RADAR> variable flagHeavyIcePrecip."""
    da = da.where(da >= 1)  # make 0 nan
    da.attrs["flag_values"] = [4, 8, 12, 16, 24, 32, 40]
    da.attrs["flag_meanings"] = [""] * 6  # TODO
    da.attrs[
        "description"
    ] = """Flag for detection of strong or severe precipitation accompanied
    by solid ice hydrometeors above the -10 degree C isotherm"""
    return da


def decode_flagAnvil(da):
    """Decode the 2A-<RADAR> variable flagAnvil."""
    da = da.where(da >= 1)  # make 0 nan
    da.attrs["flag_values"] = [0, 1, 2]  # TODO: 2 is unknown
    da.attrs["flag_meanings"] = ["not detected", "detected", "unknown"]
    da.attrs["description"] = "Flago for anvil detection by the Ku-band radar"
    return da


def decode_zFactorMeasured(da):
    """Decode the 2A-<RADAR> variable flagBB."""
    return da.where(da >= -80)  # Make -29999 and -28888 -> NaN


def decode_attenuationNP(da):
    """Decode the 2A-<RADAR> variable flagBB."""
    return da.where(da >= 0)  # Make -19999.8 -> NaN


def decode_flagBB(da):
    """Decode the 2A-<RADAR> variable flagBB."""
    da = da.where(da >= 0)  # make -1111 (no rain) nan
    if da.attrs["gpm_api_product"] == "2A-DPR":
        da.attrs["flag_values"] = [0, 1, 2, 3]
        da.attrs["flag_meanings"] = [
            "not detected",
            "detected by Ku and DFRm",
            "detected by Ku only",
            "detected by DFRm only",
        ]
    else:
        da.attrs["flag_values"] = [0, 1]
        da.attrs["flag_meanings"] = ["not detected", "detected"]

    da.attrs["description"] = "Flag for bright band detection"
    return da


def decode_flagSurfaceSnowfall(da):
    """Decode the 2A-<RADAR> variable flagSurfaceSnowfall."""
    da.attrs["flag_values"] = [0, 1]
    da.attrs["flag_meanings"] = ["no snow-cover", "snow-cover"]
    da.attrs["description"] = "Flag for snow-cover"
    return da


def decode_flagHail(da):
    """Decode the 2A-<RADAR> variable flagHail."""
    da.attrs["flag_values"] = [0, 1]
    da.attrs["flag_meanings"] = ["not detected", "detected"]
    da.attrs["description"] = "Flag for hail detection"
    return da


def decode_flagGraupelHail(da):
    """Decode the 2A-<RADAR> variable flagGraupelHail."""
    da.attrs["flag_values"] = [0, 1]
    da.attrs["flag_meanings"] = ["not detected", "detected"]
    da.attrs["description"] = "Flag for graupel hail detection "
    return da


def decode_widthBB(da):
    """Decode the 2A-<RADAR> variable widthBB."""
    return da.where(da >= 0)  # -1111.1 is set to np.nan


def decode_heightBB(da):
    """Decode the 2A-<RADAR> variable heightBB."""
    return da.where(da >= 0)  # -1111.1 is set to np.nan


def _get_decoding_function(variable):
    function_name = f"decode_{variable}"
    decoding_function = globals().get(function_name)
    if decoding_function is None or not callable(decoding_function):
        raise ValueError(f"No decoding function found for variable '{variable}'")
    return decoding_function


def decode_product(ds):
    """Decode 2A-<RADAR> products."""
    # Define variables to decode with _decode_<variable> functions
    variables = [
        "flagBB",
        "flagShallowRain",
        "flagAnvil",
        "flagHeavyIcePrecip",
        "flagHail",
        "flagGraupelHail",
        "flagSurfaceSnowfall",
        "widthBB",
        "heightBB",
        "qualityFlag",
        "qualityTypePrecip",
        "qualityBB",
        "flagPrecip",
        "phaseNearSurface",
        "phase",
        "reliabFlag",
        "landSurfaceType",
        "snowIceCover",
        "attenuationNP",
        "zFactorMeasured",
    ]
    # Decode such variables if present in the xarray object
    for variable in variables:
        if variable in ds and not is_dataarray_decoded(ds[variable]):
            with xr.set_options(keep_attrs=True):
                ds[variable] = _get_decoding_function(variable)(ds[variable])

    # Decode bin variables (set 0, -1111 and other invalid values to np.nan)
    for variable in ds.gpm.bin_variables:
        ds[variable] = ds[variable].where(ds[variable] > 0)

    # Added gpm_api_decoded flag
    ds = add_decoded_flag(ds, variables=variables + ds.gpm.bin_variables)

    #### Preprocess other variables
    #### - precipWaterIntegrated
    if "precipWaterIntegrated" in ds and not is_dataarray_decoded(ds["precipWaterIntegrated"]):
        # Extract variables
        ds["precipWaterIntegrated"] = ds["precipWaterIntegrated"] / 1000
        ds["precipWaterIntegrated_Liquid"] = ds["precipWaterIntegrated"].isel({"LS": 0})
        ds["precipWaterIntegrated_Solid"] = ds["precipWaterIntegrated"].isel({"LS": 1})
        ds = ds.drop_vars(names="precipWaterIntegrated")
        ds["precipWaterIntegrated"] = ds["precipWaterIntegrated_Liquid"] + ds["precipWaterIntegrated_Solid"]
        # Add units
        variables = [
            "precipWaterIntegrated_Liquid",
            "precipWaterIntegrated_Solid",
            "precipWaterIntegrated",
        ]
        for var in variables:
            ds[var].attrs["units"] = "kg/m^2"

        # Add GPM-API decoded flag
        ds = add_decoded_flag(ds, variables=variables)

    #### - paramDSD
    if "paramDSD" in ds and not is_dataarray_decoded(ds["paramDSD"]):
        # Extract variables
        da = ds["paramDSD"]
        ds["dBNw"] = da.isel(DSD_params=0)
        ds["Dm"] = da.isel(DSD_params=1)
        ds["Nw"] = 10 ** (ds["dBNw"] / 10)
        # Add units
        ds["Dm"].attrs["units"] = "mm"
        ds["Dm"].attrs["long_name"] = "Mass weighted mean diameter"
        ds["dBNw"].attrs["units"] = "10log10(1/(mm*m^3))"
        ds["Nw"].attrs["units"] = "1/(mm*m^3)"
        # Add gpm_api_decoded flag
        ds = add_decoded_flag(ds, variables=["dBNw", "Dm", "Nw"])
        # Drop unused variable
        ds = ds.drop_vars("paramDSD")
    return ds
