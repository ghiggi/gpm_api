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
    ceil_datarray,
    is_dataarray_decoded,
    remap_numeric_array,
)


def decode_landSurfaceType(da):
    """Decode the 2A-<RADAR> variable landSurfaceType."""
    da = da.where(da >= 0)  # < 0 set to np.nan
    da = da / 100
    da = ceil_datarray(da)
    da.attrs["flag_values"] = [0, 1, 2, 3]
    da.attrs["flag_meanings"] = ["Ocean", "Land", "Coast", "Inland Water"]
    da.attrs["description"] = "Land Surface type"
    return da


def decode_phase(da):
    """Decode the 2A-<RADAR> variable phase."""
    da = da / 100
    da = ceil_datarray(da)
    da = da.where(da >= 0)  # < 0 set to np.nan
    da.attrs["flag_values"] = [0, 1, 2]
    da.attrs["flag_meanings"] = ["solid", "mixed_phase", "liquid"]
    da.attrs["description"] = "Precipitation phase state"
    return da


def decode_phaseNearSurface(da):
    """Decode the 2A-<RADAR> variable phaseNearSurface."""
    da = da / 100
    da = ceil_datarray(da)
    da = da.where(da >= 0)  # < 0 set to np.nan
    da.attrs["flag_values"] = [0, 1, 2]
    da.attrs["flag_meanings"] = ["solid", "mixed_phase", "liquid"]
    da.attrs["description"] = "Precipitation phase state near the surface"
    return da


def decode_flagPrecip(da):
    """Decode the 2A-<RADAR> variable flagPrecip."""
    if da.attrs["gpm_api_product"] == "2A-DPR":
        da.attrs["flag_values"] = [0, 1, 10, 11]
        # TODO: 2, 10, 12, 20, 21, 22 values also present
        da.attrs["flag_meanings"] = [
            "not detected by both Ku and Ka",
            "detected by Ka only",
            "detected by Ku only",
            "detected by both Ku and Ka",
        ]
    else:
        da.attrs["flag_values"] = [0, 1]
        da.attrs["flag_meanings"] = ["not detected", "detected"]
    da.attrs["description"] = "Flag for precipitation detection"
    return da


def decode_qualityTypePrecip(da):
    """Decode the 2A-<RADAR> variable qualityTypePrecip."""
    da = da.where(da > 0, 0)
    da.attrs["flag_values"] = [0, 1]
    da.attrs["flag_meanings"] = ["no rain", "good"]
    da.attrs["description"] = "Quality of the precipitation type"
    return da


def decode_flagShallowRain(da):
    """Decode the 2A-<RADAR> variable flagShallowRain."""
    da = da.where(da > -1112, 0)
    remapping_dict = {-1111: 0, 0: 1, 10: 2, 11: 3, 20: 4, 21: 5}
    da.data = remap_numeric_array(da.data, remapping_dict)  # TODO
    da.attrs["flag_values"] = list(remapping_dict.values())
    da.attrs["flag_meanings"] = [
        "no rain",
        "no shallow rain",
        "Shallow isolated (maybe)",
        "Shallow isolated (certain)",
        "Shallow non-isolated (maybe)",
        "Shallow non-isolated (certain)",
    ]
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


def decode_binBBPeak(da):
    """Decode the 2A-<RADAR> variable binBBPeak."""
    return da.where(da >= 0)  # -1111 is set to np.nan


def decode_binBBTop(da):
    """Decode the 2A-<RADAR> variable binBBTop."""
    return da.where(da >= 0)  # -1111 is set to np.nan


def decode_binBBBottom(da):
    """Decode the 2A-<RADAR> variable binBBBottom."""
    return da.where(da >= 0)  # -1111 is set to np.nan


def decode_binDFRmMLBottom(da):
    """Decode the 2A-<RADAR> variable binDFRmMLBottom."""
    return da.where(da >= 0)  # -1111 is set to np.nan


def decode_binDFRmMLTop(da):
    """Decode the 2A-<RADAR> variable binDFRmMLTop."""
    return da.where(da >= 0)  # -1111 is set to np.nan


def decode_binHeavyIcePrecipTop(da):
    """Decode the 2A-<RADAR> variable binHeavyIcePrecipTop."""
    return da.where(da >= 0)  # -1111 is set to np.nan


def decode_binHeavyIcePrecipBottom(da):
    """Decode the 2A-<RADAR> variable binHeavyIcePrecipBottom."""
    return da.where(da >= 0)  # -1111 is set to np.nan


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
        # "flagShallowRain",
        "flagAnvil",
        "flagHeavyIcePrecip",
        "flagHail",
        "flagGraupelHail",
        "flagSurfaceSnowfall",
        "widthBB",
        "heightBB",
        "binBBPeak",
        "binBBTop",
        "binBBBottom",
        "binDFRmMLBottom",
        "binDFRmMLTop",
        "binHeavyIcePrecipTop",
        "binHeavyIcePrecipBottom",
        "qualityTypePrecip",
        "flagPrecip",
        "phaseNearSurface",
        "phase",
        "landSurfaceType",
        "attenuationNP",
        "zFactorMeasured",
    ]
    # Decode such variables if present in the xarray object
    for variable in variables:
        if variable in ds and not is_dataarray_decoded(ds[variable]):
            with xr.set_options(keep_attrs=True):
                ds[variable] = _get_decoding_function(variable)(ds[variable])

    # Added gpm_api_decoded flag
    ds = add_decoded_flag(ds, variables=variables)

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
