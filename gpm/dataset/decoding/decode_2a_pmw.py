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
"""This module contains functions to decode GPM PMW 2A products."""
import xarray as xr

from gpm.dataset.decoding.utils import (
    add_decoded_flag,
    is_dataarray_decoded,
)


def decode_surfacePrecipitation(da):
    """Decode the 2A-<PMW> variable surfacePrecipitation.

    _FillValue is often reported as -9999.9, but in data the values are -9999.0 !
    """
    return da.where(da != -9999.0)


def decode_rainWaterPath(da):
    """Decode the 2A-<PMW> variable rainWaterPath.

    _FillValue is often reported as -9999.9, but in data the values are -9999.0 !
    """
    da = da.where(da >= 0)  # < 0 set to np.nan
    da.attrs["description"] = "Total integrated rain water in the vertical atmospheric column"
    return da


def decode_cloudWaterPath(da):
    """Decode the 2A-<PMW> variable cloudWaterPath.

    _FillValue is often reported as -9999.9, but in data the values are -9999.0 !
    """
    da = da.where(da >= 0)  # < 0 set to np.nan
    da.attrs["description"] = "Total integrated cloud liquid water in the vertical atmospheric column"
    return da


def decode_iceWaterPath(da):
    """Decode the 2A-<PMW> variable iceWaterPath.

    _FillValue is often reported as -9999.9, but in data the values are -9999.0 !
    """
    da = da.where(da >= 0)  # < 0 set to np.nan
    da.attrs["description"] = "Total integrated ice water in the vertical atmospheric column"
    return da


def decode_sunGlintAngle(da):
    """Decode the 2A-<PMW> variable sunGlintAngle.

    Set -88 value (sun below horizon) to np.nan
    """
    return da.where(da >= 0)  # < 0 set to np.nan


def decode_airmassLiftIndex(da):
    """Decode the 2A-<PMW> variable airmassLiftIndex."""
    product = da.attrs["gpm_api_product"]
    if "CLIM" in product:
        value_description_dict = {
            0: "No orographic moisture enhancement, stratiform",
            1: "Orographic moisture enhancement, stratiform",
            2: "No orographic moisture enhancement, convective",
            3: "Orographic moisture enhancement, convective",
        }
        da.attrs["flag_values"] = list(value_description_dict)
        da.attrs["flag_meanings"] = list(value_description_dict.values())
    else:
        value_description_dict = {
            0: "No orographic moisture enhancement",
            1: "Orographic moisture enhancement",
        }
        da.attrs["flag_values"] = list(value_description_dict)
        da.attrs["flag_meanings"] = list(value_description_dict.values())
    return da


def decode_surfaceTypeIndex(da):
    """Decode the 2A-<PMW> variable surfaceTypeIndex."""
    value_description_dict = {
        1: "Ocean",
        2: "Sea-Ice",
        3: "High vegetation",
        4: "Medium vegetation",
        5: "Low vegetation",
        6: "Sparse vegetation",
        7: "Desert",
        8: "Elevated snow cover",
        9: "High snow cover",
        10: "Moderate snow cover",
        11: "Light snow cover",
        12: "Standing Water",
        13: "Ocean or water Coast",
        14: "Mixed land/ocean or water coast",
        15: "Land coast",
        16: "Sea-ice edge",
        17: "Mountain rain",
        18: "Mountain snow",
    }
    da.attrs["flag_values"] = list(value_description_dict)
    da.attrs["flag_meanings"] = list(value_description_dict.values())
    da.attrs["description"] = "Surface type"
    return da


def decode_precipitationYesNoFlag(da):
    """Decode the 2A-<PMW> variable precipitationYesNoFlag.

    _FillValue is reported as -9999.0, but in data the values are -99. !
    """
    da = da.where(da >= 0)  # < 0 set to np.nan
    da.attrs["flag_values"] = [0, 1]
    da.attrs["flag_meanings"] = ["non-raining", "raining"]
    da.attrs["description"] = "Precipitation Flag"
    return da


def decode_precip1stTertial(da):
    """Decode the 2A-<PMW> variable precip1stTertial."""
    da.attrs["description"] = "33.33 percentile of the precipitation distribution"
    return da


def decode_precip2ndTertial(da):
    """Decode the 2A-<PMW> variable precip2ndTertial."""
    da.attrs["description"] = "66.66 percentile of the precipitation distribution"
    return da


def decode_pixelStatus(da):
    """Decode the 2A-<PMW> variable pixelStatus."""
    value_description_dict = {
        0: "Valid pixel",
        1: "Invalid Latitude / Longitude",
        2: "Channel Tbs out of range",
        3: "Surface code / histogram mismatch",
        4: "Missing TCWV, T2m, or sfccode from preprocessor",
        5: "No Bayesian Solution",
    }
    da.attrs["flag_values"] = list(value_description_dict)
    da.attrs["flag_meanings"] = list(value_description_dict.values())
    return da


def decode_qualityFlag(da):
    """Decode the 2A-<PMW> variable qualityFlag."""
    value_description_dict = {
        0: "Good",
        1: "Use with caution",
        2: "Use with extreme caution (snow-covered)",
        3: "Use with extreme caution (missing channels).",
    }
    da.attrs["flag_values"] = list(value_description_dict)
    da.attrs["flag_meanings"] = list(value_description_dict.values())
    return da


def _get_decoding_function(variable):
    function_name = f"decode_{variable}"
    decoding_function = globals().get(function_name)
    if decoding_function is None or not callable(decoding_function):
        raise ValueError(f"No decoding function found for variable '{variable}'")
    return decoding_function


def decode_product(ds):
    """Decode 2A-<PMW> products."""
    # Define variables to decode with _decode_<variable> functions
    variables = [
        "pixelStatus",
        "qualityFlag",
        "rainWaterPath",
        "cloudWaterPath",
        "rainWaterPath",
        "airmassLiftIndex",
        "surfaceTypeIndex",
        "surfacePrecipitation",
        "sunGlintAngle",
        "precipitationYesNoFlag",
        "precip1stTertial",
        "precip2ndTertial",
    ]
    # Decode such variables if present in the xarray object
    for variable in variables:
        if variable in ds and not is_dataarray_decoded(ds[variable]):
            with xr.set_options(keep_attrs=True):
                ds[variable] = _get_decoding_function(variable)(ds[variable])
    # Added gpm_api_decoded flag
    return add_decoded_flag(ds, variables=variables)
