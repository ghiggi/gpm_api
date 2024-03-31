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


def decode_MWprecipSource(da):
    """Decode the IMERG V7 variable MWprecipSource."""
    value_description_dict = {
        0: "no observation",
        1: "TMI",
        3: "AMSR-2",
        4: "SSMI (F13, F14, F15)",
        5: "SSMIS (F16, F17, F18, F19)",
        6: "AMSU-B (NOAA15, NOAA16, NOAA17)",
        7: "MHS (METOPA, METOPB, METOPC, NOAA18, NOAA19)",
        9: "GMI",
        11: "ATMS (NOAA20, NOAA21, NPP)",
        12: "AIRS",
        13: "TOVS",
        14: "CRIS",
        15: "AMSR-E",
        16: "SSMI (F11)",
        20: "SAPHIR",
        21: "ATMS (NOAA21)",
    }
    da.attrs["flag_values"] = list(value_description_dict)
    da.attrs["flag_meanings"] = list(value_description_dict.values())
    return da


def decode_HQprecipSource(da):
    """Decode the IMERG V6 variable HQprecipSource."""
    return decode_MWprecipSource(da)


def decode_precipitationQualityIndex(da):
    """Decode the IMERG V6 and V7 variable precipitationQualityIndex."""
    da.attrs["valid_min"] = 0
    da.attrs["valid_max"] = 1
    return da


def decode_probabilityLiquidPrecipitation(da):
    """Decode the IMERG V6 and V7 variable probabilityLiquidPrecipitation."""
    da.attrs["valid_min"] = 0
    da.attrs["valid_max"] = 100
    return da


def decode_MWobservationTime(da):
    """Decode the IMERG V7 variable MWobservationTime."""
    da.attrs["valid_min"] = 0
    da.attrs["valid_max"] = 29
    return da


def decode_HQobservationTime(da):
    """Decode the IMERG V6 variable HQobservationTime."""
    return decode_MWobservationTime(da)


def decode_MWprecipitation(da):
    """Decode the IMERG V7 variable MWprecipitation."""
    da.attrs["valid_min"] = 0
    da.attrs["valid_max"] = 1000
    return da


def decode_HQprecipitation(da):
    """Decode the IMERG V6 variable HQprecipitation."""
    return decode_MWprecipitation(da)


def decode_IRprecipitation(da):
    """Decode the IMERG V6 and V7 variable IRprecipitation."""
    da.attrs["valid_min"] = 0
    da.attrs["valid_max"] = 1000
    return da


def decode_IRinfluence(da):
    """Decode the IMERG V7 variable IRinfluence."""
    da.attrs["valid_min"] = 0
    da.attrs["valid_max"] = 100
    return da


def decode_IRkalmanFilterWeight(da):
    """Decode the IMERG V6 variable IRkalmanFilterWeight."""
    return decode_IRinfluence(da)


def decode_precipitation(da):
    """Decode the IMERG V7 variable precipitation."""
    da.attrs["valid_min"] = 0
    da.attrs["valid_max"] = 1000
    return da


def decode_precipitationCal(da):
    """Decode the IMERG V6 variable precipitationCal."""
    return decode_precipitation(da)


def decode_precipitationUncal(da):
    """Decode the IMERG V6 and V7 variable precipitationUncal."""
    da.attrs["valid_min"] = 0
    da.attrs["valid_max"] = 1000
    return da


def decode_randomError(da):
    """Decode the IMERG V6 and V7 variable randomError."""
    da.attrs["valid_min"] = 0
    da.attrs["valid_max"] = 1000
    return da


def _get_decoding_function(variable):
    function_name = f"decode_{variable}"
    decoding_function = globals().get(function_name)
    if decoding_function is None or not callable(decoding_function):
        raise ValueError(f"No decoding function found for variable '{variable}'")
    return decoding_function


def decode_product(ds):
    """Decode IMERG products."""
    # Define variables to decode with _decode_<variable> functions
    variables = [
        "randomError",
        "precipitationUncal",
        "precipitation",
        "precipitationCal",
        "IRkalmanFilterWeight",
        "IRinfluence",
        "IRprecipitation",
        "HQprecipitation",
        "MWprecipitation",
        "HQobservationTime",
        "MWobservationTime",
        "probabilityLiquidPrecipitation",
        "precipitationQualityIndex",
        "HQprecipSource",
        "MWprecipSource",
    ]
    # Decode such variables if present in the xarray object
    for variable in variables:
        if variable in ds and not is_dataarray_decoded(ds[variable]):
            with xr.set_options(keep_attrs=True):
                ds[variable] = _get_decoding_function(variable)(ds[variable])
    # Added gpm_api_decoded flag
    return add_decoded_flag(ds, variables=variables)
