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
"""This module contains functions to decode GPM PMW 1C products."""
import xarray as xr

from gpm.dataset.decoding.utils import (
    add_decoded_flag,
    is_dataarray_decoded,
)


def decode_Quality(da):
    """Decode the 1C-<PMW> variable Quality."""
    value_description_dict = {
        0: "Good Data",
        1: "Possible sun glint",
        2: "Possible RFI",
        3: "Degraded geolocation",
        4: "Data corrected for warm load instrusion",
        -1: "Data is missing from file or unreadable",
        -2: "Invalid or unphysical brightness temperature",
        -3: "Geolocation error",
        -4: "Missing channel",
        -5: "Multiple channels missing",
        -6: "Lat/Lon values are out of range",
        -7: "Non-normal status modes",
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
    """Decode 1C-<PMW> products."""
    # Define variables to decode with _decode_<variable> functions
    variables = [
        "Quality",
    ]
    # Decode such variables if present in the xarray object
    for variable in variables:
        if variable in ds and not is_dataarray_decoded(ds[variable]):
            with xr.set_options(keep_attrs=True):
                ds[variable] = _get_decoding_function(variable)(ds[variable])
    # Added gpm_api_decoded flag
    return add_decoded_flag(ds, variables=variables)
