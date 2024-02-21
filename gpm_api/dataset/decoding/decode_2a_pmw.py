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


def decode_surfacePrecipitation(da):
    """Decode the 2A-<PMW> variable surfacePrecipitation.

    _FillValue is often reported as -9999.9, but in data the values are -9999.0 !
    """
    da = da.where(da != -9999.0)
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
        "surfacePrecipitation",
    ]
    # Decode such variables if present in the xarray object
    for variable in variables:
        if variable in ds and ds[variable].attrs.get("gpm_api_decoded", "no") != "yes":
            ds[variable] = _get_decoding_function(variable)(ds[variable])
            ds[variable].attrs["gpm_api_decoded"] = "yes"
    return ds
