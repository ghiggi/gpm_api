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
"""This module contains functions to decode GPM RADAR L1B products."""
import xarray as xr

from gpm.dataset.decoding.utils import (
    add_decoded_flag,
    is_dataarray_decoded,
)


def decode_echoPower(da):
    """Decode the 1B-<RADAR> variable echoPower."""
    da = da.where(da > -29999)  # Set -29999 (Outside observation area) and -30000 to  NaN
    da = da / 100
    da.attrs["units"] = "dBm"
    return da


def decode_noisePower(da):
    """Decode the 2A-<RADAR> variable noisePower."""
    da = da.where(da > -29999)  # Set -30000 to NaN
    da = da / 100
    da.attrs["units"] = "dBm"
    return da


def decode_landOceanFlag(da):
    """Decode the 1B-<RADAR> variable landOceanFlag."""
    da = da.where(da >= 0)  # < 0 set to np.nan
    da.attrs["flag_values"] = [0, 1, 2, 3]
    da.attrs["flag_meanings"] = ["Ocean", "Land", "Coast", "Inland Water"]
    da.attrs["description"] = "Land Surface type"
    return da


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
        "noisePower",
        "echoPower",
        "landOceanFlag",
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
    return ds
