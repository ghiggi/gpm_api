#!/usr/bin/env python3
"""
Created on Fri Jul 28 12:28:40 2023

@author: ghiggi
"""


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
