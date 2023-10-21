#!/usr/bin/env python3
"""
Created on Fri Jul 28 13:46:01 2023

@author: ghiggi
"""
import importlib

from gpm_api.io.products import available_products

# TODO:
# - implement accessor ds.gpm_api.decode_variable(variable)
# - use gpm_api_product attribute to recognize where decode_<variable> function is
# - run decoding only if gpm_api_decoded not present (or "no"). If "yes", raise error (already decoded)


def _get_decoding_function(module_name):
    """Retrieve the decode_product function from a specific module."""
    module = importlib.import_module(module_name)
    decode_function = getattr(module, "decode_product", None)

    if decode_function is None or not callable(decode_function):
        raise ValueError("decode_product function not found in the {module_name} module.")

    return decode_function


def decode_variables(ds, product):
    """Decode the variables of a given GPM product."""
    # Decode variables of 2A-<RADAR> products
    if product in available_products(product_category="RADAR", product_level="2A"):
        ds = _get_decoding_function("gpm_api.dataset.decoding.decode_2a_radar")(ds)

    # Decode variables of 2A-<PMW> products
    if product in available_products(product_category="PMW", product_level="2A"):
        ds = _get_decoding_function("gpm_api.dataset.decoding.decode_2a_pmw")(ds)

    # if ds.attrs.get("TotalQualityCode"):
    #     TotalQualityCode = ds.attrs.get("TotalQualityCode")
    #     ds["TotalQualityCode"] = xr.DataArray(
    #         np.repeat(TotalQualityCode, ds.dims["along_track"]), dims=["along_track"]
    #     )

    return ds
