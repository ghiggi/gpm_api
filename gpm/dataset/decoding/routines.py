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
"""This module contains functions to apply the products decoding routines."""
import importlib

from gpm.io.products import available_products

# TODO:
# - implement accessor ds.gpm.decode_variable(variable)
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
    if product in available_products(product_categories="RADAR", product_levels="2A"):
        ds = _get_decoding_function("gpm.dataset.decoding.decode_2a_radar")(ds)

    # Decode variables of 2A-<PMW> products
    elif product in available_products(product_categories="PMW", product_levels="2A"):
        ds = _get_decoding_function("gpm.dataset.decoding.decode_2a_pmw")(ds)

    # Decode variables of 1C-<PMW> products
    elif product in available_products(product_categories="PMW", product_levels="1C"):
        ds = _get_decoding_function("gpm.dataset.decoding.decode_1c_pmw")(ds)

    # Decode variables of IMERG products
    elif product in available_products(product_categories="IMERG"):
        ds = _get_decoding_function("gpm.dataset.decoding.decode_imerg")(ds)

    # if ds.attrs.get("TotalQualityCode"):
    #     TotalQualityCode = ds.attrs.get("TotalQualityCode")
    #     ds["TotalQualityCode"] = xr.DataArray(
    #         np.repeat(TotalQualityCode, ds.sizes["along_track"]), dims=["along_track"]
    #     )

    return ds
