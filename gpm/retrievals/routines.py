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
"""This module contains functions to search the GPM-API community-based retrivals."""
import importlib

from gpm.io.products import available_products


def _get_module_function_names(module_name):
    """Retrieve all function names within a module."""
    module = importlib.import_module(module_name)
    # Use dir() to get all names defined in the module
    all_names = dir(module)
    # Filter out function names from all_names
    return [name for name in all_names if callable(getattr(module, name))]


def _get_available_retrievals(module_name):
    """Get available retrievals inside a specific module."""
    function_names = _get_module_function_names(module_name)
    return [s[len("retrieve_") :] for s in function_names if s.startswith("retrieve_")]


def _get_retrieval_function(module_name, retrieval):
    """Get the retrieval function from a specific module."""
    func_name = f"retrieve_{retrieval}"
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise ValueError(f"{func_name} function not found in the {module_name} module.")
    return func


def _infer_product(ds):
    if "gpm_api_product" not in ds.attrs:
        raise ValueError(
            "The xr.Dataset does not have the global attribute 'gpm_api_product' key !",
        )
    return ds.attrs["gpm_api_product"]


def available_retrievals(ds):
    """Decode the variables of a given GPM product."""
    # Retrieve products
    product = _infer_product(ds)
    # Define retrievals for 2A-<RADAR> products
    if product in available_products(product_categories="RADAR", product_levels="2A"):
        module_name = "gpm.retrievals.retrieval_2a_radar"
        return _get_available_retrievals(module_name)
    if product in available_products(product_categories="PMW", product_levels="2A"):
        module_name = "gpm.retrievals.retrieval_2a_pmw"
        return _get_available_retrievals(module_name)
    return None


def check_retrieval_validity(ds, retrieval):
    """Check retrieval validity."""
    product = ds.attrs["gpm_api_product"]
    valid_retrievals = available_retrievals(ds)
    if valid_retrievals is None:
        raise NotImplementedError(
            f"GPM-API does not yet implements retrievals for product {product}",
        )
    if retrieval not in valid_retrievals:
        raise ValueError(
            f"{retrieval} is an invalid retrieval for {product}. Available retrievals are {valid_retrievals}",
        )


def get_retrieval_variable(ds, name, *args, **kwargs):
    """Compute the requested variable."""
    # Retrieve products
    product = _infer_product(ds)

    # Define retrievals for 2A-<RADAR> products
    if product in available_products(product_categories="RADAR", product_levels="2A"):
        module_name = "gpm.retrievals.retrieval_2a_radar"
        check_retrieval_validity(ds, name)
        return _get_retrieval_function(module_name, name)(ds, *args, **kwargs)
    if product in available_products(product_categories="PMW", product_levels="2A"):
        module_name = "gpm.retrievals.retrieval_2a_pmw"
        check_retrieval_validity(ds, name)
        return _get_retrieval_function(module_name, name)(ds, *args, **kwargs)
    return None
