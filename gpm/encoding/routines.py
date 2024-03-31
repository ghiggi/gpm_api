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
"""This module contains functions to apply the encoding to GPM product variables."""

import importlib

from gpm.io.products import available_products

# - Retrieve encoding from YAML configuration files (if present)
# - Remove source_dtype to facilitate encoding to new netcdf/zarr

# -----------------------------------------------------------------------------.


def _infer_product(ds):
    if "gpm_api_product" not in ds.attrs:
        raise ValueError(
            "The xr.Dataset does not have the global attribute 'gpm_api_product' key !",
        )
    return ds.attrs["gpm_api_product"]


def _get_module_function_names(module_name):
    """Retrieve all function names within a module."""
    module = importlib.import_module(module_name)
    # Use dir() to get all names defined in the module
    all_names = dir(module)
    # Filter out function names from all_names
    return [name for name in all_names if callable(getattr(module, name))]


def _get_encoding_function(module_name):
    """Get the get_encoding_dict function from a specific module."""
    func_name = "get_encoding_dict"
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if func is None or not callable(func):
        raise ValueError(f"get_encoding_dict function not found in the {module_name} module.")
    return func


def get_product_encoding_dict(product):
    """Read encoding dictionary from GPM product YAML file."""
    # Define retrievals for 2A-<RADAR> products
    if product in available_products(product_categories="RADAR", product_levels="2A"):
        module_name = "gpm.dataset.encoding.encode_2a_radar"
        return _get_encoding_function(module_name)()
    return None


def _get_dataset_keys(ds):
    return list(ds.data_vars) + list(ds.coords)


def set_encoding(ds, encoding_dict=None):
    if encoding_dict is None:
        product = _infer_product(ds)
        encoding_dict = get_product_encoding_dict(product)

    if encoding_dict is not None:
        ds_keys = _get_dataset_keys(ds)
        encoding_dict = {k: encoding_dict[k] for k in ds_keys if k in encoding_dict}

        # Set the variable encodings
        for k, encoding in encoding_dict.items():
            ds[k].encoding.update(encoding)

    return ds
