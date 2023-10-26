#!/usr/bin/env python3
"""
Created on Thu Aug  3 14:01:37 2023

@author: ghiggi
"""
import importlib

from gpm_api.io.products import available_products

# - Retrieve encoding from YAML configuration files (if present)
# - Remove source_dtype to facilitate encoding to new netcdf/zarr

# -----------------------------------------------------------------------------.


def _infer_product(ds):
    if "gpm_api_product" not in ds.attrs:
        raise ValueError(
            "The xr.Dataset does not have the global attribute 'gpm_api_product' key !"
        )
    product = ds.attrs["gpm_api_product"]
    return product


def _get_module_function_names(module_name):
    """Retrieve all function names within a module."""
    module = importlib.import_module(module_name)
    # Use dir() to get all names defined in the module
    all_names = dir(module)
    # Filter out function names from all_names
    function_names = [name for name in all_names if callable(getattr(module, name))]
    return function_names


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
    if product in available_products(product_category="RADAR", product_level="2A"):
        module_name = "gpm_api.dataset.encoding.encode_2a_radar"
        return _get_encoding_function(module_name)()
    else:
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
