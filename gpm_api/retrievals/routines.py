#!/usr/bin/env python3
"""
Created on Fri Jul 28 16:12:07 2023

@author: ghiggi
"""
import importlib

from gpm_api.io.products import available_products


def _get_module_function_names(module_name):
    """Retrieve all function names within a module."""
    module = importlib.import_module(module_name)
    # Use dir() to get all names defined in the module
    all_names = dir(module)
    # Filter out function names from all_names
    function_names = [name for name in all_names if callable(getattr(module, name))]
    return function_names


def _get_available_retrievals(module_name):
    """Get available retrievals inside a specific module."""
    function_names = _get_module_function_names(module_name)
    retrievals = [s[len("retrieve_") :] for s in function_names if s.startswith("retrieve_")]
    return retrievals


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
            "The xr.Dataset does not have the global attribute 'gpm_api_product' key !"
        )
    product = ds.attrs["gpm_api_product"]
    return product


def available_retrievals(ds):
    """Decode the variables of a given GPM product."""
    # Retrieve products
    product = _infer_product(ds)
    # Define retrievals for 2A-<RADAR> products
    if product in available_products(product_category="RADAR", product_level="2A"):
        module_name = "gpm_api.retrievals.retrieval_2a_radar"
        return _get_available_retrievals(module_name)
    if product in available_products(product_category="PMW", product_level="2A"):
        module_name = "gpm_api.retrievals.retrieval_2a_pmw"
        return _get_available_retrievals(module_name)


def check_retrieval_validity(ds, retrieval):
    """Check retrieval validity."""
    product = ds.attrs["gpm_api_product"]
    valid_retrievals = available_retrievals(ds)
    if valid_retrievals is None:
        raise NotImplementedError(
            f"GPM-API does not yet implements retrievals for product {product}"
        )
    if retrieval not in valid_retrievals:
        raise ValueError(
            f"{retrieval} is an invalid retrieval for {product}. Available retrievals are {valid_retrievals}"
        )


def get_retrieval_variable(ds, retrieval, *args, **kwargs):
    """Compute the requested variable."""
    # Retrieve products
    product = _infer_product(ds)

    # Define retrievals for 2A-<RADAR> products
    if product in available_products(product_category="RADAR", product_level="2A"):
        module_name = "gpm_api.retrievals.retrieval_2a_radar"
        check_retrieval_validity(ds, retrieval)
        return _get_retrieval_function(module_name, retrieval)(ds, *args, **kwargs)
    if product in available_products(product_category="PMW", product_level="2A"):
        module_name = "gpm_api.^retrievals.retrieval_2a_pmw"
        check_retrieval_validity(ds, retrieval)
        return _get_retrieval_function(module_name, retrieval)(ds, *args, **kwargs)
