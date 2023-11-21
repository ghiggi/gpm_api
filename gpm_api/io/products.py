#!/usr/bin/env python3
"""
Created on Thu Oct 13 11:13:15 2022

@author: ghiggi
"""
import datetime
import functools
import os

from gpm_api.io.checks import (
    check_product_category,
    check_product_level,
    check_product_type,
    check_product_validity,
    check_product_version,
    check_version,
)
from gpm_api.utils.yaml import read_yaml_file

### Notes

# product
# - PMW: <product_level>-<sensor>-<satellite> # 2A-AMSUB-NOAA15, # 2A-AMSUB-NOAA15_CLIM
# - RADAR: <product_level>-<....>
# - IMERG: IMERG-run_type>  # IMERG-LR
# product_level:
# - IMERG has no product level


# info_dict
# - start_time  (for time period of product)
# - end_time    (for time period of product)

# - satellite   ?   --> not for IMERG
# - product_level ?  --> can be inferred by gpm-api product name also ...

# scan_modes
# "1A-GMI": has also S3 and gmi1aHeader but can't be processed
# "1A-TMI": has also tmi1aHeader but can't be processed

### Product changes:
# In V05, GPROF products are available as 2A-CLIM and 2A
# In V06, GPROF is not available !
# In V07 RS, GPROF is available only as 2A-CLIM (except for 2A.GPM.GMI) on PPS
# In V07 RS, GPROF is available (mostly) as 2A-CLIM and 2A on GES DISC
# --> AMSUB and SSMI (F08-15) only with CLIM product on GES DISC
# In V07 NRT, GPROF is available only as 2A

# PRPS is available only in V06 (CLIM and no CLIM) (PMW SAPHIR)

# The '2A-CLIM' products differ from the '2A' by the ancillary data they use.
# '2A-CLIM' use ERA-Interim, 2A uses GANAL renalysis ?
# In 2A-CLIM the GPROF databases are also adjusted accordingly for these climate-referenced retrieval.


####--------------------------------------------------------------------------.


@functools.lru_cache(maxsize=None)
def get_info_dict():
    """Get product info dictionary."""
    from gpm_api import _root_path

    fpath = os.path.join(_root_path, "gpm_api", "etc", "product_def.yml")
    return read_yaml_file(fpath)


@functools.lru_cache(maxsize=None)
def get_product_info(product):
    """Provide the product info dictionary."""
    if not isinstance(product, str):
        raise TypeError("'product' must be a string.")
    info_dict = get_info_dict()
    valid_products = list(get_info_dict())
    if product not in valid_products:
        raise ValueError("Please provide a valid GPM product --> gpm_api.available_products().")
    product_info = info_dict[product]
    return product_info


def available_versions(product):
    """Provides a list with the available product versions."""
    versions = get_product_info(product)["available_versions"]
    return versions


def get_last_product_version(product):
    """Provide the most recent product version."""
    version = available_versions(product)[-1]
    return version


def get_product_start_time(product):
    """Provide the product start_time."""
    start_time = get_product_info(product)["start_time"]
    return start_time


def get_product_end_time(product):
    """Provide the product end_time."""
    end_time = get_info_dict()[product]["end_time"]
    if end_time is None:
        end_time = datetime.datetime.utcnow()
    return end_time


def available_products(
    product_type=None,
    product_category=None,
    product_level=None,
    version=None,
):
    """
    Provide a list of all/NRT/RS GPM data for download.

    Parameters
    ----------
    product_type : str, optional
        If None (default), provide all products (RS and NRT).
        If 'RS', provide a list of all GPM RS products available for download.
        If 'NRT', provide a list of all GPM NRT products available for download.
    product_category: (str or list), optional
        If None (default), provide products from all product categories.
        If string, must be a valid product category.
        Valid product categories are: 'PMW', 'RADAR', 'IMERG', 'CMB'.
    product_category: (str or list), optional
        If None (default), provide products from all product levels.
        If string, must be a valid product level.
        Valid product levels are: '1A','1B','1C','2A','2B'.
        For IMERG products, no product level applies.

    Returns
    -------
    List
        List of GPM available products.

    """
    info_dict = get_info_dict()
    products = list(info_dict)
    if product_type is not None:
        check_product_type(product_type)
        products = [
            product for product in products if product_type in info_dict[product]["product_types"]
        ]

    if product_category is not None:
        check_product_category(product_category)
        products = [
            product
            for product in products
            if product_category == info_dict[product]["product_category"]
        ]

    if product_level is not None:
        check_product_level(product_level)
        products = [product for product in products if product_level == product[0:2]]

    if version is not None:
        check_version(version)
        products = [
            product for product in products if version in info_dict[product]["available_versions"]
        ]

    return sorted(products)


def available_scan_modes(product, version):
    """Return the available scan_modes for a given product (and specific version)."""
    product_info = get_product_info(product)
    version = check_product_version(version, product)
    check_product_validity(product)
    scan_modes = product_info["scan_modes"]["V" + str(version)]
    return scan_modes


def get_products_pattern_dict():
    """Return the filename pattern* associated to all GPM products."""
    info_dict = get_info_dict()
    products = available_products()
    pattern_dict = {product: info_dict[product]["pattern"] for product in products}
    return pattern_dict


def get_product_pattern(product):
    """Return the filename pattern associated to GPM product."""
    info_dict = get_info_dict()
    pattern = info_dict[product]["pattern"]
    return pattern


def get_product_category(product):
    """Get the product_category of a GPM product.

    The product_category is used to organize file on disk.
    """
    product_category = get_product_info(product).get("product_category", None)
    if product_category is None:
        raise ValueError(
            f"The product_category for {product} product is not specified in the config files"
        )
    return product_category


def get_product_level(product, short=False):
    """Get the product_level of a GPM product."""
    # TODO: 2A-CLIM, 2H (CSH, LSH), 3B-HH3, 3B-DAY
    # TODO: Add product level into product.yaml ?
    pattern = get_product_info(product).get("pattern", None).split(".")[0]
    if "GPMCOR" in pattern:
        product_level = "1B"
    else:
        product_level = pattern
    if short:
        product_level = product_level[0:2]
    return product_level


def is_trmm_product(product):
    """Check if the product arises from the TRMM satellite."""
    trmm_products = [
        "1A-TMI",
        "1B-TMI",
        "1C-TMI",
        "1B-PR",
        "2A-TMI",
        "2A-TMI-CLIM",
        "2A-PR",
        "2A-ENV-PR",  # ENV not available on GES DISC
        "2B-TRMM-CSH",
        "2A-TRMM-SLH",
        "2B-TRMM-CORRA",
    ]
    if product in trmm_products:
        return True
    else:
        return False


def is_gpm_product(product):
    """Check if the product arises from the GPM Core satellite."""
    gpm_products = [
        "1A-GMI",
        "1B-GMI",
        "1C-GMI",
        "1B-Ka",
        "1B-Ku",
        "2A-GMI",
        "2A-GMI-CLIM",
        "2A-DPR",
        "2A-Ka",
        "2A-Ku",
        "2A-ENV-DPR",
        "2A-ENV-Ka",
        "2A-ENV-Ku" "2B-GPM-CSH",
        "2A-GPM-SLH",
        "2B-GPM-CORRA",
    ]
    if product in gpm_products:
        return True

    else:
        return False
