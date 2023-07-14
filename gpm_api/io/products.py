#!/usr/bin/env python3
"""
Created on Thu Oct 13 11:13:15 2022

@author: ghiggi
"""
import functools
import os

from gpm_api.io.checks import (
    check_product_category,
    check_product_level,
    check_product_type,
    check_product_validity,
    check_version,
)
from gpm_api.utils.yaml import read_yaml_file

### Notes and TODO list

# product
# - PMW: <product_level>-<sensor>-<satellite> # 2A-AMSUB-NOAA15, # 2A-AMSUB-NOAA15_CLIM
# - RADAR: <product_level>-<....>
# - IMERG: IMERG-run_type>  # IMERG-LR
# product_level:
# - IMERG has no product level

# - Download and reader for L3 product not yet implemented
# --> https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/

# info_dict --> additional future keys
# - ges_disc_dir
# - product_level ?  --> can be inferred by gpm-api product name also ...
# - satellite   ?   --> not for IMERG
# - sensor      ?   --> not for IMERG
# - start_time  (for time period of product)
# - end_time    (for time period of product)

# scan_modes
# "1A-GMI": has also S3 and gmi1aHeader but can't be processed
# "1A-TMI": has also tmi1aHeader but can't be processed
# "1C-AMSRE": scan_modes need to be specified

### Product changes:
# In V05, GPROF products are available as 2A-CLIM and 2A
# In V06, GPROF is not available !
# In V07 RS, GPROF is available only as 2A-CLIM (except for 2A.GPM.GMI) on PPS
# In V07 RS, GPROF is available (mostly) as 2A-CLIM and 2A on GES DISC
# --> AMSUB and SSMI (F08-15) only with CLIM product on GES DISC
# In V07 NRT, GPROF is available only as 2A

# PRPS is available only in V06 (CLIM and no CLIM) (PMW Saphir)

# --> Maybe some check on input should be done to facilitate users ...
# --> Exploit available_products (with product_category and level asked to suggest valid alternatives?)

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


def available_products(product_type=None, product_category=None, product_level=None):
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

    return sorted(products)


def available_scan_modes(product, version):
    """Return the available scan_modes for a given product (and specific version)."""
    check_version(version)
    check_product_validity(product)
    info_dict = get_info_dict()
    if version == 7:
        key = "scan_modes_v7"
    else:  # <= 6
        key = "scan_modes_v6"
    scan_modes = info_dict[product].get(key)
    return scan_modes


def get_products_pattern_dict():
    """Return the filename pattern* associated to all GPM products."""
    info_dict = get_info_dict()
    products = available_products()
    pattern_dict = {product: info_dict[product]["pattern"] for product in products}
    return pattern_dict
