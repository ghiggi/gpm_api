#!/usr/bin/env python3
"""
Created on Thu Oct 13 11:13:15 2022

@author: ghiggi
"""
import datetime
import functools
import os

import numpy as np

from gpm_api.io.checks import (
    check_full_product_levels,
    check_product_categories,
    check_product_levels,
    check_product_types,
    check_product_validity,
    check_product_version,
    check_satellites,
    check_sensors,
    check_versions,
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


def get_info_dict_subset(
    sensors=None,
    satellites=None,
    product_categories=None,  # RADAR, PMW, CMB, ...
    product_types=None,  # RS, NRT
    versions=None,
    full_product_levels=None,
    product_levels=None,
):
    """Retrieve info dictionary filtered by keys."""
    info_dict = get_info_dict()
    if satellites is not None:
        satellites = check_satellites(satellites)
        info_dict = _subset_info_dict_by_key(
            key="satellite", values=satellites, info_dict=info_dict
        )
    if product_categories is not None:
        product_categories = check_product_categories(product_categories)
        info_dict = _subset_info_dict_by_key(
            key="product_category", values=product_categories, info_dict=info_dict
        )
    if sensors is not None:
        sensors = check_sensors(sensors)
        info_dict = _subset_info_dict_by_key(key="sensor", values=sensors, info_dict=info_dict)
    if product_types is not None:
        product_types = check_product_types(product_types)
        info_dict = _subset_info_dict_by_key(
            key="product_types", values=product_types, info_dict=info_dict
        )
    if full_product_levels is not None:
        full_product_levels = check_full_product_levels(full_product_levels)
        info_dict = _subset_info_dict_by_key(
            key="full_product_level", values=full_product_levels, info_dict=info_dict
        )
    if product_levels is not None:
        product_levels = check_product_levels(product_levels)
        info_dict = _subset_info_dict_by_key(
            key="product_level", values=product_levels, info_dict=info_dict
        )
    if versions is not None:
        versions = check_versions(versions)
        info_dict = _subset_info_dict_by_key(
            key="available_versions", values=versions, info_dict=info_dict
        )
    return info_dict


def _subset_info_dict_by_key(key, values, info_dict=None):
    """Subset the info dictionary by key and value(s)."""
    if info_dict is None:
        info_dict = get_info_dict()
    if not isinstance(values, list):
        values = [values]
    subset_dict = {
        product: product_info
        for product, product_info in info_dict.items()
        if np.any(np.isin(values, product_info.get(key, None)))
    }
    return subset_dict


def _get_unique_key_values(info_dict, key):
    """Get all unique key values."""
    names = []
    for product_info in info_dict.values():
        value = product_info.get(key, None)
        if value is not None and isinstance(value, str):
            names.append(value)
    return np.unique(names).tolist()


def _get_sensor_satellite_names(info_dict, key="sensor", combine_with=None):
    """Helper function to extract and optionally combine sensor or satellite names."""
    names = []
    for product_info in info_dict.values():
        primary = product_info.get(key, None)
        secondary = product_info.get(combine_with, None) if combine_with else None
        if primary is not None and isinstance(primary, str):
            if secondary is not None and isinstance(secondary, str):
                names.append(
                    f"{primary}-{secondary}" if key == "sensor" else f"{secondary}-{primary}"
                )
            else:
                names.append(primary)
    return np.unique(names).tolist()


def get_available_product_types():
    """Get the list of available product types."""
    return ["RS", "NRT"]


def get_available_versions():
    """Get the list of available versions."""
    return [4, 5, 6, 7]


@functools.lru_cache(maxsize=None)
def get_available_products():
    """Get the list of all available products."""
    info_dict = get_info_dict()
    return list(info_dict)


@functools.lru_cache(maxsize=None)
def get_available_product_levels(full=False):
    """Get the list of all available product levels."""
    if not full:
        key = "product_level"
    else:
        key = "full_product_level"
    info_dict = get_info_dict()
    return _get_unique_key_values(info_dict, key=key)


@functools.lru_cache(maxsize=None)
def get_available_product_categories():
    """Get the list of all available product categories."""
    info_dict = get_info_dict()
    return _get_unique_key_values(info_dict, key="product_category")


@functools.lru_cache(maxsize=None)
def get_available_satellites(prefix_with_sensor=False):
    """Get the list of all available satellites."""
    info_dict = get_info_dict()
    return _get_sensor_satellite_names(
        info_dict, key="satellite", combine_with="sensor" if prefix_with_sensor else None
    )


@functools.lru_cache(maxsize=None)
def get_available_sensors(suffix_with_satellite=False):
    """Get the list of all available sensors."""
    info_dict = get_info_dict()
    return _get_sensor_satellite_names(
        info_dict, key="sensor", combine_with="satellite" if suffix_with_satellite else None
    )


def available_product_levels(
    satellites=None,
    sensors=None,
    product_categories=None,
    product_types=None,
    versions=None,
    full=False,
):
    """Return the available product levels."""
    # Define product level key
    if not full:
        key = "product_level"
    else:
        key = "full_product_level"
    # Retrieve info dictionary
    info_dict = get_info_dict_subset(
        sensors=sensors,
        satellites=satellites,
        product_categories=product_categories,  # RADAR, PMW, CMB, ...
        product_types=product_types,  # RS, NRT
        versions=versions,
        product_levels=None,
        full_product_levels=None,
    )
    # Retrieve key values
    return _get_unique_key_values(info_dict, key=key)


def available_satellites(
    sensors=None,
    product_categories=None,
    product_types=None,
    versions=None,
    product_levels=None,
    full_product_levels=None,
    prefix_with_sensor=False,
):
    """Return the available satellites.

    If prefix_with_sensor=True, it prefixes the satellite name with the satellite name: {sensor}-{satellite}.
    """
    info_dict = get_info_dict_subset(
        sensors=sensors,
        satellites=None,
        product_categories=product_categories,  # RADAR, PMW, CMB, ...
        product_types=product_types,  # RS, NRT
        versions=versions,
        product_levels=product_levels,
        full_product_levels=full_product_levels,
    )

    return _get_sensor_satellite_names(
        info_dict, key="satellite", combine_with="sensor" if prefix_with_sensor else None
    )


def available_sensors(
    satellites=None,
    product_categories=None,
    product_types=None,
    versions=None,
    product_levels=None,
    full_product_levels=None,
    suffix_with_satellite=False,
):
    """Return the available sensors.

    If suffix_with_satellite=True, it suffixes the sensor name with the satellite name: {sensor}-{satellite}.
    """
    info_dict = get_info_dict_subset(
        sensors=None,
        satellites=satellites,
        product_categories=product_categories,  # RADAR, PMW, CMB, ...
        product_types=product_types,  # RS, NRT
        versions=versions,
        product_levels=product_levels,
        full_product_levels=full_product_levels,
    )
    return _get_sensor_satellite_names(
        info_dict, key="sensor", combine_with="satellite" if suffix_with_satellite else None
    )


def available_product_categories(
    satellites=None,
    sensors=None,
    product_types=None,
    versions=None,
    product_levels=None,
    full_product_levels=None,
    suffix_with_satellite=False,
):
    """Return the available product categories."""
    info_dict = get_info_dict_subset(
        sensors=sensors,
        satellites=satellites,
        product_categories=None,  # RADAR, PMW, CMB, ...
        product_types=product_types,  # RS, NRT
        versions=versions,
        product_levels=product_levels,
        full_product_levels=full_product_levels,
    )
    return _get_unique_key_values(info_dict, key="product_category")


def available_products(
    satellites=None,
    sensors=None,
    product_categories=None,
    product_types=None,
    versions=None,
    product_levels=None,
    full_product_levels=None,
):
    """
    Provide a list of all/NRT/RS GPM data for download.

    Parameters
    ----------
    product_types : (str or list), optional
        If None (default), provide all products (RS and NRT).
        If 'RS', provide a list of all GPM RS products available for download.
        If 'NRT', provide a list of all GPM NRT products available for download.
    product_categories: (str or list), optional
        If None (default), provide products from all product categories.
        If string, must be a valid product category.
        Valid product categories are: 'PMW', 'RADAR', 'IMERG', 'CMB'.
    product_levels: (str or list), optional
        If None (default), provide products from all product levels.
        If string, must be a valid product level.
        Valid product levels are: '1A','1B','1C','2A','2B','3B'.
        For IMERG products, no product level applies.

    Returns
    -------
    List
        List of GPM available products.

    """
    info_dict = get_info_dict_subset(
        sensors=sensors,
        satellites=satellites,
        product_categories=product_categories,  # RADAR, PMW, CMB, ...
        product_types=product_types,  # RS, NRT
        versions=versions,
        product_levels=product_levels,
        full_product_levels=full_product_levels,
    )
    products = list(info_dict)
    return sorted(products)


#### Single Product Info
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


def available_scan_modes(product, version):
    """Return the available scan_modes for a given product (and specific version)."""
    product_info = get_product_info(product)
    version = check_product_version(version, product)
    product = check_product_validity(product)
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
            f"The product_category for {product} product is not specified in the config files."
        )
    return product_category


def get_product_level(product, full=False):
    """Get the product_level of a GPM product."""
    # TODO: Add L3 products: i.e  3B-HH3, 3B-DAY
    # Notes:
    # - "GPMCOR" --> 1B
    # - full_product_level = info_dict[product]["pattern"].split(".")[0]
    # - product_level --> product_level[0:2]
    if full:
        return get_product_info(product).get("full_product_level", None)
    else:
        return get_product_info(product).get("product_level", None)


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
    """Check if the product arises from the GPM satellite."""
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
