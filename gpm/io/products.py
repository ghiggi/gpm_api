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
"""This module contains functions to obtain GPM products information."""
import datetime
import functools
import os

import numpy as np

from gpm.io.checks import (
    check_full_product_levels,
    check_product_categories,
    check_product_levels,
    check_product_types,
    check_product_validity,
    check_product_version,
    check_satellites,
    check_sensors,
    check_time,
    check_versions,
)
from gpm.utils.list import flatten_list
from gpm.utils.yaml import read_yaml

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


@functools.cache
def get_info_dict():
    """Get product info dictionary."""
    from gpm import _root_path

    filepath = os.path.join(_root_path, "gpm", "etc", "products.yaml")
    return read_yaml(filepath)


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
            key="satellite",
            values=satellites,
            info_dict=info_dict,
        )
    if product_categories is not None:
        product_categories = check_product_categories(product_categories)
        info_dict = _subset_info_dict_by_key(
            key="product_category",
            values=product_categories,
            info_dict=info_dict,
        )
    if sensors is not None:
        sensors = check_sensors(sensors)
        info_dict = _subset_info_dict_by_key(key="sensor", values=sensors, info_dict=info_dict)
    if product_types is not None:
        product_types = check_product_types(product_types)
        info_dict = _subset_info_dict_by_key(
            key="product_types",
            values=product_types,
            info_dict=info_dict,
        )
    if full_product_levels is not None:
        full_product_levels = check_full_product_levels(full_product_levels)
        info_dict = _subset_info_dict_by_key(
            key="full_product_level",
            values=full_product_levels,
            info_dict=info_dict,
        )
    if product_levels is not None:
        product_levels = check_product_levels(product_levels)
        info_dict = _subset_info_dict_by_key(
            key="product_level",
            values=product_levels,
            info_dict=info_dict,
        )
    if versions is not None:
        versions = check_versions(versions)
        info_dict = _subset_info_dict_by_key(
            key="available_versions",
            values=versions,
            info_dict=info_dict,
        )
    return info_dict


def filter_info_dict_by_time(info_dict, start_time, end_time):
    """Filter ``info_dict`` by ``start_time`` and ``end_time``.

    Assume that either ``start_time`` or ``end_time`` are not ``None``.

    Parameters
    ----------
    start_time : (datetime.datetime, datetime.date, np.datetime64, str)
        Start time.
        Accepted types: ``datetime.datetime``, ``datetime.date``, ``np.datetime64`` or ``str``
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.
    end_time : (datetime.datetime, datetime.date, np.datetime64, str)
        Start time.
        Accepted types:  ``datetime.datetime``, ``datetime.date``, ``np.datetime64`` or ``str``
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.

    """
    from gpm.io.filter import is_granule_within_time

    if start_time is None:
        start_time = datetime.datetime(1987, 7, 9, 0, 0, 0)
    if end_time is None:
        end_time = datetime.datetime.utcnow()

    start_time = check_time(start_time)
    end_time = check_time(end_time)

    new_info_dict = {}
    for product, product_info in info_dict.items():
        sensor_start_time = product_info["start_time"]
        sensor_end_time = product_info["end_time"]
        if sensor_end_time is None:
            sensor_end_time = datetime.datetime.utcnow()
        if is_granule_within_time(
            start_time=start_time,
            end_time=end_time,
            file_start_time=sensor_start_time,
            file_end_time=sensor_end_time,
        ):
            new_info_dict[product] = product_info
    return new_info_dict


@functools.cache
def get_products_pattern_dict():
    """Return the filename pattern* associated to all GPM products."""
    info_dict = get_info_dict()
    products = available_products()
    return {product: info_dict[product]["pattern"] for product in products}


####----------------------------------------------------------------------------------
#### Product utilities


@functools.cache
def get_product_info(product):
    """Provide the product info dictionary."""
    if not isinstance(product, str):
        raise TypeError("'product' must be a string.")
    info_dict = get_info_dict()
    valid_products = list(get_info_dict())
    if product not in valid_products:
        raise ValueError("Please provide a valid GPM product --> gpm.available_products().")
    return info_dict[product]


def get_product_start_time(product):
    """Provide the product ``start_time``."""
    return get_product_info(product)["start_time"]


def get_product_end_time(product):
    """Provide the product ``end_time``."""
    end_time = get_info_dict()[product]["end_time"]
    if end_time is None:
        end_time = datetime.datetime.utcnow()
    return end_time


def get_product_pattern(product):
    """Return the filename pattern associated to GPM product."""
    info_dict = get_info_dict()
    return info_dict[product]["pattern"]


def get_product_category(product):
    """Get the ``product_category`` of a GPM product.

    The ``product_category`` is used to organize file on disk.
    """
    product_category = get_product_info(product).get("product_category", None)
    if product_category is None:
        raise ValueError(
            f"The product_category for {product} product is not specified in the config files.",
        )
    return product_category


def get_product_level(product, full=False):
    """Get the ``product_level`` of a GPM product."""
    # TODO: Add L3 products: i.e  3B-HH3, 3B-DAY
    # Notes:
    # - "GPMCOR" --> 1B
    # - full_product_level = info_dict[product]["pattern"].split(".")[0]
    # - product_level --> product_level[0:2]
    if full:
        return get_product_info(product).get("full_product_level", None)
    return get_product_info(product).get("product_level", None)


def available_product_versions(product):
    """Provides a list with the available product versions."""
    return get_product_info(product)["available_versions"]


def available_scan_modes(product, version):
    """Return the available ``scan_modes`` for a given product (and specific version)."""
    product_info = get_product_info(product)
    version = check_product_version(version, product)
    product = check_product_validity(product)
    return product_info["scan_modes"]["V" + str(version)]


def get_last_product_version(product):
    """Provide the most recent product version."""
    return available_product_versions(product)[-1]


def is_trmm_product(product):
    """Check if the product arises from the TRMM satellite."""
    # 2A-ENV-PR" not available on GES DISC
    # 2B-TRMM-CSAT not available on GES DISC

    trmm_products = available_products(satellites="TRMM")
    return product in trmm_products


def is_gpm_api_product(product):
    """Check if the product arises from the GPM satellite."""
    #  '2B-GPM-CSAT', not available on GES DISC
    gpm_api_products = available_products(satellites="GPM")
    return product in gpm_api_products


####----------------------------------------------------------------------------------
#### GPM Tools


def _subset_info_dict_by_key(key, values, info_dict=None):
    """Subset the info dictionary by key and value(s)."""
    if info_dict is None:
        info_dict = get_info_dict()
    if not isinstance(values, list):
        values = [values]
    return {
        product: product_info
        for product, product_info in info_dict.items()
        if np.any(np.isin(values, product_info.get(key, None)))
    }


def _get_unique_key_values(info_dict, key):
    """Get all unique key values."""
    names = []
    for product_info in info_dict.values():
        value = product_info.get(key, None)
        if value is not None:
            names.append(value)
    names = flatten_list(names)
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
                    f"{primary}-{secondary}" if key == "sensor" else f"{secondary}-{primary}",
                )
            else:
                names.append(primary)
    return np.unique(names).tolist()


def get_available_versions():
    """Get the list of available versions."""
    return [4, 5, 6, 7]


@functools.cache
def get_available_products():
    """Get the list of all available products."""
    info_dict = get_info_dict()
    return list(info_dict)


def get_available_product_types():
    """Get the list of available product types."""
    return ["RS", "NRT"]


@functools.cache
def get_available_product_levels(full=False):
    """Get the list of all available product levels."""
    key = "product_level" if not full else "full_product_level"
    info_dict = get_info_dict()
    return _get_unique_key_values(info_dict, key=key)


@functools.cache
def get_available_product_categories():
    """Get the list of all available product categories."""
    info_dict = get_info_dict()
    return _get_unique_key_values(info_dict, key="product_category")


@functools.cache
def get_available_satellites(prefix_with_sensor=False):
    """Get the list of all available satellites."""
    info_dict = get_info_dict()
    return _get_sensor_satellite_names(
        info_dict,
        key="satellite",
        combine_with="sensor" if prefix_with_sensor else None,
    )


@functools.cache
def get_available_sensors(suffix_with_satellite=False):
    """Get the list of all available sensors."""
    info_dict = get_info_dict()
    return _get_sensor_satellite_names(
        info_dict,
        key="sensor",
        combine_with="satellite" if suffix_with_satellite else None,
    )


def available_versions(
    satellites=None,
    sensors=None,
    product_types=None,
    product_categories=None,
    product_levels=None,
    full_product_levels=None,
):
    """Provide a list of available GPM versions.

    Parameters
    ----------
    product_types : (str or list), optional
        If ``None`` (default), provide all products (``RS`` and ``NRT``).
        If ``RS``, provide a list of all GPM RS products available for download.
        If ``NRT``, provide a list of all GPM NRT products available for download.
    product_categories: (str or list), optional
        If ``None`` (default), provide products from all product categories.
        If ``str``, must be a valid product category.
        Valid product categories are: ``PMW``, ``RADAR``, ``IMERG``, ``CMB``.
        The list of available sensors can also be retrieved using ``available_product_categories()``.
    product_levels: (str or list), optional
        If ``None`` (default), provide products from all product levels.
        If ``str``, must be a valid product level.
        Valid product levels are: ``1A``, ``1B``, ``1C``, ``2A``, ``2B``, ``3B``.
        The list of available sensors  also be retrieved using ``available_product_levels()``.
    satellites: (str or list), optional
        If ``None`` (default), provide products from all satellites.
        If ``str``, must be a valid satellites.
        The list of available satellites can be retrieved using ``available_satellites()``.
    sensors: (str or list), optional
        If ``None`` (default), provide products from all sensors.
        If ``str``, must be a valid sensor.
        The list of available sensors can be retrieved using ``available_sensors()``.

    Returns
    -------
    List
        List of available GPM versions.

    """
    info_dict = get_info_dict_subset(
        sensors=sensors,
        satellites=satellites,
        product_categories=product_categories,  # RADAR, PMW, CMB, ...
        product_types=product_types,  # RS, NRT
        versions=None,
        product_levels=product_levels,
        full_product_levels=full_product_levels,
    )
    return _get_unique_key_values(info_dict, key="available_versions")


def available_products(
    satellites=None,
    sensors=None,
    product_categories=None,
    product_types=None,
    versions=None,
    product_levels=None,
    full_product_levels=None,
    start_time=None,
    end_time=None,
):
    """Provide a list of available GPM products for download.

    Parameters
    ----------
    product_types : (str or list), optional
        If ``None`` (default), provide all products (``RS`` and ``NRT``).
        If ``RS``, provide a list of all GPM RS products available for download.
        If ``NRT``, provide a list of all GPM NRT products available for download.
    product_categories: (str or list), optional
        If ``None`` (default), provide products from all product categories.
        If ``str``, must be a valid product category.
        Valid product categories are: ``PMW``, ``RADAR``, ``IMERG``, ``CMB``.
        The list of available sensors can also be retrieved using ``available_product_categories()``.
    product_levels: (str or list), optional
        If ``None`` (default), provide products from all product levels.
        If ``str``, must be a valid product level.
        Valid product levels are: ``1A``, ``1B``, ``1C``, ``2A``, ``2B``, ``3B``.
        The list of available sensors  also be retrieved using ``available_product_levels()``.
    versions: (int or list), optional
        If ``None`` (default), provide products from all versions.
        If ``int``, must be a valid version.
        Valid product levels are: ``4``, ``5``, ``6``, ``7``.
        The list of available sensors can also be retrieved using ``available_versions()``.
    satellites: (str or list), optional
        If ``None`` (default), provide products from all satellites.
        If ``str``, must be a valid satellites.
        The list of available satellites can be retrieved using ``available_satellites()``.
    sensors: (str or list), optional
        If ``None`` (default), provide products from all sensors.
        If ``str``, must be a valid sensor.
        The list of available sensors can be retrieved using ``available_sensors()``.
    start_time : (datetime.datetime, datetime.date, np.datetime64, str)
        Start time period over which to search for available products.
        The default is ``None`` (start of the GPM mission).
        Accepted types: ``datetime.datetime``, ``datetime.date``, ``np.datetime64`` or ``str``.
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.
    end_time : (datetime.datetime, datetime.date, np.datetime64, str)
        End time period over which to search for available products.
        The default is ``None`` (current time).
        Accepted types: ``datetime.datetime``, ``datetime.date``, ``np.datetime64`` or ``str``.
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.

    Returns
    -------
    List
        List of available GPM products.

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

    # Subset by time period if specified
    if start_time is not None or end_time is not None:
        info_dict = filter_info_dict_by_time(info_dict, start_time=start_time, end_time=end_time)

    products = list(info_dict)
    return sorted(products)


def available_product_levels(
    satellites=None,
    sensors=None,
    product_categories=None,
    product_types=None,
    versions=None,
    full=False,
):
    """Provide a list of available GPM product levels.

    Parameters
    ----------
    product_types : (str or list), optional
        If ``None`` (default), provide all products (``RS`` and ``NRT``).
        If ``RS``, provide a list of all GPM RS products available for download.
        If ``NRT``, provide a list of all GPM NRT products available for download.
    product_categories: (str or list), optional
        If ``None`` (default), provide products from all product categories.
        If ``str``, must be a valid product category.
        Valid product categories are: ``PMW``, ``RADAR``, ``IMERG``, ``CMB``.
        The list of available sensors can also be retrieved using ``available_product_categories()``.
    versions: (int or list), optional
        If ``None`` (default), provide products from all versions.
        If ``int``, must be a valid version.
        Valid product levels are: ``4``, ``5``, ``6``, ``7``.
        The list of available sensors can also be retrieved using ``available_versions()``.
    satellites: (str or list), optional
        If ``None`` (default), provide products from all satellites.
        If ``str``, must be a valid satellites.
        The list of available satellites can be retrieved using ``available_satellites()``.
    sensors: (str or list), optional
        If ``None`` (default), provide products from all sensors.
        If ``str``, must be a valid sensor.
        The list of available sensors can be retrieved using ``available_sensors()``.

    Returns
    -------
    List
        List of available GPM product levels.

    """
    # Define product level key
    key = "product_level" if not full else "full_product_level"
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


def available_product_categories(
    satellites=None,
    sensors=None,
    product_types=None,
    versions=None,
    product_levels=None,
    full_product_levels=None,
):
    """Provide a list of available GPM product categories.

    Parameters
    ----------
    product_types : (str or list), optional
        If ``None`` (default), provide all products (``RS`` and ``NRT``).
        If ``RS``, provide a list of all GPM RS products available for download.
        If ``NRT``, provide a list of all GPM NRT products available for download.
    product_levels: (str or list), optional
        If ``None`` (default), provide products from all product levels.
        If ``str``, must be a valid product level.
        Valid product levels are: ``1A``, ``1B``, ``1C``, ``2A``, ``2B``, ``3B``.
        The list of available sensors  also be retrieved using ``available_product_levels()``.
    versions: (int or list), optional
        If ``None`` (default), provide products from all versions.
        If ``int``, must be a valid version.
        Valid product levels are: ``4``, ``5``, ``6``, ``7``.
        The list of available sensors can also be retrieved using ``available_versions()``.
    satellites: (str or list), optional
        If ``None`` (default), provide products from all satellites.
        If ``str``, must be a valid satellites.
        The list of available satellites can be retrieved using ``available_satellites()``.
    sensors: (str or list), optional
        If ``None`` (default), provide products from all sensors.
        If ``str``, must be a valid sensor.
        The list of available sensors can be retrieved using ``available_sensors()``.

    Returns
    -------
    List
        List of available GPM product categories.

    """
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


def available_satellites(
    sensors=None,
    product_categories=None,
    product_types=None,
    versions=None,
    product_levels=None,
    full_product_levels=None,
    start_time=None,
    end_time=None,
    prefix_with_sensor=False,
):
    """Provide a list of available GPM satellites.

    If ``prefix_with_sensor=True``, it prefixes the satellite name with the satellite name: ``{sensor}-{satellite}``.

    Parameters
    ----------
    product_types : (str or list), optional
        If ``None`` (default), provide all products (``RS`` and ``NRT``).
        If ``RS``, provide a list of all GPM RS products available for download.
        If ``NRT``, provide a list of all GPM NRT products available for download.
    product_categories: (str or list), optional
        If ``None`` (default), provide products from all product categories.
        If ``str``, must be a valid product category.
        Valid product categories are: ``PMW``, ``RADAR``, ``IMERG``, ``CMB``.
        The list of available sensors can also be retrieved using ``available_product_categories()``.
    product_levels: (str or list), optional
        If ``None`` (default), provide products from all product levels.
        If ``str``, must be a valid product level.
        Valid product levels are: ``1A``, ``1B``, ``1C``, ``2A``, ``2B``, ``3B``.
        The list of available sensors  also be retrieved using ``available_product_levels()``.
    versions: (int or list), optional
        If ``None`` (default), provide products from all versions.
        If ``int``, must be a valid version.
        Valid product levels are: ``4``, ``5``, ``6``, ``7``.
        The list of available sensors can also be retrieved using ``available_versions()``.
    sensors: (str or list), optional
        If ``None`` (default), provide products from all sensors.
        If ``str``, must be a valid sensor.
        The list of available sensors can be retrieved using ``available_sensors()``.
    start_time : (datetime.datetime, datetime.date, np.datetime64, str)
        Start time period over which to search for available products.
        The default is ``None`` (start of the GPM mission).
        Accepted types: ``datetime.datetime``, ``datetime.date``, ``np.datetime64`` or ``str``.
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.
    end_time : (datetime.datetime, datetime.date, np.datetime64, str)
        End time period over which to search for available products.
        The default is ``None`` (current time).
        Accepted types: ``datetime.datetime``, ``datetime.date``, ``np.datetime64`` or ``str``.
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.
    prefix_with_sensor: bool, optional
        If ``True``, it prefixes the satellite name with the satellite name: ``{sensor}-{satellite}``.
        If ``False`` (the default), it just return the name of the satellite.

    Returns
    -------
    List
        List of available GPM satellites.

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
    # Subset by time period if specified
    if start_time is not None or end_time is not None:
        info_dict = filter_info_dict_by_time(info_dict, start_time=start_time, end_time=end_time)

    combine_with = "sensor" if prefix_with_sensor else None
    return _get_sensor_satellite_names(
        info_dict,
        key="satellite",
        combine_with=combine_with,
    )


def available_sensors(
    satellites=None,
    product_categories=None,
    product_types=None,
    versions=None,
    product_levels=None,
    full_product_levels=None,
    start_time=None,
    end_time=None,
    suffix_with_satellite=False,
):
    """Provide a list of available GPM sensors.

    If ``suffix_with_satellite=True``, it suffixes the sensor name with the satellite name: ``{sensor}-{satellite}``.

    Parameters
    ----------
    product_types : (str or list), optional
        If ``None`` (default), provide all products (``RS`` and ``NRT``).
        If ``RS``, provide a list of all GPM RS products available for download.
        If ``NRT``, provide a list of all GPM NRT products available for download.
    product_categories: (str or list), optional
        If ``None`` (default), provide products from all product categories.
        If ``str``, must be a valid product category.
        Valid product categories are: ``PMW``, ``RADAR``, ``IMERG``, ``CMB``.
        The list of available sensors can also be retrieved using ``available_product_categories()``.
    product_levels: (str or list), optional
        If ``None`` (default), provide products from all product levels.
        If ``str``, must be a valid product level.
        Valid product levels are: ``1A``, ``1B``, ``1C``, ``2A``, ``2B``, ``3B``.
        The list of available sensors  also be retrieved using ``available_product_levels()``.
    versions: (int or list), optional
        If ``None`` (default), provide products from all versions.
        If ``int``, must be a valid version.
        Valid product levels are: ``4``, ``5``, ``6``, ``7``.
        The list of available sensors can also be retrieved using ``available_versions()``.
    satellites: (str or list), optional
        If ``None`` (default), provide products from all satellites.
        If ``str``, must be a valid satellites.
        The list of available satellites can be retrieved using ``available_satellites()``.
    start_time : (datetime.datetime, datetime.date, np.datetime64, str)
        Start time period over which to search for available products.
        The default is ``None`` (start of the GPM mission).
        Accepted types: ``datetime.datetime``, ``datetime.date``, ``np.datetime64`` or ``str``.
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.
    end_time : (datetime.datetime, datetime.date, np.datetime64, str)
        End time period over which to search for available products.
        The default is ``None`` (current time).
        Accepted types: ``datetime.datetime``, ``datetime.date``, ``np.datetime64`` or ``str``.
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.
    suffix_with_satellite: bool, optional
        If ``True``, it suffixes the sensor name with the satellite name: ``{sensor}-{satellite}``.
        If ``False`` (the default), it just return the name of the sensor.

    Returns
    -------
    List
        List of available GPM sensors.

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
    # Subset by time period if specified
    if start_time is not None or end_time is not None:
        info_dict = filter_info_dict_by_time(info_dict, start_time=start_time, end_time=end_time)

    combine_with = "satellite" if suffix_with_satellite else None
    return _get_sensor_satellite_names(
        info_dict,
        key="sensor",
        combine_with=combine_with,
    )
