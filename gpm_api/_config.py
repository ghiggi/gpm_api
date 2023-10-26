#!/usr/bin/env python3
"""
Created on Fri Oct 20 18:11:18 2023

@author: ghiggi
"""

from donfig import Config

# GPM-API main configuration object
# See https://donfig.readthedocs.io/en/latest/configuration.html for more info.


_CONFIG_DEFAULTS = {
    "warn_non_contiguous_scans": True,
    "warn_non_regular_timesteps": True,
    "warn_invalid_spatial_coordinates": True,
    "warn_multiple_product_versions": True,
}

_CONFIG_PATHS = []

config = Config("gpm_api", defaults=[_CONFIG_DEFAULTS], paths=_CONFIG_PATHS)

# gpm_api.config.pprint()
