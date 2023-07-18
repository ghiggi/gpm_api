#!/usr/bin/env python3
"""
Created on Thu Jun 22 14:52:38 2023

@author: ghiggi
"""
from gpm_api.dataset.attrs import (
    DYNAMIC_GLOBAL_ATTRS,
    GRANULE_ONLY_GLOBAL_ATTRS,
    STATIC_GLOBAL_ATTRS,
)
from gpm_api.utils.utils_HDF5 import hdf5_file_attrs


def get_granule_attrs(hdf):
    """Get granule global attributes."""
    # Retrieve attributes dictionary (per group)
    hdf_attr = hdf5_file_attrs(hdf)
    # Flatten attributes (without group)
    attrs = {}
    _ = [attrs.update(group_attrs) for group, group_attrs in hdf_attr.items()]
    # Subset only required attributes
    valid_keys = GRANULE_ONLY_GLOBAL_ATTRS + DYNAMIC_GLOBAL_ATTRS + STATIC_GLOBAL_ATTRS
    attrs = {key: attrs[key] for key in valid_keys if key in attrs}
    return attrs
