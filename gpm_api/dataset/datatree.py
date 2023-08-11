#!/usr/bin/env python3
"""
Created on Tue Jul 18 17:06:05 2023

@author: ghiggi
"""
import datatree

from gpm_api.dataset.dimensions import _rename_datatree_dimensions

# TODO:
# --> open datatrees and concat datatrees
# --> create datatree with option "flattened_scan_modes"
# --> gpm_api.open_granule(datatree=False)  # or if multiple scan_modes provided
# --> gpm_api.open_dataset(datatree=False)  # or if multiple scan_modes provided


def open_datatree(filepath, chunks={}, decode_cf=False, use_api_defaults=True):
    """Open HDF5 in datatree object.

    - chunks={} --> Lazy map to dask.array
      --> Wait for https://github.com/pydata/xarray/pull/7948
      --> Maybe need to implement "auto" option manually that defaults to full shape"
    - chunks="auto" --> datatree fails !
    - chunks=None --> lazy map to numpy.array
    """
    dt = datatree.open_datatree(filepath, engine="netcdf4", chunks=chunks, decode_cf=decode_cf)
    # Assign dimension names
    dt = _rename_datatree_dimensions(dt, use_api_defaults=use_api_defaults)
    return dt
