#!/usr/bin/env python3
"""
Created on Fri Jul 28 16:07:37 2023

@author: ghiggi
"""


def retrieve_binClutterFreeBottomHeight(ds):
    """Retrieve clutter height."""
    da = ds.gpm_api.get_height_at_bin(bin="binClutterFreeBottom")
    da.name = "binClutterFreeBottomHeight"
    return da


def retrieve_binRealSurfaceHeightKa(ds):
    """Retrieve bin height of real surface at Ka band."""
    da = ds.gpm_api.get_height_at_bin(bin=ds["binRealSurface"].sel({"radar_frequency": "Ka"}))
    da.name = "binRealSurfaceHeightKa"
    return da


def retrieve_binRealSurfaceHeightKu(ds):
    """Retrieve bin height of real surface at Ku band."""
    da = ds.gpm_api.get_height_at_bin(bin=ds["binRealSurface"].sel({"radar_frequency": "Ku"}))
    da.name = "binRealSurfaceHeightKu"
    return da
