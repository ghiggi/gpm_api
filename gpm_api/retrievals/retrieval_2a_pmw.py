#!/usr/bin/env python3
"""
Created on Mon Jul 31 17:53:42 2023

@author: ghiggi
"""


def retrieve_totalWaterPath(ds):
    """Retrieve clutter height."""
    da = ds["rainWaterPath"] + ds["cloudWaterPath"] + ds["iceWaterPath"]
    da.name = "totalWaterPath"
    return da
