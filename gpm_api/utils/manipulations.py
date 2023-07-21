#!/usr/bin/env python3
"""
Created on Thu Jul 20 17:23:16 2023

@author: ghiggi
"""
import numpy as np


def _integrate_concentration(data, heights):
    # Compute the thickness of each level (difference between adjacent heights)
    thickness = np.diff(heights)
    thickness = np.append(thickness, thickness[-1])
    thickness = np.broadcast_to(thickness, data.shape)
    # Identify where along the profile is always nan
    mask_nan = np.all(np.isnan(data), axis=-1)  # all profile is nan
    # Compute the path by summing up the product of concentration and thickness
    path = np.nansum(thickness * data, axis=-1)
    path[mask_nan] = np.nan
    return path


def integrate_profile_concentration(dataarray, name, scale_factor=None, units=None):
    """Utility to convert LWC or IWC to LWP or IWP.

    Input data have unit g/m³.
    Output data will have unit kg/m²

    height a list or array of corresponding heights for each level.
    """
    if scale_factor is not None:
        if units is None:
            raise ValueError("Specify output 'units' when the scale_factor is applied.")
    # Compute integrated value
    data = dataarray.data.copy()
    heights = np.asanyarray(dataarray["range"].data)
    output = _integrate_concentration(data, heights)
    # Scale output value
    if scale_factor is not None:
        output = output / scale_factor
    # Create DataArray
    da_path = dataarray.isel({"range": 0}).copy()
    da_path.name = name
    da_path.data = output
    if scale_factor:
        da_path.attrs["units"] = units
    return da_path
