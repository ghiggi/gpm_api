#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 18:40:03 2022

@author: ghiggi
"""
import numpy as np
import matplotlib.pyplot as plt
from gpm_api.utils.checks import check_is_xarray_dataarray
from gpm_api.patch.generator import get_patch_generator
from gpm_api.visualization.plot import plot_image


def plot_patches(
    data_array,
    min_value_threshold=-np.inf,
    max_value_threshold=np.inf,
    min_area_threshold=1,
    max_area_threshold=np.inf,
    footprint=None,
    sort_by="area",
    sort_decreasing=True,
    n_patches=None,
    patch_margin=None,
    add_colorbar=True, 
    interpolation="nearest",
    fig_kwargs={}, 
    cbar_kwargs={},
    **plot_kwargs,
):
    check_is_xarray_dataarray(data_array)

    # Define generator
    gpm_da_patch_gen = get_patch_generator(
        xr_obj=data_array,
        min_value_threshold=min_value_threshold,
        max_value_threshold=max_value_threshold,
        min_area_threshold=min_area_threshold,
        max_area_threshold=max_area_threshold,
        footprint=footprint,
        sort_by=sort_by,
        sort_decreasing=sort_decreasing,
        n_patches=n_patches,
        patch_margin=patch_margin,
    )

    # Plot patches
    for da in gpm_da_patch_gen:
        # TODO: ENSURE PATCH SIZE IS SUFFICIENT (NOT DIM 1)(AND CONSTANT)
        try:
            plot_image(da, interpolation=interpolation, add_colorbar=add_colorbar, 
                       fig_kwargs=fig_kwargs, 
                       cbar_kwargs=cbar_kwargs,
                       **plot_kwargs)
            plt.show()
        except: 
            pass

    return None
