#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:56:39 2022

@author: ghiggi
"""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from gpm_api.visualization.plot import (
    _plot_cartopy_background,
    _plot_cartopy_imshow,
    _plot_mpl_imshow,
    _plot_xr_imshow,
    get_colorbar_settings,
)

figsize = (12, 10)
dpi = 100
interpolation = "nearest"
subplot_kw = {"projection": ccrs.PlateCarree()}
