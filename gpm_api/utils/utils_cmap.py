#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:33:35 2021

@author: ghiggi
"""

## Colormaps 
# - Drpy: https://github.com/dopplerchase/DRpy/blob/master/drpy/graph/colormaps_drpy.py 


import copy
import warnings
from matplotlib import cm, colors
import matplotlib.pylab as plt
import numpy as np

PRECIP_VALID_TYPES = ("intensity", "depth", "prob")
PRECIP_VALID_UNITS = ("mm/h", "mm", "dBZ")

def get_colormap(ptype, units="mm/h", colorscale="pysteps", bad_color="gray"):
    """Function to generate a colormap (cmap) and norm.
    Parameters
    ----------
    ptype : {'intensity', 'depth', 'prob'}, optional
        Type of the map to plot: 'intensity' = precipitation intensity field,
        'depth' = precipitation depth (accumulation) field,
        'prob' = exceedance probability field.
    units : {'mm/h', 'mm', 'dBZ'}, optional
        Units of the input array. If ptype is 'prob', this specifies the unit of
        the intensity threshold.
    colorscale : {'pysteps', 'STEPS-BE', 'BOM-RF3'}, optional
        Which colorscale to use. Applicable if units is 'mm/h', 'mm' or 'dBZ'.
    Returns
    -------
    cmap : Colormap instance
        colormap
    norm : colors.Normalize object
        Colors norm
    clevs: list(float)
        List of precipitation values defining the color limits.
    clevs_str: list(str)
        List of precipitation values defining the color limits (with correct
        number of decimals).
    """
    if ptype in ["intensity", "depth"]:
        # Get list of colors
        color_list, clevs, clevs_str = _get_colorlist(units, colorscale)

        cmap = colors.LinearSegmentedColormap.from_list(
            "cmap", color_list, len(clevs) - 1
        )

        if colorscale == "BOM-RF3":
            cmap.set_over("black", 1)
        if colorscale == "pysteps":
            cmap.set_over("darkred", 1)
        if colorscale == "STEPS-BE":
            cmap.set_over("black", 1)
        if colorscale == "GPM_solid":
            cmap.set_over("#8c149c", 1)
        if colorscale == "GPM_liquid":
            cmap.set_over("#910000", 1)
            
        norm = colors.BoundaryNorm(clevs, cmap.N)

        cmap.set_bad(bad_color, alpha=0.5)
        cmap.set_under("none")

        return cmap, norm, clevs, clevs_str

    if ptype == "prob":
        cmap = copy.copy(plt.get_cmap("OrRd", 10))
        cmap.set_bad(bad_color, alpha=0.5)
        cmap.set_under("none")
        
        clevs = np.linspace(0, 1, 11)
        clevs[0] = 1e-3  # to set zeros to transparent
        norm = colors.BoundaryNorm(clevs, cmap.N)
        clevs_str = [f"{clev:.1f}" for clev in clevs]
        return cmap, norm, clevs, clevs_str

    return cm.get_cmap("jet"), colors.Normalize(), None, None


def _get_colorlist(units="mm/h", colorscale="pysteps"):
    """
    Function to get a list of colors to generate the colormap.
    Parameters
    ----------
    units : str
        Units of the input array (mm/h, mm or dBZ)
    colorscale : str
        Which colorscale to use (BOM-RF3, pysteps, STEPS-BE)
    Returns
    -------
    color_list : list(str)
        List of color strings.
    clevs : list(float)
        List of precipitation values defining the color limits.
    clevs_str : list(str)
        List of precipitation values defining the color limits
        (with correct number of decimals).
    """

    if colorscale == "BOM-RF3":
        color_list = np.array(
            [
                (255, 255, 255),  # 0.0
                (245, 245, 255),  # 0.2
                (180, 180, 255),  # 0.5
                (120, 120, 255),  # 1.5
                (20, 20, 255),  # 2.5
                (0, 216, 195),  # 4.0
                (0, 150, 144),  # 6.0
                (0, 102, 102),  # 10
                (255, 255, 0),  # 15
                (255, 200, 0),  # 20
                (255, 150, 0),  # 30
                (255, 100, 0),  # 40
                (255, 0, 0),  # 50
                (200, 0, 0),  # 60
                (120, 0, 0),  # 75
                (40, 0, 0),   # 100 
            ]
        )   
        color_list = color_list / 255.0
        if units == "mm/h":
            clevs = [
                0.0,
                0.2,
                0.5,
                1.5,
                2.5,
                4,
                6,
                10,
                15,
                20,
                30,
                40,
                50,
                60,
                75,
                100,
                150,
            ]
        elif units == "mm":
            clevs = [
                0.0,
                0.2,
                0.5,
                1.5,
                2.5,
                4,
                5,
                7,
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
            ]
        else:
            raise ValueError("Wrong units in get_colorlist: %s" % units)
    elif colorscale == "pysteps":
        # pinkHex = '#%02x%02x%02x' % (232, 215, 242)
        redgrey_hex = "#%02x%02x%02x" % (156, 126, 148)
        color_list = [
            redgrey_hex,
            "#640064",
            "#AF00AF",
            "#DC00DC",
            "#3232C8",
            "#0064FF",
            "#009696",
            "#00C832",
            "#64FF00",
            "#96FF00",
            "#C8FF00",
            "#FFFF00",
            "#FFC800",
            "#FFA000",
            "#FF7D00",
            "#E11900",
        ]
        if units in ["mm/h", "mm"]:
            clevs = [
                0.08,
                0.16,
                0.25,
                0.40,
                0.63,
                1,
                1.6,
                2.5,
                4,
                6.3,
                10,
                16,
                25,
                40,
                63,
                100,
                160,
            ]
        elif units == "dBZ":
            clevs = np.arange(10, 65, 5)
        else:
            raise ValueError("Wrong units in get_colorlist: %s" % units)
    elif colorscale == "STEPS-BE":
        color_list = [
            "cyan",
            "deepskyblue",
            "dodgerblue",
            "blue",
            "chartreuse",
            "limegreen",
            "green",
            "darkgreen",
            "yellow",
            "gold",
            "orange",
            "red",
            "magenta",
            "darkmagenta",
        ]
        if units in ["mm/h", "mm"]:
            clevs = [0.1, 0.25, 0.4, 0.63, 1, 1.6, 2.5, 4, 6.3, 10, 16, 25, 40, 63, 100]
        elif units == "dBZ":
            clevs = np.arange(10, 65, 5)
        else:
            raise ValueError("Wrong units in get_colorlist: %s" % units)
    elif colorscale == "GPM_liquid":
        color_list = [
            "#008114",  # [0.1 - 0.2]
            "#008b52",  # [0.2 - 0.3]
            "#00b330",  # [0.3 - 0.5]
            "#60d300",  # [0.5 - 1.0]
            "#b4e700",  # [1.0 - 2.0]
            "#fffb00",  # [2.0 - 3.0]
            "#ffc400",  # [3.0 - 5.0]
            "#ff9300",  # [5.0 - 10.0]
            "#ff0000",  # [10.0 - 20.0]
            "#c80000",  # [20.0 - 50.0]
            ]
            # Over  (> 50)
            #910000
            # Under (< 0.1)
            #3a3d48
        if units in ["mm/h","mm"]:
            clevs = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50]
        elif units == "dBZ":
            clevs = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50]
    elif colorscale == "GPM_solid":
        color_list = [
            "#45ffff", # [0.1 - 0.2]
            "#49ddff", # [0.2 - 0.3]
            "#4dbbff", # [0.3 - 0.5]
            "#4f9aff", # [0.5 - 1.0]
            "#517aff", # [1.0 - 2.0]
            "#525dff", # [2.0 - 3.0]
            "#5346ff", # [3.0 - 5.0]
            "#5a2fd9", # [5.0 - 10.0]
            "#7321bb", # [10.0 - 20.0]
            "#8c149c", # [20.0 - 50.0]
         ]
        # Under (> 50)
        #3a3d48
        # Over (< 0.1)
        #8c149c
        if units in ["mm/h","mm"]:
            clevs = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50]
        elif units == "dBZ":
            clevs = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50]
        
    else:
        print("Invalid colorscale", colorscale)
        raise ValueError("Invalid colorscale " + colorscale)

    # Generate color level strings with correct amount of decimal places
    clevs_str = _dynamic_formatting_floats(clevs)

    return color_list, clevs, clevs_str

def _dynamic_formatting_floats(float_array, colorscale="pysteps"):
    """Function to format the floats defining the class limits of the colorbar."""
    float_array = np.array(float_array, dtype=float)

    labels = []
    for label in float_array:
        if 0.1 <= label < 1:
            if colorscale == "pysteps":
                formatting = ",.2f"
            else:
                formatting = ",.1f"
        elif 0.01 <= label < 0.1:
            formatting = ",.2f"
        elif 0.001 <= label < 0.01:
            formatting = ",.3f"
        elif 0.0001 <= label < 0.001:
            formatting = ",.4f"
        elif label >= 1 and label.is_integer():
            formatting = "i"
        else:
            formatting = ",.1f"

        if formatting != "i":
            labels.append(format(label, formatting))
        else:
            labels.append(str(int(label)))

    return labels



# get colormap and color levels
colorscale = "STEPS-BE"
colorscale = "BOM-RF3"
colorscale = "pysteps"
colorscale= "GPM_liquid"
colorscale= "GPM_solid"

units="mm/h"
ptype="intensity"
cmap, norm, clevs, clevs_str = get_colormap(ptype=ptype,
                                            units=units,
                                            colorscale=colorscale)
if ptype in ["intensity", "depth"]:
    extend = "max"
else:
    extend = "neither"
      
cbar_kwargs = {'ticks': clevs,
               'spacing': 'uniform', 
               'extend': extend,
               'shrink': 0.8
              }       
# p = da_precip_subset.plot.imshow(x="along_track", y="cross_track",
#                                  interpolation="bilinear", # "nearest", "bicubic"
#                                  cmap=cmap, norm=norm, cbar_kwargs=cbar_kwargs)
# cbar = p.colorbar
# cbar.ax.set_yticklabels(clevs_str)
 
### TODO 
# - Infer IMERG_Clim cmap at https://svs.gsfc.nasa.gov/4837 
# - Refactor cmaps and read from disk ... 
