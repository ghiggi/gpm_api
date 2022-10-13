#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:33:35 2021

@author: ghiggi
"""
####--------------------------------------------------------------------------.
### TODO 
# - Infer IMERG_Clim cmap at https://svs.gsfc.nasa.gov/4837 
# - Refactor cmaps and read from disk ... 

## Colormaps 
# - Drpy: https://github.com/dopplerchase/DRpy/blob/master/drpy/graph/colormaps_drpy.py 

# https://matplotlib.org/stable/gallery/color/colormap_reference.html 
# https://matplotlib.org/stable/tutorials/colors/colormaps.html 

# https://stackoverflow.com/questions/69303578/how-to-make-discrete-colormap-contiguous-in-matplotlib 
# https://matplotlib.org/stable/api/colors_api.html

# bad --> masked values 
# np.nan ? --> 

#-----------------------------------------------------------------------------.
# cmap = plt.cm.get_cmap("magma")
# cmap = plt.cm.get_cmap("inferno")
# cmap = plt.cm.get_cmap("plasma")
# cmap = plt.cm.get_cmap("viridis")
# cmap = plt.cm.get_cmap("turbo")
# dir(cmap)
# # Get colors 
# cmap.colors
 
# cmap = plt.cm.get_cmap("parula")
# homeyer rainbow 
# --> https://github.com/dopplerchase/DRpy/blob/master/drpy/graph/colormaps_drpy.py#L1113 
# parula_data 
# --> https://github.com/dopplerchase/DRpy/blob/master/drpy/graph/colormaps_drpy.py#L1048
# cividis 
# ---> https://github.com/dopplerchase/DRpy/blob/master/drpy/graph/colormaps_drpy.py#L1369 

# NOAA radar reflectivity: https://en.wikipedia.org/wiki/DBZ_(meteorology)#/media/File:NOAA_Doppler_DBZ_scale.jpg

# pyart.
# pysteps.

# bivariate.
# trivariate.

#-----------------------------------------------------------------------------.
# cmaps[name] = mpl.colorsListedColormap(data, name=name)
  
####--------------------------------------------------------------------------.
# ListedColormap, which produces a discrete colormap, 
# dir(matplotlib.colors)

# # Continous colormap 
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", color_list)
# norm = mpl.colors.Normalize(vmin, vmax) 

# Discrete colormap 
# cmap = mpl.colors.LinearSegmentedColormap.from_list("cmap", color_list, len(clevs) - 1)
# norm = mpl.colors.BoundaryNorm(clevs, cmap.N)

# https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale 
# Look at pycolorbar existing code 

####--------------------------------------------------------------------------.
import copy
import warnings
import matplotlib 
import matplotlib.colors
import numpy as np
import matplotlib as mpl 
import matplotlib.pylab as plt


PRECIP_VALID_TYPES = ("intensity", "depth", "prob")
PRECIP_VALID_UNITS = ("mm/h", "mm", "dBZ")

CMAP_DICT = { 
   "IMERG_SolidPrecip": {
       "type": "hex", 
       "color_list" : [
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
          ],
     },
     "IMERG_liquid": {
         "type": "hex", 
         "color_list" : [
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
          ],
       },
       "pysteps": {
           "type": "hex", 
           "color_list" : [
             "#9c7e94", # redgray or '#e8d7f2' pink
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
         ],
      }, 
       "STEPS-BE": {
           "type": "name", 
           "color_list" : [
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
           ],
     },
    "BOM-RF3": {
        "type": "rgb255", 
        "color_list" : np.array(
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
            ])/255,  # convert to 0-1 
     },        
}

COLOR_DICT = {    
    'IMERG_Solid': 
        {'over_color': '#8c149c', 
         'under_color': '#3a3d48',
         'bad_color': 'gray', 
         'bad_alpha': 0.5, 
         'cmap': 'IMERG_solid', 
         'cmap_type': "LinearSegmented",
         'levels': [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50],
         "extend": "max",
         # TODO: Add label 
         },
    'IMERG_Liquid': 
        {'over_color': '#910000', 
         'under_color': '#3a3d48',
         'bad_color': 'gray', 
         'bad_alpha': 0.5, 
         'cmap': 'IMERG_liquid',
         'cmap_type': "LinearSegmented",
         'levels': [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50], 
         "extend": "max"
         # TODO: Add label
        },
    'GPM_Z': 
        {'bad_color': 'gray', 
         'bad_alpha': 0.5, 
         'cmap': 'Spectral_r',
         # 'cmap_n': 10, 
         'cmap_type': "Colormap",
         "vmin": 10,
         "vmax": 40,
         "extend": "both",
         'extendfrac': 0.05, 
         "label": 'Reflectivity [$dBZ$]', #$Z_{e}$
         },
    'GPM_DFR': 
        {'bad_color': 'gray', 
         'bad_alpha': 0.5, 
         'cmap': 'turbo',
         # 'cmap_n': 10, 
         'cmap_type': "Colormap",
         "vmin": -2,
         "vmax": 10,
         "extend": "both",
         'extendfrac': 0.05, 
         "label": '$DFR_{Ku-Ka}$, [$dB$]',
         },
    'GPM_Dm': 
        {'bad_color': 'gray', 
         'bad_alpha': 0.5, 
         'cmap': 'plasma',
         # 'cmap_n': 10, 
         'cmap_type': "Colormap",
         "vmin": 0,
         "vmax": 2,
         "extend": "both",
         "label": '$D_m$ [$mm$]', 
         },
    'GPM_Nw': 
        {'bad_color': 'gray', 
         'bad_alpha': 0.5, 
         'cmap': 'plasma',
         # 'cmap_n': 10, 
         'cmap_type': "Colormap",
         "vmin": -1,
         "vmax": 6,
         "extend": "both",
         "label": '$N_w$ [$\log{(mm^{-1} \ m^{-3})}$]',
         },    
    'GPM_LatentHeating': 
        {'bad_color': 'gray', 
         'bad_alpha': 0.5,
         'cmap': 'RdBu_r',
         'cmap_type': "Colormap",
         'norm': "SymLogNorm",
         "linthresh": 1, 
         "base": 10, 
         "vmin": -400,
         "vmax": 400,
         'extend': "both", 
         'extendfrac': 0.05, 
         "label": "Latent Heating [K/hr]",
         },   
    'Precip_Probability': 
        {'over_color':  'none', 
         'under_color': 'none',
         'bad_color': 'gray', 
         'bad_alpha': 0.5, 
         'cmap': 'OrRd',
         'cmap_n': 10, 
         'cmap_type': "Colormap",
         'levels': [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ],
         # 0.001 to set 0 to transparent 
        },
    'BOM-RF3_mm/hr': 
        {'over_color': 'black',
         'bad_color': 'gray', 
         'bad_alpha': 0.5, 
         'cmap': 'BOM-RF3', 
         'cmap_type': "LinearSegmented",
         'levels': [0.0, 0.2, 0.5, 1.5, 2.5, 4, 6, 10, 15, 20, 30, 40, 50, 60, 75, 100, 150],
        },
    'BOM-RF3_mm': 
        {'over_color': 'black',
         'bad_color': 'gray', 
         'bad_alpha': 0.5, 
         'cmap': 'BOM-RF3', 
         'cmap_type': "LinearSegmented",
         'levels': [0.0, 0.2, 0.5, 1.5, 2.5, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        },
    'pysteps_mm': 
         {'over_color': 'darkred',  
         'bad_color': 'gray', 
         'bad_alpha': 0.5, 
         'cmap': 'pysteps',
         'cmap_type': "LinearSegmented",
         'levels': [0.08, 0.16, 0.25, 0.4, 0.63, 1, 1.6, 2.5, 4, 6.3, 10, 16, 25, 40, 63, 100, 160],
         },
    'pysteps_mm/hr': 
        {'over_color': 'darkred',  
         'under_color': "none", 
         'bad_color': 'gray', 
         'bad_alpha': 0.2, 
         'cmap': 'pysteps',
         'cmap_type': "LinearSegmented",
         'levels': [0.08, 0.16, 0.25, 0.4, 0.63, 1, 1.6, 2.5, 4, 6.3, 10, 16, 25, 40, 63, 100, 160],
         'extend': 'max',
         'extendrect': False,
         'label': 'Precipitation intensity [$mm \ hr^{-1}$]', 
         },
    'pysteps_dBZ': 
        {'over_color': 'darkred', 
         'bad_color': 'gray', 
         'bad_alpha': 0.5, 
        'cmap': 'pysteps', 
        'cmap_type': "LinearSegmented",
        'levels': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        },  
    'STEPS-BE_mm': 
        {'over_color': 'black', 
         'bad_color': 'gray', 
         'bad_alpha': 0.5, 
         'cmap': 'STEPS-BE', 
         'cmap_type': "LinearSegmented",
         'levels': [0.1, 0.25, 0.4, 0.63, 1, 1.6, 2.5, 4, 6.3, 10, 16, 25, 40, 63, 100],
        },
    'STEPS-BE_mm/hr': 
        {'over_color': 'black', 
         'bad_color': 'gray', 
         'bad_alpha': 0.5, 
         'cmap': 'STEPS-BE', 
         'cmap_type': "LinearSegmented",
         'levels': [0.1, 0.25, 0.4, 0.63, 1, 1.6, 2.5, 4, 6.3, 10, 16, 25, 40, 63, 100],
        },
    'STEPS-BE_mm/hr': 
        {'over_color': 'black',
         'bad_color': 'gray', 
         'bad_alpha': 0.5, 
         'cmap': 'STEPS-BE', 
         'cmap_type': "LinearSegmented",
         'levels': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        },
} 
       
    
####-------------------------------------------------------------------------.


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


####--------------------------------------------------------------------------.


def get_colormap_setting(cbar_settings_name):
    # TODO: --> Accept other kwargs, check it,  and modify at the end ... 
    if not isinstance(cbar_settings_name, (str, type(None))):
        raise TypeError("Expecting the colorscale name.")
        
    # Retrieve colormap and colorbar information 
    color_dict = COLOR_DICT.get(cbar_settings_name, None)
    if color_dict is None: 
        color_dict = {}
        color_dict["cmap"] = "jet"
        color_dict["cmap_type"] = "Colormap"
        if cbar_settings_name is None:
            print("Returning default colorbar settings.")
        else: 
            color_dict["cmap"] = cbar_settings_name
    
        # raise ValueError("{cbar_settings_name} cbar_settings  does not exist.")   
    
    ####---------------------------------------------------------------------.
    # Get cmap 
    cmap_type = color_dict['cmap_type']
    cmap_name = color_dict['cmap']
    # Get colorbar levels (if available)
    clevs = color_dict.get("levels",None)
    #------------------------------------------------------------------------.
    if cmap_type == "LinearSegmented":
        # TODO: Check level is a list > 2, length + 1 than color_list 

        # Get color list 
        color_list = CMAP_DICT[cmap_name]['color_list']

        # Get colormap       
        cmap = mpl.colors.LinearSegmentedColormap.from_list("cmap", 
                                                        color_list,
                                                        len(clevs) - 1
        )
  
    #------------------------------------------------------------------------.
    # TODO: implement other cmap options 
    elif cmap_type == "Colormap": 
        # Get colormap  
        cmap_n = color_dict.get('cmap_n', None)
        cmap = copy.copy(plt.get_cmap(cmap_name, cmap_n))
        vmin = color_dict.get('vmin', None)
        vmax = color_dict.get('vmax', None)
        norm = None
    # matplotlib.colors.NoNorm
    else: 
        ticks = None
        # vmin and vmax to be defined 
        raise NotImplementedError
   
    ####---------------------------------------------------------------------.  
    # Set norm if specified 
    if color_dict.get('norm', None): 
        if color_dict['norm'] == "SymLogNorm": 
            norm = mpl.colors.SymLogNorm(linthresh=color_dict["linthresh"],
                                         vmin=vmin, vmax=vmax,  
                                         base=color_dict.get("base",10))
        else:
            raise NotImplementedError()
    
    ####---------------------------------------------------------------------.
    # Define BoundaryNorm 
    if clevs is not None: 
        norm = mpl.colors.BoundaryNorm(clevs, cmap.N)
        vmin = None # cartopy and matplotlib complain if not None when norm is provided !
        vmax = None
        ticks = clevs 
    else: 
        ticks = None 
        # vmin and vmax to be defined 
   
    ####---------------------------------------------------------------------.
    # Define ticklabels 
    # - Generate color level strings with correct amount of decimal places
    if clevs is not None: 
        ticklabels = _dynamic_formatting_floats(ticks)
        # clevs_str = [f"{tick:.1f}" for tick in ticks] # for 0.1 probability
    else: 
        ticklabels = None 
     
    ####---------------------------------------------------------------------.
    # Set over, under and alpha 
    # If not specified, do not set ---> Will fill with the first/last color value 
    # If 'none' --> It will be depicted in white 
    if color_dict.get("over_color", None):
        cmap.set_over(color=color_dict.get("over_color"),
                      alpha=color_dict.get("over_alpha", None))
    if color_dict.get("over_color", None):
        cmap.set_under(color=color_dict.get("under_color"),
                       alpha=color_dict.get("under_alpha", None))
    #------------------------------------------------------------------------.
    # Set bad color 
    # - If not 0, can cause cartopy bug 
    # --> https://stackoverflow.com/questions/60324497/specify-non-transparent-color-for-missing-data-in-cartopy-map
    cmap.set_bad(color=color_dict.get("bad_color", "none"),
                 alpha=color_dict.get("bad_alpha", None))
     
    #------------------------------------------------------------------------.
    #### Set cbar kwargs 
    cbar_kwargs = {
        'ticks': color_dict.get("ticks", ticks),        
        'spacing': color_dict.get("spacing", "uniform"), # or proportional
        'extend': color_dict.get("extend", "neither"),
        'extendfrac': color_dict.get("extendfrac", "auto"),
        'extendrect': color_dict.get("extendrect", False),
        'label': color_dict.get("label", None),
        'drawedges': color_dict.get("drawedges", False),
        'ticklocation': color_dict.get("ticklocation", 'auto'),
        'shrink': color_dict.get("ticklocation", 1),
    }     
    # format 
    # 'orientation':'horizontal'
    # 'aspect':40,
    # filled=True
    
    # ticklocation{'auto', 'left', 'right', 'top', 'bottom'}
    # extend{'neither', 'both', 'min', 'max'}
    # extendfrac  # {None, 'auto', length, lengths} 
    # extendrect  # True or False 
    # drawedges  # True or False
    # 'label': 
    
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html            
    # https://matplotlib.org/stable/api/colorbar_api.html
  
    # set_ticks 
    # set_ticklabels
    # levels --> for colormap 
    # ticks --> for adding points on the colorbar 
    
    #------------------------------------------------------------------------.
    # Define plot kwargs  
    # - In xarray, cbar_kwargs could be inserted in plot_kwargs
    plot_kwargs = {
        "cmap": cmap, 
        "norm": norm,
        "vmin": vmin, 
        "vmax": vmax, 
        }
    #------------------------------------------------------------------------.
    return plot_kwargs, cbar_kwargs,  ticklabels
 
    
####---------------------------------------------------------------------------.
def get_colormap(ptype, units="mm/hr", colorscale="pysteps"):
    """Function to generate a colormap (cmap) and norm.
    Parameters
    ----------
    ptype : {'intensity', 'depth', 'prob'}, optional
        Type of the map to plot: 'intensity' = precipitation intensity field,
        'depth' = precipitation depth (accumulation) field,
        'prob' = exceedance probability field.
    units : {'mm/hr', 'mm', 'dBZ'}, optional
        Units of the input array. If ptype is 'prob', this specifies the unit of
        the intensity threshold.
    colorscale : {'pysteps', 'STEPS-BE', 'BOM-RF3'}, optional
        Which colorscale to use. Applicable if units is 'mm/hr', 'mm' or 'dBZ'.
        
    Returns
    -------
    cmap : Colormap instance
        colormap
    norm : matplotlibcolors.Normalize object
        Colors norm
    clevs: list(float)
        List of precipitation values defining the color limits.
    clevs_str: list(str)
        List of precipitation values defining the color limits (with correct
        number of decimals).
    """
    if ptype in ["intensity", "depth"]:
        cbar_settings_name = colorscale + "_" + units
        plot_kwargs, cbar_kwargs, ticklabels = get_colormap_setting(cbar_settings_name)
        cbar_kwargs['extend'] = "max" 
      
    elif ptype == "prob":
        plot_kwargs, cbar_kwargs, ticklabels = get_colormap_setting("Precip_Probability")
        cbar_kwargs['extend'] = "neither"
    else: 
        plot_kwargs = {"cmap": plt.cm.get_cmap("jet"), 
                       "norm": mpl.colors.Normalize(), 
                       }
        cbar_kwargs = {}
        ticklabels = None
       
    return plot_kwargs, cbar_kwargs,  ticklabels


####---------------------------------------------------------------------------.
#################
### Examples ####
#################
# get colormap and color levels
# colorscale = "STEPS-BE" 
# colorscale = "BOM-RF3"
# colorscale= "IMERG_liquid"
# colorscale= "IMERG_solid"
# colorscale = "pysteps"


# plot_kwargs, cbar_kwargs, ticklabels = get_colormap_setting("IMERG_Liquid")
# plot_kwargs, cbar_kwargs, ticklabels = get_colormap_setting("pysteps_mm/hr")


# plot_kwargs, cbar_kwargs,  ticklabels = get_colormap(ptype="default",
#                                                      units="mm/hr",
#                                                      colorscale="pysteps")


# plot_kwargs, cbar_kwargs,  ticklabels = get_colormap(ptype="intensity",
#                                                      units="mm/hr",
#                                                      colorscale="pysteps")


# plot_kwargs, cbar_kwargs,  ticklabels = get_colormap(ptype="prob",
#                                                      units="mm/hr",
#                                                      colorscale="pysteps")
 

# p = da_precip_subset.plot.imshow(x="along_track", y="cross_track",
#                                  interpolation="bilinear", # "nearest", "bicubic"
#                                    cbar_kwargs=cbar_kwargs, 
#                                    **plot_kwargs)
# cbar = p.colorbar
# cbar.ax.set_yticklabels(ticklabels)                 




