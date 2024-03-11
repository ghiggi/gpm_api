#!/usr/bin/env python3
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

# -----------------------------------------------------------------------------.
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

# -----------------------------------------------------------------------------.
# cmaps[name] = mpl.colorsListedColormap(data, name=name)

# --------------------------------------------------------------------------.
# ListedColormap, which produces a discrete colormap,
# dir(matplotlib.colors)

# # Continuous colormap
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", color_list)
# norm = mpl.colors.Normalize(vmin, vmax)

# Discrete colormap
# cmap = mpl.colors.LinearSegmentedColormap.from_list("cmap", color_list, len(clevs) - 1)
# norm = mpl.colors.BoundaryNorm(clevs, cmap.N)

# https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
# Look at pycolorbar existing code

# --------------------------------------------------------------------------.
import copy

import matplotlib
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pylab as plt
import numpy as np

PRECIP_VALID_TYPES = ("intensity", "depth", "prob")
PRECIP_VALID_UNITS = ("mm/h", "mm", "dBZ")

####--------------------------------------------------------------------------.
#### CMAP DICTIONARY

CMAP_DICT = {
    "IMERG_Solid": {
        "type": "hex",
        "color_list": [
            "#45ffff",  # [0.1 - 0.2]
            "#49ddff",  # [0.2 - 0.3]
            "#4dbbff",  # [0.3 - 0.5]
            "#4f9aff",  # [0.5 - 1.0]
            "#517aff",  # [1.0 - 2.0]
            "#525dff",  # [2.0 - 3.0]
            "#5346ff",  # [3.0 - 5.0]
            "#5a2fd9",  # [5.0 - 10.0]
            "#7321bb",  # [10.0 - 20.0]
            "#8c149c",  # [20.0 - 50.0]
        ],
    },
    "IMERG_Liquid": {
        "type": "hex",
        "color_list": [
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
    "GOES_CloudPhase": {
        "type": "name",
        "color_list": [
            "lightblue",  # Clear Sky
            "darkblue",  # Liquid
            "lightgreen",  # SC Liquid
            "orange",  # Mixed
            "white",  # Ice
            "darkgray",  # Unknown
        ],
    },
    "GOES_BinaryCloudMask": {
        "type": "name",
        "color_list": [
            "lightblue",  # Clear Sky
            "darkgray",  # Cloudy
        ],
    },
    "pysteps": {
        "type": "hex",
        "color_list": [
            "#9c7e94",  # redgray or '#e8d7f2' pink
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
        "color_list": [
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
        "color_list": np.array(
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
                (40, 0, 0),  # 100
            ]
        )
        / 255,  # convert to 0-1
    },
}

####--------------------------------------------------------------------------.
#### GPM COLOR DICTIONARY


COLOR_DICT = {
    "IMERG_Solid": {
        "over_color": "#8c149c",
        "under_color": "none",  # "#3a3d48",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "IMERG_Solid",
        "cmap_type": "LinearSegmented",
        "levels": [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50],
        "extend": "max",
        "label": "Precipitation intensity [$mm \\ hr^{-1}$]",
    },
    "IMERG_Liquid": {
        "over_color": "#910000",
        "under_color": "none",  # "#3a3d48",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "IMERG_Liquid",
        "cmap_type": "LinearSegmented",
        "levels": [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50],
        "extend": "max",
        "label": "Precipitation intensity [$mm \\ hr^{-1}$]",
    },
    "GPM_Z": {
        "bad_color": "none",
        "bad_alpha": 0.5,
        "cmap": "Spectral_r",
        # 'cmap_n': 10,
        "cmap_type": "Colormap",
        "vmin": 10,
        "vmax": 50,
        "extend": "both",
        "extendfrac": 0.05,
        "label": "Reflectivity [$dBZ$]",  # $Z_{e}$
    },
    "GPM_DFR": {
        "bad_color": "none",
        "cmap": "turbo",
        # 'cmap_n': 10,
        "cmap_type": "Colormap",
        "vmin": -2,
        "vmax": 10,
        "extend": "both",
        "extendfrac": 0.05,
        "label": "$DFR_{Ku-Ka}$, [$dB$]",
    },
    "GPM_Dm": {
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "plasma",
        # 'cmap_n': 10,
        "cmap_type": "Colormap",
        "vmin": 0,
        "vmax": 2,
        "extend": "both",
        "label": "$D_m$ [$mm$]",
    },
    "GPM_Nw": {
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "plasma",
        # 'cmap_n': 10,
        "cmap_type": "Colormap",
        "vmin": -1,
        "vmax": 6,
        "extend": "both",
        "label": "$N_w$ [$\\log{(mm^{-1} \\ m^{-3})}$]",
    },
    "latentHeating": {
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "RdBu_r",
        "cmap_type": "Colormap",
        "norm": "SymLogNorm",
        "linthresh": 1,
        "base": 10,
        "vmin": -400,
        "vmax": 400,
        "extend": "both",
        "extendfrac": 0.05,
        "label": "Latent Heating [K/hr]",
    },
    "Brightness_Temperature": {
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "Spectral_r",
        "cmap_type": "Colormap",
        # "vmin": 140,
        # "vmax": 300,
        "extend": "both",
        "extendfrac": 0.05,
        "label": "Brightness Temperature [K]",
    },
    "Precip_Probability": {
        "over_color": "none",
        "under_color": "none",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "OrRd",
        "cmap_n": 10,
        "cmap_type": "Colormap",
        "levels": [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        # 0.001 to set 0 to transparent
    },
    "BOM-RF3_mm/hr": {
        "over_color": "black",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "BOM-RF3",
        "cmap_type": "LinearSegmented",
        "levels": [
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
        ],
    },
    "BOM-RF3_mm": {
        "over_color": "black",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "BOM-RF3",
        "cmap_type": "LinearSegmented",
        "levels": [
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
        ],
    },
    "pysteps_mm": {
        "over_color": "darkred",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "pysteps",
        "cmap_type": "LinearSegmented",
        "levels": [
            0.08,
            0.16,
            0.25,
            0.4,
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
        ],
    },
    "pysteps_mm/hr": {
        "over_color": "darkred",
        "under_color": "none",
        "bad_color": "gray",
        "bad_alpha": 0.2,
        "cmap": "pysteps",
        "cmap_type": "LinearSegmented",
        "levels": [
            0.08,
            0.16,
            0.25,
            0.4,
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
        ],
        "extend": "max",
        "extendrect": False,
        "label": "Precipitation intensity [$mm \\ hr^{-1}$]",
    },
    "pysteps_dBZ": {
        "over_color": "darkred",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "pysteps",
        "cmap_type": "LinearSegmented",
        "levels": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    },
    "STEPS-BE_mm": {
        "over_color": "black",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "STEPS-BE",
        "cmap_type": "LinearSegmented",
        "levels": [0.1, 0.25, 0.4, 0.63, 1, 1.6, 2.5, 4, 6.3, 10, 16, 25, 40, 63, 100],
    },
    # "STEPS-BE_mm/hr": {
    #     "over_color": "black",
    #     "bad_color": "gray",
    #     "bad_alpha": 0.5,
    #     "cmap": "STEPS-BE",
    #     "cmap_type": "LinearSegmented",
    #     "levels": [0.1, 0.25, 0.4, 0.63, 1, 1.6, 2.5, 4, 6.3, 10, 16, 25, 40, 63, 100],
    # },
    "STEPS-BE_mm/hr": {
        "over_color": "black",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "STEPS-BE",
        "cmap_type": "LinearSegmented",
        "levels": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    },
    "precipWaterIntegrated": {
        "under_color": "none",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "YlGnBu",
        "cmap_type": "Colormap",
        "vmin": 0.1,  # kg/m2
        "vmax": 20,  # 3000
        "extend": "max",
        "extendfrac": 0.05,
        "label": "Total Precipitable Water [$kg \\ m^{-2}$]",
    },
    "totalWaterPath": {
        "under_color": "none",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "YlGnBu",
        "cmap_type": "Colormap",
        "vmin": 0.1,  # kg/m2
        "vmax": 20,  # 3000
        "extend": "max",
        "extendfrac": 0.05,
        "label": "Total Precipitable Water [$kg \\ m^{-2}$]",
    },
    "rainWaterPath": {
        "under_color": "none",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "YlGnBu",
        "cmap_type": "Colormap",
        "vmin": 0.1,  # kg/m2
        "vmax": 5,  # 3000
        "extend": "max",
        "extendfrac": 0.05,
        "label": "Rain Water Path [$kg \\ m^{-2}$]",
    },
    "cloudWaterPath": {
        "under_color": "none",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "YlGnBu",
        "cmap_type": "Colormap",
        "vmin": 0.1,  # kg/m2
        "vmax": 1,  # 3000
        "extend": "max",
        "extendfrac": 0.05,
        "label": "Cloud Liquid Water Path [$kg \\ m^{-2}$]",
    },
    "cloudLiquidWaterPath": {
        "under_color": "none",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "YlGnBu",
        "cmap_type": "Colormap",
        "vmin": 0.1,  # kg/m2
        "vmax": 1,  # 3000
        "extend": "max",
        "extendfrac": 0.05,
        "label": "Cloud Liquid Water Path [$kg \\ m^{-2}$]",
    },
    "liquidWaterPath": {
        "under_color": "none",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "YlGnBu",
        "cmap_type": "Colormap",
        "vmin": 0.5,  # kg/m2
        # "vmax": 3000,
        "extend": "max",
        "extendfrac": 0.05,
        "label": "Liquid Water Path [$kg \\ m^{-2}$]",
    },
    "iceWaterPath": {
        "under_color": "none",
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "YlGnBu",
        "cmap_type": "Colormap",
        "vmin": 0.5,  # kg/m2
        "vmax": 10,  # 3000
        "extend": "max",
        "extendfrac": 0.05,
        "label": "Ice Water Path [$kg \\ m^{-2}$]",
    },
}


precip_variables = [
    # 2A-<RADAR>
    "precipRate",
    "precipRateNearSurface",
    "precipRateESurface",
    "precipRateESurface2",
    "precipRateAve24",
    # 2B-<RADAR>-CORRA
    "precipTotRate",
    "nearSurfPrecipTotRate",
    "nearSurfPrecipLiqRate",
    "estimSurfPrecipTotRate",
    "OEestimSurfPrecipTotRate",
    "estimSurfPrecipLiqRate",
    "OEestimSurfPrecipLiqRate",
    # 2A <PMW>
    "surfacePrecipitation",
    "mostLikelyPrecipitation",
    "convectivePrecipitation",
    "frozenPrecipitation",  # has really small values !
    # 2A-<RADAR>-SLH
    "nearSurfacePrecipRate",
    # 2B-<RADAR>-CSH
    "surfacePrecipRate",
    # IMERG
    "precipitationCal",
    "precipitationUncal",
    "IRprecipitation",
    # GEO RRQPE
    "RRQPE",
]

for var in precip_variables:
    COLOR_DICT[var] = COLOR_DICT["pysteps_mm/hr"]

reflectivity_variables = [
    "zFactorFinalNearSurface",
    "zFactorFinalESurface",
    "zFactorFinalNearSurface",
    "zFactorMeasured",
    "zFactorFinal",
    "REFC",
]
for var in reflectivity_variables:
    COLOR_DICT[var] = COLOR_DICT["GPM_Z"]

for var in ["dfrMeasured", "dfrFinal", "dfrFinalNearSurface"]:
    COLOR_DICT[var] = COLOR_DICT["GPM_DFR"]


COLOR_DICT["Tb"] = COLOR_DICT["Brightness_Temperature"]
COLOR_DICT["Tc"] = COLOR_DICT["Brightness_Temperature"]
COLOR_DICT["simulatedBrightTemp"] = COLOR_DICT["Brightness_Temperature"]


####--------------------------------------------------------------------------.
#### GOES DICTIONARY


GOES_DICT = {
    "COD": {
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "viridis",
        "cmap_type": "Colormap",
        "vmin": 0,
        "vmax": 50,
        "extend": "max",
        "extendfrac": 0.05,
        "label": "Cloud Optical Depth at 640 nm [-]",
    },
    "CPS": {
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "turbo_r",
        "cmap_type": "Colormap",
        "vmin": 0,
        "vmax": 50,
        "extend": "max",
        "extendfrac": 0.05,
        "label": "Cloud Particle Size [um]",
    },
    "CTH": {
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "viridis",
        "cmap_type": "Colormap",
        "vmin": 0,
        "vmax": 15000,
        "extend": "max",
        "extendfrac": 0.05,
        "label": "Cloud Top Height [m]",
    },
    "CTT": {
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "RdYlBu_r",
        "cmap_type": "Colormap",
        "vmin": 200,
        "vmax": 300,
        "extend": "both",
        "extendfrac": 0.05,
        "label": "Cloud Top Temperature [K]",
    },
    "CTP": {
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "RdYlBu_r",
        "cmap_type": "Colormap",
        "vmin": 100,
        "vmax": 1000,
        "extend": "both",
        "extendfrac": 0.05,
        "label": "Cloud Top Pressure [hPa]",
    },
    "Phase": {
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "GOES_CloudPhase",
        "cmap_type": "Categorical",
        "labels": ["Clear Sky", "Liquid", "SC Liquid", "Mixed", "Ice", "Unknown"],
    },
    "CM": {
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "GOES_BinaryCloudMask",
        "cmap_type": "Categorical",
        "labels": ["Clear Sky", "Cloudy"],
    },
}
COLOR_DICT.update(GOES_DICT)

# ABI Reflectance channels (C01-C06)
reflectance_channels = ["C" + str(i).zfill(2) for i in range(1, 7)]

for channel in reflectance_channels:
    GOES_DICT[channel] = {
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "gray",
        "cmap_type": "Colormap",
        "vmin": 0,
        "vmax": 100,
        "extend": "neither",
        "extendfrac": 0.05,
        "label": "Reflectance [%]",
    }


# ABI BT channels (C07-C17)
bt_channels = ["C" + str(i).zfill(2) for i in range(7, 17)]
for channel in bt_channels:
    GOES_DICT[channel] = {
        "bad_color": "gray",
        "bad_alpha": 0.5,
        "cmap": "RdYlBu_r",
        "cmap_type": "Colormap",
        "vmin": 200,
        "vmax": 300,
        "extend": "both",
        "extendfrac": 0.05,
        "label": "Brightness Temperature [K]",
    }


# DQF channels colormaps
default_dqf = {
    "under_color": "none",
    "bad_color": "gray",
    "bad_alpha": 0.5,
    "cmap": "viridis",
    "cmap_type": "Colormap",
    "vmin": 1,  # 0 is transparent with under_color="none"
    "extend": "neither",
    "extendfrac": 0.05,
    "label": "Data Quality Flag",
}
for dqf_name in ["DQF_C" + str(i).zfill(2) for i in range(1, 17)]:
    GOES_DICT[dqf_name] = default_dqf.copy()
    GOES_DICT[dqf_name]["vmax"] = 4

# DQF CLOUDS colormaps
for dqf_name in ["DQF_" + var for var in ["COD", "CPS"]]:
    GOES_DICT[dqf_name] = default_dqf.copy()
    GOES_DICT[dqf_name]["vmax"] = 16

for dqf_name in ["DQF_" + var for var in ["CTT", "CTH", "CTP"]]:
    GOES_DICT[dqf_name] = default_dqf.copy()
    GOES_DICT[dqf_name]["vmax"] = 6

for dqf_name in ["DQF_" + var for var in ["Phase"]]:
    GOES_DICT[dqf_name] = default_dqf.copy()
    GOES_DICT[dqf_name]["vmax"] = 5

for dqf_name in ["DQF_" + var for var in ["RRQPE"]]:
    GOES_DICT[dqf_name] = default_dqf.copy()
    GOES_DICT[dqf_name]["vmax"] = 64

for dqf_name in ["DQF_" + var for var in ["CM"]]:
    GOES_DICT[dqf_name] = default_dqf.copy()
    GOES_DICT[dqf_name]["vmax"] = 2

####--------------------------------------------------------------------------.
#### Final color dictionary

COLOR_DICT.update(GOES_DICT)

# Reflectivity
# "2A-DPR":
#     "zFactorFinalESurface",
#     "zFactorFinalNearSurface",
#     "zFactorFinal",

# 2A-DPR
# "airTemperature",
# "landSurfaceType",

# # '2A-ENV-DPR':
# "cloudLiquidWater",
# "waterVapor",
# "airPressure"],

# # 2A-GMI
# "rainWaterPath",
# "cloudWaterPath",
# "iceWaterPath",

# # '2B-GPM-CORRA': [
# "precipTotWaterCont",
# "cloudIceWaterCont",
# "cloudLiqWaterCont",
# "OEcolumnCloudLiqWater",
# "OEcloudLiqWaterCont",
# "OEcolumnWaterVapor"

# # 1B
# "Tb",
# # 1C
# "Tc",
# # '2B-GPM-CORRA'
# "OEsimulatedBrightTemp",

# Latent Heat
# '2B-GPM-CSH': ["latentHeating"],
# '2A-GPM-SLH': ["latentHeating"],


# -----------------------------------------------------------------------------.


def _dynamic_formatting_floats(float_array, colorscale="pysteps"):
    """Function to format the floats defining the class limits of the colorbar."""
    float_array = np.array(float_array, dtype=float)

    labels = []
    for label in float_array:
        if 0.1 <= label < 1:
            formatting = ",.2f" if colorscale == "pysteps" else ",.1f"
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


# --------------------------------------------------------------------------.


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

    # --------------------------------------------------------------------------.
    # Get cmap info
    cmap_type = color_dict["cmap_type"]
    cmap_name = color_dict["cmap"]
    clevs = color_dict.get("levels", None)
    vmin = color_dict.get("vmin", None)
    vmax = color_dict.get("vmax", None)

    # Initialize cbar_kwargs
    cbar_kwargs = {}

    # ------------------------------------------------------------------------.
    if cmap_type == "LinearSegmented":
        # TODO: Check level is a list > 2, length + 1 than color_list

        # Get color list
        color_list = CMAP_DICT[cmap_name]["color_list"]

        # Get colormap
        cmap = mpl.colors.LinearSegmentedColormap.from_list("cmap", color_list, len(clevs) - 1)

    # ------------------------------------------------------------------------.
    elif cmap_type == "Categorical":
        # Get color list
        color_list = CMAP_DICT[cmap_name]["color_list"]

        # Get class labels
        labels = color_dict.get("labels", None)

        # Check validity
        if labels is None:
            raise ValueError(f"If cmap_type is {cmap_type}, 'labels' list is required.")
        if not isinstance(labels, list):
            raise ValueError("'labels' must be a list.")
        if len(color_list) != len(labels):
            raise ValueError("'labels' must have same length as the 'color_list'.")

        # Define colormap and colorbar settings
        nlabels = len(labels)
        cmap = mpl.colors.ListedColormap(color_list)

        # Define norms
        category_first_value = 9  # TODO: optional parameter
        norm_bins = np.arange(category_first_value - 1, nlabels + category_first_value) + 0.5
        norm = mpl.colors.BoundaryNorm(norm_bins, nlabels)

        # Update cbar_kwargs
        fmt = mpl.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
        tickz = norm_bins[:-1] + 0.5
        cbar_kwargs["format"] = fmt
        cbar_kwargs["ticks"] = tickz

    # ------------------------------------------------------------------------.
    # TODO: implement other cmap options
    elif cmap_type == "Colormap":
        # Get colormap
        cmap_n = color_dict.get("cmap_n", None)
        cmap = copy.copy(plt.get_cmap(cmap_name, cmap_n))
        vmin = color_dict.get("vmin", None)
        vmax = color_dict.get("vmax", None)
        norm = None

    # matplotlib.colors.NoNorm
    else:
        ticks = None
        # vmin and vmax to be defined
        raise NotImplementedError

    # -------------------------------------------------------------------------.
    # Set norm if specified
    if color_dict.get("norm", None):
        if color_dict["norm"] == "SymLogNorm":
            norm = mpl.colors.SymLogNorm(
                linthresh=color_dict["linthresh"],
                vmin=vmin,
                vmax=vmax,
                base=color_dict.get("base", 10),
            )
        else:
            raise NotImplementedError()

    # -------------------------------------------------------------------------.
    # Define BoundaryNorm
    if clevs is not None:
        norm = mpl.colors.BoundaryNorm(clevs, cmap.N)
        vmin = None  # cartopy and matplotlib complain if not None when norm is provided !
        vmax = None
        ticks = clevs
    else:
        ticks = None
        # vmin and vmax to be defined

    # -------------------------------------------------------------------------.
    # Define ticklabels
    # - Generate color level strings with correct amount of decimal places
    if clevs is not None:
        ticklabels = _dynamic_formatting_floats(ticks)
        # clevs_str = [f"{tick:.1f}" for tick in ticks] # for 0.1 probability
    else:
        ticklabels = None

    # -------------------------------------------------------------------------.
    # Set over, under and alpha
    # If not specified, do not set ---> Will fill with the first/last color value
    # If 'none' --> It will be depicted in white
    if color_dict.get("over_color", None):
        cmap.set_over(color=color_dict.get("over_color"), alpha=color_dict.get("over_alpha", None))
    if color_dict.get("under_color", None):
        cmap.set_under(
            color=color_dict.get("under_color"),
            alpha=color_dict.get("under_alpha", None),
        )
    # ------------------------------------------------------------------------.
    # Set bad color
    # - If not 0, can cause cartopy bug
    # --> https://stackoverflow.com/questions/60324497/specify-non-transparent-color-for-missing-data-in-cartopy-map
    cmap.set_bad(
        color=color_dict.get("bad_color", "none"),
        alpha=color_dict.get("bad_alpha", None),
    )

    # ------------------------------------------------------------------------.
    #### Set cbar kwargs
    default_cbar_kwargs = {
        "ticks": color_dict.get("ticks", ticks),
        "spacing": color_dict.get("spacing", "uniform"),  # or proportional
        "extend": color_dict.get("extend", "neither"),
        "extendfrac": color_dict.get("extendfrac", "auto"),
        "extendrect": color_dict.get("extendrect", False),
        "label": color_dict.get("label", None),
        "drawedges": color_dict.get("drawedges", False),
        "ticklocation": color_dict.get("ticklocation", "auto"),
        "shrink": color_dict.get("ticklocation", 1),
    }
    default_cbar_kwargs.update(cbar_kwargs)

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

    # ------------------------------------------------------------------------.
    # Define plot kwargs
    # - In xarray, cbar_kwargs could be inserted in plot_kwargs
    plot_kwargs = {
        "cmap": cmap,
        "norm": norm,
        "vmin": vmin,
        "vmax": vmax,
    }
    # ------------------------------------------------------------------------.
    return plot_kwargs, default_cbar_kwargs, ticklabels


def get_colorbar_settings(name, plot_kwargs={}, cbar_kwargs={}):
    # Try: partial match if not found !
    try:
        default_plot_kwargs, default_cbar_kwargs, default_ticklabels = get_colormap_setting(name)
        default_cbar_kwargs["ticklabels"] = None
    except:
        default_plot_kwargs = {}
        default_cbar_kwargs = {}

    # If the default is a categorical colormap
    # --> TODO: Set all to none

    # If the default is a segmented colormap (with ticks and ticklabels)
    if default_cbar_kwargs.get("ticks", None) is not None:
        # If specifying a custom vmin, vmax, norm or cmap args, remove  the defaults
        # - The discrete colorbar makes no sense anymore
        if np.any(np.isin(["vmin", "vmax", "norm", "cmap"], list(plot_kwargs.keys()))):
            default_cbar_kwargs.pop("ticks", None)
            default_cbar_kwargs.pop("ticklabels", None)
            default_plot_kwargs.pop("cmap", None)
            default_plot_kwargs.pop("norm", None)
            default_plot_kwargs.pop("vmin", None)
            default_plot_kwargs.pop("vmax", None)

    # If cmap is a string, retrieve colormap
    if isinstance(plot_kwargs.get("cmap", None), str):
        plot_kwargs["cmap"] = plt.get_cmap(plot_kwargs["cmap"])

    # Update defaults with custom kwargs
    default_plot_kwargs.update(plot_kwargs)
    default_cbar_kwargs.update(cbar_kwargs)

    # Return the kwargs
    plot_kwargs = default_plot_kwargs
    cbar_kwargs = default_cbar_kwargs

    # Set default cmap if not available
    if plot_kwargs.get("cmap", None) is None:
        plot_kwargs["cmap"] = plt.get_cmap("viridis")

    # Remove vmin, vmax
    # --> vmin and vmax is not accepted by PolyCollection
    # --> create norm or modify norm accordingly
    norm = plot_kwargs.get("norm", None)
    if norm is None:
        norm = mpl.colors.Normalize(
            vmin=plot_kwargs.pop("vmin", None), vmax=plot_kwargs.pop("vmax", None)
        )
        plot_kwargs["norm"] = norm
    else:
        if "vmin" in plot_kwargs:
            if plot_kwargs["vmin"] is None:
                _ = plot_kwargs.pop("vmin")
            else:
                plot_kwargs["norm"].vmin = plot_kwargs.pop("vmin")

        if "vmax" in plot_kwargs:
            if plot_kwargs["vmax"] is None:
                _ = plot_kwargs.pop("vmax")
            else:
                plot_kwargs["norm"].vmax = plot_kwargs.pop("vmax")

    return (plot_kwargs, cbar_kwargs)
