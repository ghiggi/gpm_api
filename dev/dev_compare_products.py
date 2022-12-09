#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:55:43 2022

@author: ghiggi
"""
import gpm_api
import datetime
import matplotlib
from gpm_api.visualization.comparison import compare_products

matplotlib.rcParams["axes.facecolor"] = [0.9, 0.9, 0.9]
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["legend.facecolor"] = "w"
matplotlib.rcParams["savefig.transparent"] = False

base_dir = "/home/ghiggi"

product_variable_1 = {"2A-DPR": "precipRateNearSurface"}
product_variable_2 = {"2A-GMI": "surfacePrecipitation"}
product_variable_2 = {"2B-GPM-CORRA": "nearSurfPrecipTotRate"}


#### Define analysis time period and bounding box
start_time = datetime.datetime.strptime("2020-10-28 08:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-10-28 09:00:00", "%Y-%m-%d %H:%M:%S")
bbox = [-100, -85, 18, 32]

start_time = datetime.datetime.strptime("2016-03-09 10:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2016-03-09 11:00:00", "%Y-%m-%d %H:%M:%S")
bbox = [-102, -92, 28, 35]

fig = compare_products(
    base_dir=base_dir,
    start_time=start_time,
    end_time=end_time,
    bbox=bbox,
    product_variable_1=product_variable_1,
    product_variable_2=product_variable_2,
    version=7,
    product_type="RS",
)
