#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 23:02:04 2022

@author: ghiggi
"""
# Directory GPM_NRT, GPM_RS 
# GPM/version/<RS/NRT>/<PRODUCT>

# Search files without internet connection 


#-----------------------------------------------------------------------------.
# io
# orbit 
# grid 
# utils 
# visualization 
# overpass

# dpr 
# imerg
# pmw

#-----------------------------------------------------------------------------.
#### GPM High-Level Classes
# ORBIT CLASS 
# --> TRMM PR 
# --> GPM DPR 
# --> GPM CMB 
# -> pyresample.SwathDef 

# GRID CLASS 
# --> GPM IMERG 
# -> pyresample.AreaDef 

#-----------------------------------------------------------------------------.

# 2A-ENV-DPR --> 2A-DPR-ENV
# Add cross_track_id for GPM !!!!
# Add range_id 

### Add granule_id coord (time) to xr.Dataset

# Improve decoding and DataArray attributes 
# --> Decode before or after each Dataset 


# Product: 1B-PR
# No data found on PPS on date 2014-08-09 for product 1B-PR
# Download of available GPM 1B-PR product completed.
# Product: 1B-Ka
# No data found on PPS on date 2014-08-09 for product 1B-Ka
# Download of available GPM 1B-Ka product completed.
# Product: 1B-Ku
# No data found on PPS on date 2014-08-09 for product 1B-Ku
