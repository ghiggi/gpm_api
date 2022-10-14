#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 23:02:04 2022

@author: ghiggi
"""
#-----------------------------------------------------------------------------.
# dpr 
# imerg
# pmw

# gpm = gpm_api.GPM(...)

# PR 
# DPR
# PMW 
# CMB
# IMERG

#-----------------------------------------------------------------------------.
#### GPM High-Level Classes
# Swath CLASS 
# --> TRMM PR 
# --> GPM DPR 
# --> GPM CMB 
# -> pyresample.SwathDef 

# Grid CLASS 
# --> GPM IMERG 
# -> pyresample.AreaDef 

#-----------------------------------------------------------------------------.

# 2A-ENV-DPR --> 2A-DPR-ENV ? 
# Ka--> KA , Ku --> KU ?

#-----------------------------------------------------------------------------.
# Add cross_track_id for GPM !!!!
# Add range_id 

### Add granule_id coord (time) to xr.Dataset

#-----------------------------------------------------------------------------.
### After download, implement loop to check if file is corrupted (open with h5py) 

### Implement function that check no missing timesteps


### Filter files by version 

### get_disk_fpaths_from_pps_fpaths
#-----------------------------------------------------------------------------.

# Improve decoding and DataArray attributes 
# --> Decode before or after each Dataset 


