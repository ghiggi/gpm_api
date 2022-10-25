#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 23:02:04 2022

@author: ghiggi
"""
# - SwathDefinition(ds_template['lons'], ds_template['lats']).plot()
# gpm_api lon, lat --> longitude-latitude? 
# ds_gpm.gpm_api.pyresample_area.plot()  # property !!!

#-----------------------------------------------------------------------------.
# If data alread download, print a better message than: 
# --> Now: Download of available GPM 2A-DPR product completed.
# --> Future: All data already on disk 

# import gpm_api 
# gpm_api.download(base_dir='/home/ghiggi/GPM', 
#                  username="gionata.ghiggi@epfl.ch",
#                  product='2A-DPR', 
#                  product_type='RS',
#                  start_time='2020-07-05 00:58:02', 
#                  end_time='2020-07-05 01:14:37',
#                  version=7,
#                  force_download=False, 
#                  verbose=True)


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



#-----------------------------------------------------------------------------.
### After download, implement loop to check if file is corrupted (open with h5py) 

### Implement function that check no missing timesteps


### Filter files by version 

### get_disk_fpaths_from_pps_fpaths
#-----------------------------------------------------------------------------.

# Improve decoding and DataArray attributes 
# --> Decode before or after each Dataset 

