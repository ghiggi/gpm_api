#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:46:28 2023

@author: ghiggi
"""
import numpy as np
import gpm_api
fpath = "/ltenas8/data/GPM/RS/V07/RADAR/2A-DPR/2020/07/15/2A.GPM.DPR.V9-20211125.20200715-S130215-E143448.036245.V07A.HDF5"
fpath = "/ltenas8/data/GPM/RS/V07/RADAR/2A-DPR/2020/07/20/2A.GPM.DPR.V9-20211125.20200720-S023603-E040838.036316.V07A.HDF5"
fpath = "/ltenas8/data/GPM/RS/V07/RADAR/2A-DPR/2020/07/20/2A.GPM.DPR.V9-20211125.20200720-S180156-E193431.036326.V07A.HDF5"

ds_gpm = gpm_api.open_granule(fpath)
list_regular_slices = ds_gpm.gpm_api.get_contiguous_scan_slices()
print(list_regular_slices)
slc = list_regular_slices[0]

ds_1 = ds_gpm.isel(along_track=list_regular_slices[0])
ds_2 = ds_gpm.isel(along_track=list_regular_slices[1])

 # - scanStatus
 #   - geoError
 #   - missing
 #   - modeStatus
 #   - geoWarning
 
np.unique(ds_gpm["scanStatus/operationalMode"].isel(frequency=0).data.compute(), return_counts=True)  # 1, 3, 4, 5  -99
np.unique(ds_gpm["scanStatus/geoWarning"].isel(frequency=0).data.compute(), return_counts=True) # 0, 8
np.unique(ds_gpm["scanStatus/geoError"].isel(frequency=0).data.compute(), return_counts=True)  # 0
np.unique(ds_gpm["scanStatus/modeStatus"].isel(frequency=0).data.compute(), return_counts=True)  # 0, 16, 22
np.unique(ds_gpm["scanStatus/missing"].isel(frequency=0).data.compute(), return_counts=True)  # 0, 3, 9
np.unique(ds_gpm["scanStatus/dataWarning"].isel(frequency=0).data.compute(), return_counts=True)  # 0, 8
np.unique(ds_gpm["scanStatus/dataQuality"].isel(frequency=0).data.compute(), return_counts=True)  # 0, 1, 64, 65

np.unique(ds_gpm["scanStatus/SCorientation"].data.compute()) # 180, -8000, 0

# get slices of contiguous granules 
#   get slices of contiguous scans


ds_gpm.data_vars
