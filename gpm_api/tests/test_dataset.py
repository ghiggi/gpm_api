#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:06:51 2022

@author: ghiggi
"""
from gpm_api.dataset.reader import open_granule, open_dataset

scan_mode = None
groups = None
variables = None
decode_cf = False
chunks = "auto"
prefix_group = True

filepath = "/home/ghiggi/GPM/RS/V06/IMERG/IMERG-FR/2019/07/13/3B-HHR.MS.MRG.3IMERG.20190713-S123000-E125959.0750.V06B.HDF5"


# IMERG-EARLY
# 3B-HHR-E.MS.MRG.3IMERG.20221201-S020000-E022959.0120.V06C.RT-H5
# # IMERG-LATE
# 3B-HHR-L.MS.MRG.3IMERG.20221201-S023000-E025959.0150.V06C.RT-H5
# # IMERG FINAL
# 3B-HHR.MS.MRG.3IMERG.20190713-S123000-E125959.0750.V06B.HDF5
