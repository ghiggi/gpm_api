#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:56:52 2020

@author: ghiggi
"""
##.----------------------------------------------------------------------------.        
## Download GPM data after May 21 2018

gpm.DPR.plot(timestep, bbox, product)
gpm.ZoomMax(bbox, product, n_scan).plot()
gpm.ZoomMin(bbox.product, n_scan).plot()
gpm.ZoomRandom(min_val, max_val, bbox, product, n_scan).plot()
gpm.isel(), sel() 

##----------------------------------------------------------------------------.
## TODO FEATURES
# Scan filtering using :
# ScanStatus/dataQuality 
# ScanStatus/geoError 
# ScanStatus/modeStatus
# ScanStatus/dataWarning
# ScanStatus/operationalMode
# DataQualityFiltering = {'TotalQualityCode' : ['Good'],  # ”Fair” or ”EG”
           
## Masking functions
# - masking when NA on other variable  
# - drop scan with FLG/qualityFlag low or bad

# Compute height 
#  'alt':(['along_track', 'cross_track','range']
# Height[binRangeNo] ={(binEllipsoid 2A − binRangeNo) × rangeBinSize
# Spatial Resolution 5.2 km (Nadir at the height of 407 km)

# Add optional coordinates
# nscan = hdf5_file_attrs(hdf[scan_mode])['SwathHeader']['NumberScansGranule']
# nray = hdf5_file_attrs(hdf[scan_mode])['SwathHeader']['NumberPixels']
# # TODO: generalize the follow 
# if (scan_mode == 'HS'):
#     nrange = 130 # at HS 
# else:
#     nrange = 260 # at MS, NS

# along_track_ID = np.arange(0,nscan) # do not make sense to add ... is time
# cross_track_ID = np.arange(0,nray)  # make sense to add ???
# range_ID = np.arange(0,nrange)      # make sense to add ???
# altitude   

# 'range': range_ID,
# 'cross_track': cross_track_ID),
# 'height': (['along_track', 'cross_track','range'], altitude)
# 'scan_angle' : (['cross_track'], scan_angle), 

# TODO
# - Subset by country 
# - Subset by lat/lon coord (nearest neighbor over x km)

# Animation / Plot Dates
# - 2018/09/11 : Florence, Isaac, Helene 

# Xradar folder on PPS?

# 2B products 2HCSH, CORRA 

## DOWNLOAD_NRT_DATA 
# - wrapper with NRT set to TRUE 

## Remove NRT product older than X days 


