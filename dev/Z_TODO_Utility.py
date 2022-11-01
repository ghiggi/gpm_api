#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:27:55 2022

@author: ghiggi
"""
##----------------------------------------------------------------------------.
# TODO: Add to pyresample: SLine.extend(distance) 
import pyproj
from pyproj import Geod
g = Geod(ellps='WGS84')
fwd_az, back_az, dist = g.inv(*start_lonlat, *end_lonlat, radians=False)
lon_r, lat_r, _ = g.fwd(*start_lonlat, az=fwd_az, dist=dist+50000) # dist in m 
fwd_az, back_az, dist = g.inv(*end_lonlat, *start_lonlat, radians=False)
lon_l, lat_l, _ = g.fwd(*end_lonlat, az=fwd_az, dist=dist+50000) # dist in m

##----------------------------------------------------------------------------.
#### Scan Filtering  
# ScanStatus/dataQuality 
# ScanStatus/geoError 
# ScanStatus/modeStatus
# ScanStatus/dataWarning
# ScanStatus/operationalMode
# DataQualityFiltering = {'TotalQualityCode' : ['Good'],  # ”Fair” or ”EG”

#### Check non-missing scan 
# - Footprint distance ... 

##----------------------------------------------------------------------------.
#### Masking functions
# - Masking when NA on other variable  
# - Drop scan with FLG/qualityFlag low or bad


##----------------------------------------------------------------------------.
#### Patch 
# - get patch from gpm_geo and yasser code !!! 
# - get_patch_around_max 
# - get_patch_from_center(lon, lat) 

#### Iterate through patches and plot 

##----------------------------------------------------------------------------.
#### GridBucket - Collect overpass values onto grid cell --> Parquet optmized 
# --> pyresample bucket resampler 
# --> https://pyresample.readthedocs.io/en/latest/api/pyresample.bucket.html

##----------------------------------------------------------------------------.
#### Plotting 
# - Retrieve outer coordinates instead of centroid 
# - Enlarge swath line to account for pixel centroid  
# - plot_swath_lines 
# - plot_swath_polygon

# - Add log ticks to colorbar 
# - Add world map inset 
#    --> https://github.com/dopplerchase/DRpy/blob/master/drpy/graph/graph.py#L147 
#    --> https://stackoverflow.com/questions/45527584/how-to-easily-add-a-sub-axes-with-proper-position-and-size-in-matplotlib-and-car

# - Reimplement plotter drpy 
# - Randy DRPy plots 

# - DPR vs GMI 

# - Plot with pcolormesh when crossing the antimeridan

# - Add temperate, pressure contours ....  
# https://github.com/dopplerchase/DRpy/blob/7d4246d977e926d02b19059de0a0c3793711e2f1/drpy/graph/graph.py#L638

#----------------------------------------------------------------------------.
#### Download 
# Solve curl/wget issue 

# Document: 
# allows access to all ports in the range of 64000-65000 for the DNS names of: 
#  ‘arthurhou.pps.eosdis.nasa.gov’ and ‘arthurhouftps.pps.eosdis.nasa.gov’.
# open/allow access to all ports in the range of 64000-65000 for the system
# ‘arthurhouftps.pps.eosdis.nasa.gov’  (ftps)
# ‘arthurhou.pps.eosdis.nasa.gov’ (ftps)
# Python 3 ftplib 
# curl ftps
# Explicit FTPS

#----------------------------------------------------------------------------.
#### xrimage
## xr.upsample function (image) pixel ... duplicate or interpolate 
## xr.downsample function  

#----------------------------------------------------------------------------.
#### Subset by country 
#### Subset by lat/lon coord (nearest neighbor over x km)
# gpm.DPR.plot(timestep, bbox, product)
# gpm.ZoomMax(bbox, product, n_scan).plot()
# gpm.ZoomMin(bbox.product, n_scan).plot()
# gpm.ZoomRandom(min_val, max_val, bbox, product, n_scan).plot()
# gpm.isel(), sel() 

#----------------------------------------------------------------------------.
#### Compute height 

# Add range distance (based on length):
# --> Ku: 250 meters (m) 
# --> Ka: 250/500 meters (m) 

# Add 3D altitude and 3D lat/lon array 
# --> Watch https://docs.wradlib.org/en/stable/notebooks/match3d/wradlib_match_workflow.html 
# --> DPR ATBD explain how 


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

##----------------------------------------------------------------------------.
#### Trasect Profile
# - Extract oblique transect (i.e. with longest amount of )
# - get_max_latlon_coordinates() ... and pass lat lon to get_transect

##----------------------------------------------------------------------------.
#### Test download NRT data
 