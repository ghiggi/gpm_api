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
list_regular_slices = ds_gpm.gpm_api.get_slices_contiguous_scan()
print(list_regular_slices)
slc = list_regular_slices[0]

ds_1 = ds_gpm.isel(along_track=list_regular_slices[0])
ds_2 = ds_gpm.isel(along_track=list_regular_slices[1])

#### DPR scanStatus
# - scanStatus
#   - geoError
#   - missing
#   - modeStatus
#   - geoWarning

# dataQuality
# - A summary of data quality in the scan. 
# - Unless this is 0 (normal), the scan data is meaningless to higher precipitation processing. 
# - 0 --> OK 
# - 1 = 2**0 --> missing 
# - 32 = 2**5 --> geoError not zero 
# - 64 = 2**6 --> modeStatus not zero 
np.unique(ds_gpm["scanStatus/dataQuality"].isel(frequency=0).data.compute(), return_counts=True)  # 0, 1, 64, 65

# dataWarning
# - 0 --> OK 
# - 1 = 2**0 --> beam Matching is abnormal 
# - 2 = 2**1 --> VPRF table is abnormal
# - 4 = 2**2 --> surface Table is abnormal 
# - 8 = 2**3 --> geoWarning is not Zero
# - 16 = 2**4 --> operational mode is not observation mode
# - 32 = 2**5 --> GPS status is abnormal 
# - 64 = 2**6 --> modeStatus not zero 
np.unique(ds_gpm["scanStatus/dataWarning"].isel(frequency=0).data.compute(), return_counts=True)  # 0, 8

# missing
# - Indicates whether information is contained in the scan data.
# - 0 --> OK 
# - 1 = 2**0 --> Scan is missing
# - 2 = 2**1 --> Science telemetry packet missing
# - 4 = 2**2 -->  Science telemetry segment withing packet missing
# - 8 = 2**3 --> Science telemetry other missing
# - 16 = 2**4 -->  Housekeeping (HK) telemetry packet missing
# - 32 = 2**5 --> : Spare (always 0) 
# - 64 = 2**6 --> : Spare (always 0)
# - 128 = 2**7 --> : Spare (always 0)
np.unique(ds_gpm["scanStatus/missing"].isel(frequency=0).data.compute(), return_counts=True)  # 0, 3, 9

# modeStatus
# - A summary of status modes
# - 0 --> OK 
# - 1 = 2**0 --> Spare (always 0)
# - 2 = 2**1 --> SCorientation not 0 or 180
# - 4 = 2**2 --> pointingStatus not 0
# - 8 = 2**3 -->  Non-routine limitErrorFlag
# - 16 = 2**4 --> Non-routine operationalMode (not 1 or 11)
# - 32 = 2**5 --> Spare (always 0)
# - 64 = 2**6 --> Spare (always 0)
# - 128 = 2**7 --> Spare (always 0)
np.unique(ds_gpm["scanStatus/modeStatus"].isel(frequency=0).data.compute(), return_counts=True)  # 0, 16, 22

# geoError
# - A summary of status modes
# - 0 --> OK 
# - 1 = 2**0 --> Latitude limit exceeded for viewed pixel locations
# - 2 = 2**1 --> Negative scan time, invalid input
# - 4 = 2**2 --> Error getting spacecraft attitude at scan mid-time
# - 8 = 2**3 -->  Error getting spacecraft ephemeris at scan mid-time
# - 16 = 2**4 --> Invalid input non-unit ray vector for any pixel
# - 32 = 2**5 --> Ray misses Earth for any pixel with normal pointing
# - 64 = 2**6 --> Nadir calculation error for subsatellite position
# - 128 = 2**7 --> Pixel count with geolocation error over threshold
# - 256 = 2**8 --> Error in getting spacecraft attitude for any pixel
# - 512 = 2**9 --> Error in getting spacecraft ephemeris for any pixel
np.unique(ds_gpm["scanStatus/geoError"].isel(frequency=0).data.compute(), return_counts=True)  # 0


np.unique(ds_gpm["scanStatus/operationalMode"].isel(frequency=0).data.compute(), return_counts=True)  # 1, 3, 4, 5  -99
# geoWarning
# - A summary of geolocation warnings in the scan
# - geoWarning does not set a bit in dataQuality
# - Warnings indicate unusual conditions 
# --> 10 bits
flag_dict = {
    0: "Ephemeris Gap Interpolated",
    1: "Attitude Gap Interpolated",
    2: "Attitude jump/discontinuity",
    3: "Attitude out of range",
    4: "Anomalous Time Step",
    5: "GHA not calculated due to error",
    6: "SunData (Group) not calculated due to error",
    7: "Failure to calculate Sun in inertial coordinates",
    8: "Fallback to GES ephemeris",
    9: "Fallback to GEONS ephemeris",
    10: "Fallback to PVT ephemeris",
    11: "Fallback to OBP ephemeris",
    12: "Spare (always 0)",
    13: "Spare (always 0)",
    14: "Spare (always 0)",
    15: "Spare (always 0)"
}
np.unique(ds_gpm["scanStatus/geoWarning"].isel(frequency=0).data.compute(), return_counts=True) # 0, 8

#### SCorientation
# - The positive angle of the spacecraft vector (v) from the satellite forward direction of motion,
#   measured clockwise facing down. 
# - We define v in the same direction as the spacecraft axis +X, which
#   is also the center of the GMI scan.
np.unique(ds_gpm["scanStatus/SCorientation"].data.compute()) # 0, 180, -8000

flag_values=[0, 180, -8000, -9999]
flag_dict = {
    0: "+X forward (yaw 0)",     # scanning forward
    180: "-X forward (yaw 180)", # scanning backward
    -8000: "Non-nominal pointing",
    -9999: "Missing"
}


def get_bits_values(n):
    bits = [int(bit) for bit in bin(n)[2:]]
    bits_values = [i for i, bit in enumerate(bits[::-1]) if bit == 1]
    return bits_values

print(get_bits_values(10)) # [1, 0, 1]



### scanStatus       
# dataQuality is 0     
# -> modeStatus is 0
# -> geoError is 0 
# -> missing is 0 

# ScOrientation contiguous values (0 or 180)

# ScOrientation 
# - also -90 for TMI
# - 0 to 360 for PMW 1C 

# Yaw turns are performed approximately every 40 days for thermal control, 
#   as the angle between the spacecraft's orbit and the sun changes. 
#   This keeps the side of the spacecraft designed to remain cold from overheating.



# dataQuality
# - A summary of data quality in the scan. 
# - Unless this is 0 (normal), the scan data is meaningless to higher precipitation processing. 
# - 0 --> OK 
# - 1 = 2**0 --> missing 
# - 32 = 2**5 --> geoError not zero 
# - 64 = 2**6 --> modeStatus not zero
# ... 
# -->  (available in 1A, 1B, 1C, 2A-DPR, 2A-PR, 2B-CMB)
 

# get slices of contiguous granules 
#   get slices of contiguous scans

