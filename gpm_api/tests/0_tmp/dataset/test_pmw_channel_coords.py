#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:30:32 2023

@author: ghiggi
"""
import re
import datetime

import gpm_api


#### Define analysis time period
start_time = datetime.datetime.strptime("2020-08-01 12:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-08-10 12:00:00", "%Y-%m-%d %H:%M:%S")

product_type = "RS"
variable = "Tc"

# Loop over 1C products
product = "1C-GMI"

### Retrieve channels from 1C-products
filepath = "/home/ghiggi/data/GPM/RS/V07/PMW/1C-MHS-METOPB/2018/07/01/1C.METOPB.MHS.XCAL2016-V.20180701-S063009-E081129.030013.V07A.HDF5"
filepath = "/home/ghiggi/data/GPM/RS/V07/PMW/1C-ATMS-NPP/2018/07/01/1C.NPP.ATMS.XCAL2019-V.20180701-S075948-E094117.034588.V07A.HDF5"
filepath = "/home/ghiggi/data/GPM/RS/V07/PMW/1C-AMSUB-NOAA17/2004/07/01/1C.NOAA17.AMSUB.XCAL2017-V.20040701-S081606-E095717.010492.V07A.HDF5"
for i in range(1, 6):
    scan_mode = "S" + str(i)
    ds = gpm_api.open_granule(filepath, scan_mode=scan_mode)
    print(scan_mode)
    print(ds["pmw_frequency"].shape)
    print(ds["Tc"].attrs["LongName"])

#### We should define the channels coordinates based on the content in LongName
string = ds["Tc"].attrs["LongName"]
list_element = re.sub(r"(\d)\)", r"\n ", string).replace("and", "").replace(",", "").split("\n")
list_element = [s.strip() for s in list_element]
list_element = [s for s in list_element if len(s) > 1]
list_element = list_element[1:]
list_element

# Satellites:
#   - AMSUB: Advanced Microwave Sounding Unit-B
#   - AMSR2: Advanced Microwave Scanning Radiometer 2
#   - AMSRE: Advanced Microwave Scanning Radiometer - EOS
#   - ATMS: Advanced Technology Microwave Sounder
#   - GMI: GPM Microwave Imager
#   - MSH: Microwave Sounder for Hydrology and Climate
#   - SAPHIR: Sounder for Atmospheric Profiling of Humidity in the Inter-Tropical Regions
#   - SSMI: Special Sensor Microwave Imager
#   - SSMIS: Special Sensor Microwave Imager/Sounder
#   - TMI: TRMM Microwave Imager


# GMI S1
# 1) 10.65 GHz V-Pol
# 2) 10.65 GHz H-Pol
# 3) 18.7 GHz V-Pol
# 4) 18.7 GHz H-Pol
# 5) 23.8 GHz V-Pol
# 6) 36.64 GHz V-Pol
# 7) 36.64 GHz H-Pol
# 8) 89.0 GHz V-Pol
# 9) 89.0 GHz H-Pol

# GMI S2
# 1) 166.0 GHz V-Pol
# 2) 166.0 GHz H-Pol
# 3) 183.31 +/-3 GHz V-Pol
# 4) 183.31 +/-7 GHz V-Pol


# TMI
# S1: 10V 10H 19V 19H 23V 37V 37H 89V 89H

# GMI GMI
# S1: 10V 10H 19V 19H 23V 37V 37H 89V 89H
# S2: 165V 165H 183+/-3V 183+/-7V

# SSMI
# S1: 19V 19H 22V 37V 37H
# S2: 85V 85H

# SSMIS
# S1: 19V 19H 22V
# S2: 37V 37H
# S3: 150H 183+/-1H 183+/-3H 183+/-7H
# S4: 91V 91H

# AMSRE
# S1: 10.65V 10.65H
# S2: 18.7V 18.7H
# S3: 23.8V 23.8H
# S4: 36.5V 36.5H
# S5: 89V 89H
# S6: 89V 89H

# AMSR2
# S1: 10.65V 10.65H
# S2: 18.7V 18.7H
# S3: 23.8V 23.8H
# S4: 36.5V 36.5H
# S5: 89V 89H
# S6: 89V 89H

# MSH
# 89.0GHzV, 157.0GHzV, 183.31GHz+/-1GHzH, 183.31GHz+/-3GHzH, and 190.31GHzV

# SAPHIR
# S1: 183.31H0.2, 183.31H1.1, 183.31H2.8, 183.31H4.2, 183.31H6.8, 183.31H11.0


# AMSU-B (166V 166H 183+/-3V 183+/-8V)
# S1: 89.0 +/- 0.9 GHz, 150.0 +/- 0.9 GHz, 183.31 +/- 1 GHz, 183.31 +/- 3 183.31 +/- 7 GHz

# ATMS

# S1 23.8QV
# S2 31.4QV
# S3 88.2QV
# S4 183.31QH7 , 183.31QH4.5 183.31QH, 183.31QH1.8, 23 183.31QH1
