#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 18:30:32 2023

@author: ghiggi
"""
import datetime

import gpm_api

base_dir = "/home/ghiggi"

#### Define analysis time period
start_time = datetime.datetime.strptime("2020-08-01 12:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-08-10 12:00:00", "%Y-%m-%d %H:%M:%S")

product_type = "RS"


variable = "Tc"

# Loop over 1C products
product = "1C-GMI"

### Retrieve channels from 1C-products
filepath = "/home/ghiggi/GPM/RS/V07/PMW/1C-MHS-METOPB/2018/07/01/1C.METOPB.MHS.XCAL2016-V.20180701-S063009-E081129.030013.V07A.HDF5"
ds = gpm_api.open_granule(filepath)
print(ds["Tc"].attrs["LongName"])


import re

string = ds["Tc"].attrs["LongName"]
list_element = re.sub(r"(\d)\)", r"\n ", string).replace("and", "").replace(",", "").split("\n")
list_element = [s.strip() for s in list_element]
list_element = [s for s in list_element if len(s) > 1]
list_element = list_element[1:]


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

# TRMM TMI (10V 10H 19V 19H 23V 37V 37H 89V 89H)
# AMSU-B (166V 166H 183+/-3V 183+/-8V)
