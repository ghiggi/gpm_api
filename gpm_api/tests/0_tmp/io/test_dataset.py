#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 13:52:46 2022

@author: ghiggi
"""
import datetime
import gpm_api

#### Define analysis time period
start_time = datetime.datetime.strptime("2020-10-28 08:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-10-28 09:00:00", "%Y-%m-%d %H:%M:%S")

products = [
    "2A-DPR",
    "2A-Ka",
    "2A-ENV-DPR",
    "1B-Ka",
    "1B-Ku",
    "1C-GMI",
    "2A-GMI",
    "2B-GPM-CORRA",
    "2B-GPM-CSH",
    "2A-GPM-SLH",
]

start_time = datetime.datetime.strptime("2014-07-01 08:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2014-07-01 09:00:00", "%Y-%m-%d %H:%M:%S")

products = [
    "2A-DPR",
    "2A-Ka",
    "2A-ENV-DPR",
    "1B-Ka",
    "1B-Ku",
    "1C-GMI",
    "2A-GMI",
    "2B-GPM-CORRA",
    "2B-GPM-CSH",
    "2A-GPM-SLH",
]
products = ["2B-TRMM-CORRA", "2B-TRMM-CSH", "2A-TRMM-SLH", "1B-TMI", "1C-TMI", "2A-ENV-PR", "2A-PR"]

products = gpm_api.available_products(product_category="CMB")
products = gpm_api.available_products(product_category="IMERG")
products = gpm_api.available_products(product_category="RADAR")
products = gpm_api.available_products(product_category="PMW")


start_time = datetime.datetime.strptime("2018-07-01 08:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2018-07-01 09:00:00", "%Y-%m-%d %H:%M:%S")
product_type = "RS"

#### Download products
for product in products:
    print(product)
    gpm_api.download(
        product=product,
        product_type=product_type,
        start_time=start_time,
        end_time=end_time,
        force_download=False,
        transfer_tool="curl",
        progress_bar=True,
        verbose=True,
        n_threads=1,
    )


# -----------------------------------------------------------------------------.
# Test open_granule
filepaths = [
    "/home/ghiggi/GPM/RS/V07/PMW/1B-TMI/2014/07/01/1B.TRMM.TMI.Tb2021.20140701-S063014-E080236.094691.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/PMW/1C-SSMI-F16/2014/07/01/1C.F16.SSMIS.XCAL2021-V.20140701-S074007-E092202.055213.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/PMW/1C-TMI/2014/07/01/1C.TRMM.TMI.XCAL2021-V.20140701-S063014-E080236.094691.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/PMW/1C-GMI/2020/10/28/1C.GPM.GMI.XCAL2016-C.20201028-S075448-E092720.037875.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/PMW/2A-GMI/2020/10/28/2A.GPM.GMI.GPROF2021v1.20201028-S075448-E092720.037875.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/CMB/2B-TRMM-CORRA/2014/07/01/2B.TRMM.PRTMI.CORRA2022T.20140701-S063014-E080236.094691.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/CMB/2B-GPM-CORRA/2020/10/28/2B.GPM.DPRGMI.CORRA2022.20201028-S075448-E092720.037875.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/CMB/2B-TRMM-CSH/2014/07/01/2B.TRMM.PRTMI.2HCSHv7-0.20140701-S063014-E080236.094691.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/CMB/2B-GPM-CSH/2020/10/28/2B.GPM.DPRGMI.2HCSHv7-0.20201028-S075448-E092720.037875.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/RADAR/2A-TRMM-SLH/2014/07/01/2A.TRMM.PR.TRMM-SLH.20140701-S063014-E080236.094691.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/RADAR/2A-GPM-SLH/2020/10/28/2A.GPM.DPR.GPM-SLH.20201028-S075448-E092720.037875.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/RADAR/1B-PR/2014/07/01/1B.TRMM.PR.V9-20210630.20140701-S063014-E080236.094691.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/RADAR/2A-PR/2014/07/01/2A.TRMM.PR.V9-20220125.20140701-S063014-E080236.094691.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/RADAR/2A-ENV-PR/2014/07/01/2A-ENV.TRMM.PR.V9-20220125.20140701-S063014-E080236.094691.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/RADAR/1B-Ka/2020/10/28/GPMCOR_KAR_2010280754_0927_037875_1BS_DAB_07A.h5",
    "/home/ghiggi/GPM/RS/V07/RADAR/1B-Ku/2020/10/28/GPMCOR_KUR_2010280754_0927_037875_1BS_DUB_07A.h5",
    "/home/ghiggi/GPM/RS/V07/RADAR/2A-Ka/2020/10/28/2A.GPM.Ka.V9-20211125.20201028-S075448-E092720.037875.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/RADAR/2A-DPR/2020/10/28/2A.GPM.DPR.V9-20211125.20201028-S075448-E092720.037875.V07A.HDF5",
    "/home/ghiggi/GPM/RS/V07/RADAR/2A-ENV-DPR/2020/10/28/2A-ENV.GPM.DPR.V9-20211125.20201028-S075448-E092720.037875.V07A.HDF5",
    # "/home/ghiggi/GPM/RS/V06/IMERG/IMERG-FR/2020/10/28/3B-HHR.MS.MRG.3IMERG.20201028-S080000-E082959.0480.V06B.HDF5",
]

filepath = filepaths[0]

groups = None
variables = None
scan_mode = None
decode_cf = False
chunks = "auto"
prefix_group = False
version = 7
for filepath in filepaths:
    print(filepath)
    ds = gpm_api.open_granule(
        filepath,
        scan_mode=scan_mode,
        groups=groups,
        variables=variables,
        decode_cf=False,
        chunks={},
        prefix_group=True,
    )
    print(ds)


# -----------------------------------------------------------------------------.
# Test open_dataset

#### Define analysis time period
start_time = datetime.datetime.strptime("2020-10-28 08:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-10-28 09:00:00", "%Y-%m-%d %H:%M:%S")

products = [
    "2A-DPR",
    "2A-Ka",
    "2A-ENV-DPR",
    "1B-Ka",
    "1B-Ku",
    "1C-GMI",
    "2A-GMI",
    "2B-GPM-CORRA",
    "2B-GPM-CSH",
    "2A-GPM-SLH",
]

start_time = datetime.datetime.strptime("2014-07-01 08:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2014-07-01 09:00:00", "%Y-%m-%d %H:%M:%S")

products = ["2B-TRMM-CORRA", "2B-TRMM-CSH", "2A-TRMM-SLH", "1B-TMI", "1C-TMI", "2A-ENV-PR", "2A-PR"]

products = ["1C-ATMS-NPP", "1C-GMI", "1C-MHS-METOPA", "1C-SSMI-F16"]


product = products[0]
groups = None
variables = None
scan_mode = None
decode_cf = False
chunks = "auto"
prefix_group = False
version = None

for product in products:
    print(product)
    ds = gpm_api.open_dataset(
        product=product,
        start_time=start_time,
        end_time=end_time,
        # Optional
        variables=variables,
        groups=groups,  # TODO implement
        scan_mode=scan_mode,
        version=version,
        product_type="RS",
        chunks={},
        decode_cf=False,
        prefix_group=True,
    )
    print(ds)

# -----------------------------------------------------------------------------.
