#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 18:46:16 2023

@author: ghiggi
"""

import datatree

from gpm_api.dataset.attrs import decode_attrs, get_granule_attrs
from gpm_api.dataset.coords import get_coords
from gpm_api.dataset.datatree import open_datatree
from gpm_api.dataset.granule import (
    _get_scan_mode_info,
    _get_scan_mode_dataset,
)
from gpm_api.dataset.dimensions import (
    _rename_datatree_dimensions,
    _get_datatree_dim_dict,
)
from gpm_api.dataset.groups_variables import (
    _get_group_variables_dict,
    _get_variables_path_dict,
    _get_variables_group_dict,
    _get_available_groups,
    _get_available_variables,
    _get_available_scan_modes,
)


# Not readable as gpm_api dataset because of file design
filepath = "/home/ghiggi/data/GPM/RS/V04/RADAR/2B-GPM-CSAT/2016/03/07/2B.CSATGPM.COIN.55N_044W_00000_000_272_673.20160307-S052818-E053057.011491.V04.nc4"


# OK
filepath = "/home/ghiggi/data/GPM/RS/V05/PMW/1B-TMI/2014/07/01/1B.TRMM.TMI.Tb2017.20140701-S045751-E063013.094690.V05A.HDF5"
filepath = "/home/ghiggi/data/GPM/RS/V07/PMW/1B-TMI/2014/07/01/1B.TRMM.TMI.Tb2021.20140701-S063014-E080236.094691.V07A.HDF5"
filepath = "/home/ghiggi/data/GPM/RS/V07/PMW/1C-ATMS-NPP/2018/07/01/1C.NPP.ATMS.XCAL2019-V.20180701-S075948-E094117.034588.V07A.HDF5"

filepath = "/home/ghiggi/data/GPM/RS/V07/RADAR/2A-TRMM-SLH/2014/07/01/2A.TRMM.PR.TRMM-SLH.20140701-S080237-E093500.094692.V07A.HDF5"
filepath = "/home/ghiggi/data/GPM/RS/V07/RADAR/2A-ENV-PR/2014/07/01/2A-ENV.TRMM.PR.V9-20220125.20140701-S063014-E080236.094691.V07A.HDF5"
filepath = "/home/ghiggi/data/GPM/RS/V07/RADAR/1B-PR/2014/07/01/1B.TRMM.PR.V9-20210630.20140701-S080237-E093500.094692.V07A.HDF5"
filepath = "/home/ghiggi/data/GPM/RS/V07/RADAR/1B-Ku/2020/10/28/GPMCOR_KUR_2010280754_0927_037875_1BS_DUB_07A.h5"
filepath = "/home/ghiggi/data/GPM/RS/V07/RADAR/2A-DPR/2022/07/06/2A.GPM.DPR.V9-20211125.20220706-S043937-E061210.047456.V07A.HDF5"

filepath = "/home/ghiggi/data/GPM/RS/V07/CMB/2B-GPM-CORRA/2016/03/09/2B.GPM.DPRGMI.CORRA2022.20160309-S091322-E104552.011525.V07A.HDF5"

filepath = "/home/ghiggi/data/GPM/RS/V06/IMERG/IMERG-FR/2020/02/01/3B-HHR.MS.MRG.3IMERG.20200201-S180000-E182959.1080.V06B.HDF5"
index = 0

# OK
filepath = "/home/ghiggi/data/GPM/RS/V05/PMW/1A-GMI/2015/08/01/1A.GPM.GMI.COUNT2016.20150801-S131408-E144641.008089.V05A.HDF5"
filepath = "/home/ghiggi/data/GPM/RS/V07/PMW/2A-GMI/2020/08/01/2A.GPM.GMI.GPROF2021v1.20200801-S105247-E122522.036508.V07A.HDF5"
filepath = "/home/ghiggi/data/GPM/RS/V07/PMW/2A-MHS-METOPC/2020/08/02/2A.METOPC.MHS.GPROF2017v2.20200802-S023515-E041635.009008.V05D.HDF5"
index = 1

dt = datatree.open_datatree(
    filepath, engine="netcdf4", chunks={}, decode_cf=False
)  # --> file chunks when Martin PR merged
scan_modes = _get_available_scan_modes(dt)
scan_mode = scan_modes[index]

dt = open_datatree(filepath, use_api_defaults=False)
dt

dt = open_datatree(filepath, use_api_defaults=True)
dt

ds = _get_scan_mode_dataset(
    dt, scan_mode, variables=None, groups=None, prefix_group=False, chunks={}, decode_cf=False
)


### Test _get_granule_dataset functions

src_dt = datatree.open_datatree(
    filepath, engine="netcdf4", chunks={}, decode_cf=False
)  # --> file chunks when Martin PR merged


src_dt.groups
groups = list(src_dt.groups)


prefix_group = False
variables = None
chunks = {}
decode_cf = False

dim_dict = _get_datatree_dim_dict(src_dt)
dt = _rename_datatree_dimensions(src_dt, use_api_defaults=False)
dt = _rename_datatree_dimensions(src_dt, use_api_defaults=True)

scan_modes = _get_available_scan_modes(dt)
scan_mode = scan_modes[0]
groups = _get_available_groups(dt, scan_mode, name=True)
variables = _get_available_variables(dt, scan_mode)
var_path_dict = _get_variables_path_dict(dt, scan_mode)
var_group_dict = _get_variables_group_dict(dt, scan_mode)
group_vars_dict = _get_group_variables_dict(dt, scan_mode)


ds = _get_scan_mode_dataset(dt, scan_mode, groups, variables=None, prefix_group=False)

dt.attrs
decode_attrs(dt.attrs)
get_granule_attrs(dt)  # subsetted by gpm_api defaults configs
get_coords(dt, scan_mode)

coords, attrs, groups, variables = _get_scan_mode_info(
    dt, scan_mode, variables=variables[0:2], groups=None
)
coords, attrs, groups, variables = _get_scan_mode_info(
    dt, scan_mode, variables=None, groups=groups[0:2]
)
coords, attrs, groups, variables = _get_scan_mode_info(dt, scan_mode, variables=None, groups=None)


ds = _get_scan_mode_dataset(
    dt, scan_mode, variables=None, groups=None, prefix_group=False, chunks={}, decode_cf=False
)
