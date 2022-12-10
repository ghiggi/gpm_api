#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:00:39 2022

@author: ghiggi
"""
import time
import dask 
import gpm_api
import datetime
import numpy as np
from gpm_api.io.pps import _find_pps_daily_filepaths, find_pps_filepaths
from gpm_api.io.info import get_start_time_from_filepaths, get_end_time_from_filepaths
 

 
username="gionata.ghiggi@epfl.ch"
version=7
verbose=False
start_year = 1997
end_year=datetime.datetime.utcnow().year
step_months = 6
products = gpm_api.available_products("RS")
product = "2A-DPR"

#-----------------------------------------------------------------------------.
#### List product temporal availability 
# TODO: find_pps_filepaths ...retry when PPS currently unavailable
from gpm_api.utils.archive import get_product_temporal_coverage

t_i = time.time() 
info_dict = get_product_temporal_coverage(product=product,  
                                          username=username,  
                                          version=version, 
                                          start_year=start_year, 
                                          end_year=end_year, 
                                          step_months=step_months,
                                          verbose=verbose)
t_f = time.time()
t_elapsed = np.round(t_f - t_i, 2)
print(t_elapsed, "seconds") # 40 seconds per month 


### Save list of files, to test reader ....


# Source	RP Start	RP End
# AQUA_AMSRE	2002-06-01	2011-10-04
# F08_SSMI	1987-07-09	1991-12-31
# F10_SSMI	1990-12-08	1997-11-14
# F11_SSMI	1991-12-03	2000-05-16
# F13_SSMI	1995-05-03	2009-11-20
# F14_SSMI	1997-05-07	2008-08-24
# F15_SSMI	2000-02-23	2006-08-14
# F16_SSMIS	2005-11-20	2022-04-30
# F17_SSMIS	2008-03-19	2022-04-30
# F18_SSMIS	2010-03-08	2022-04-30
# F19_SSMIS	2014-12-18	2016-02-11
# GCOMW1_AMSR2	2012-07-02	2022-04-30
# GPM_GMI	2014-03-04	2022-04-30
# METOPA_MHS	2006-11-23	2019-12-23
# METOPB_MHS	2012-09-25	2022-04-30
# METOPC_MHS	2018-11-16	2022-04-30
# NOAA15_AMSUB	2000-01-01	2010-09-15
# NOAA16_AMSUB	2000-10-04	2010-05-01
# NOAA17_AMSUB	2002-06-28	2009-12-18
# NOAA18_MHS	2005-05-25	2018-10-20
# NOAA19_MHS	2009-02-12	2022-04-30
# NOAA20_ATMS	2017-11-29	2022-04-30
# NPP_ATMS	2011-11-08	2022-04-30
# GPM_DPRGMI	2014-03-08	2022-04-30
# TRMM_TMI	1997-12-07	2015-04-08
# MT1_SAPHIR	2011-10-13	2022-01-01
# TRMM_VIRS	1997-12-20	2014-03-21
# TRMM_PR	1997-12-07	2015-04-01
# TRMM_PRTMI	1997-12-07	2015-04-01
 