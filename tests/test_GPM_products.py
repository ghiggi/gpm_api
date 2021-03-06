#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:35:58 2020

@author: ghiggi
"""
import os
os.chdir('/home/ghiggi/gpm_api') # change to the 'scripts_GPM.py' directory

from gpm_api.io import GPM_DPR_1B_RS_products
from gpm_api.io import GPM_DPR_1B_NRT_products
from gpm_api.io import GPM_DPR_2A_RS_products
from gpm_api.io import GPM_DPR_2A_NRT_products
from gpm_api.io import GPM_DPR_2A_ENV_RS_products
from gpm_api.io import GPM_DPR_2A_ENV_NRT_products
from gpm_api.io import GPM_DPR_RS_products
from gpm_api.io import GPM_DPR_NRT_products

from gpm_api.io import GPM_PMW_1B_RS_products
from gpm_api.io import GPM_PMW_1B_NRT_products
from gpm_api.io import GPM_PMW_1C_RS_products
from gpm_api.io import GPM_PMW_1C_NRT_products
from gpm_api.io import GPM_PMW_2A_GPROF_RS_products
from gpm_api.io import GPM_PMW_2A_GPROF_NRT_products
from gpm_api.io import GPM_PMW_2A_PRPS_RS_products
from gpm_api.io import GPM_PMW_2A_PRPS_NRT_products
from gpm_api.io import GPM_PMW_RS_products
from gpm_api.io import GPM_PMW_NRT_products

from gpm_api.io import GPM_IMERG_NRT_products
from gpm_api.io import GPM_IMERG_RS_products
from gpm_api.io import GPM_IMERG_products

from gpm_api.io import GPM_1B_RS_products
from gpm_api.io import GPM_1B_NRT_products
from gpm_api.io import GPM_1C_RS_products 
from gpm_api.io import GPM_1C_NRT_products
from gpm_api.io import GPM_2A_RS_products
from gpm_api.io import GPM_2A_NRT_products

from gpm_api.io import GPM_RS_products
from gpm_api.io import GPM_NRT_products
from gpm_api.io import GPM_products

GPM_DPR_1B_RS_products()
GPM_DPR_1B_NRT_products()
GPM_DPR_2A_RS_products()
GPM_DPR_2A_NRT_products()
GPM_DPR_2A_ENV_RS_products()
GPM_DPR_2A_ENV_NRT_products()
GPM_DPR_RS_products()
GPM_DPR_NRT_products()
GPM_PMW_1B_RS_products()
GPM_PMW_1B_NRT_products()
GPM_PMW_1C_RS_products()
GPM_PMW_1C_NRT_products()
GPM_PMW_2A_GPROF_RS_products()
GPM_PMW_2A_GPROF_NRT_products()
GPM_PMW_2A_PRPS_RS_products()
GPM_PMW_2A_PRPS_NRT_products()
GPM_PMW_RS_products()
GPM_PMW_NRT_products()

GPM_IMERG_NRT_products()
GPM_IMERG_RS_products()
GPM_IMERG_products()
 
GPM_1B_RS_products()
GPM_1B_NRT_products()
GPM_1C_RS_products()
GPM_1C_NRT_products()
GPM_2A_RS_products()
GPM_2A_NRT_products()

GPM_RS_products()
GPM_NRT_products()
GPM_products()
GPM_products("NRT")
GPM_products("RS")
