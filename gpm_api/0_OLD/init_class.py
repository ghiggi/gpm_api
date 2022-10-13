#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:55:47 2020

@author: ghiggi
"""
from .DPR.DPR import create_DPR
from .DPR.DPR_ENV import create_DPR_ENV
from .PMW.GMI import create_GMI
from .IMERG.IMERG import create_IMERG

def create_GPM_class(base_DIR, product, bbox=None, start_time=None, end_time=None):
    # TODO add for ENV and SLH
    if (product in ['1B-Ka','1B-Ku','2A-Ku','2A-Ka','2A-DPR','2A-SLH']):
        x = create_DPR(base_DIR=base_DIR, product=product,
                       bbox=bbox, start_time=start_time, end_time=end_time)
    elif (product in ['IMERG-FR','IMERG-ER','IMERG-LR']):
        x = create_IMERG(base_DIR=base_DIR, product=product,
                         bbox=bbox, start_time=start_time, end_time=end_time)
    elif (product in ['TODO list GMI products']):
        x = create_GMI(base_DIR=base_DIR, product=product,
                       bbox=bbox, start_time=start_time, end_time=end_time)
    elif (product in GPM_DPR_2A_ENV_RS_products()):
        x = create_DPR_ENV(base_DIR=base_DIR, product=product,
                           bbox=bbox, start_time=start_time, end_time=end_time)
    else:
        raise ValueError("Class method for such product not yet implemented")
    return(x)