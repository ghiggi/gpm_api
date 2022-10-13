#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 19:05:10 2020

@author: ghiggi
"""

class create_DPR_ENV():
    "Define methods for 2A-ENV data."
    def __init__(self, base_DIR, product, bbox=None, start_time=None, end_time=None):
        self.name = product
        self.base_DIR = base_DIR
        self.start_time = start_time 
        self.end_time = end_time 
        self.bbox = bbox
        self.NS = None
        self.MS = None
        self.HS = None
    ### Here below the plotting functions 

    ### Here below the post-processing functions     
 