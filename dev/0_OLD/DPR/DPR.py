#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 19:05:10 2020

@author: ghiggi
"""


class create_DPR:
    "Define methods for DPR data."

    def __init__(self, base_DIR, product, bbox=None, start_time=None, end_time=None):
        self.name = product
        self.product = product
        self.base_DIR = base_DIR
        self.start_time = start_time
        self.end_time = end_time
        self.bbox = bbox
        self.NS = None
        self.MS = None
        self.HS = None
        self.Swath = None

    ### Here below the plotting functions
    # TODO
    ### Here below the post-processing functions
    def retrieve_ENV(self):
        from ..dataset import read_GPM

        if self.product == "2A-Ka":
            ENV = read_GPM(
                base_DIR=self.base_DIR,
                product="2A-ENV-Ka",
                start_time=self.start_time,
                end_time=self.end_time,
            )

            # Check for coordinates, where not same
            # - Discard missing scan , add NaN to missing scan
            self.MS = self.MS.combine_first(ENV.HS)  # xr.merge([ENV.NS,DPR.NS])
            self.HS = self.HS.combine_first(ENV.MS)
        elif self.product == "2A-Ku":
            ENV = read_GPM(
                base_DIR=self.base_DIR,
                product="2A-ENV-Ku",
                start_time=self.start_time,
                end_time=self.end_time,
            )
            # Check for coordinates, where not same
            # - Discard missing scan , add NaN to missing scan
            self.NS = self.NS.combine_first(ENV.NS)  # xr.merge([ENV.NS,DPR.NS])
        elif self.product == "2A-DPR":
            ENV = read_GPM(
                base_DIR=self.base_DIR,
                product="2A-ENV-DPR",
                start_time=self.start_time,
                end_time=self.end_time,
            )
            # Check for coordinates, where not same
            # - Discard missing scan , add NaN to missing scan
            self.NS = self.NS.combine_first(ENV.NS)  # xr.m
            self.HS = self.HS.combine_first(ENV.MS)
        else:
            raise ValueError("Not implemented for 1B-Ku,1B-Ka and 2A-SLH")
