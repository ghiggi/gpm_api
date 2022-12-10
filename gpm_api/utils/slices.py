#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 18:46:00 2022

@author: ghiggi
"""
import numpy as np


def ensure_is_slice(slc):
    if isinstance(slc, slice):
        return slc
    else:
        if isinstance(slc, int):
            slc = slice(slc, slc + 1)
        elif isinstance(slc, (list, tuple)) and len(slc) == 1:
            slc = slice(slc[0], slc[0] + 1)
        elif isinstance(slc, np.ndarray) and slc.size == 1:
            slc = slice(slc.item(), slc.item() + 1)
        else:
            # TODO: check if continuous
            raise ValueError("Impossibile to convert to a slice object.")
    return slc


def get_slice_size(slc):
    if not isinstance(slc, slice):
        raise TypeError("Expecting slice object")
    size = slc.stop - slc.start
    return size
