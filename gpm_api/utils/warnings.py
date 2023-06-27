#!/usr/bin/env python3
"""
Created on Thu Jan 19 15:30:36 2023

@author: ghiggi
"""


class GPM_Warning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class GPMDownloadWarning(Warning):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)
