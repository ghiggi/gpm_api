#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:18:30 2022

@author: ghiggi
"""
import datetime

start_time = datetime.datetime(2020, 7, 11, 0, 25, 25)
end_time = datetime.datetime(2020, 7, 11, 0, 26, 27)
date = datetime.datetime(2020, 7, 11, 0, 0)

start_hhmmss = datetime.datetime.strftime(start_time, "%H%M%S")
end_hhmmss = datetime.datetime.strftime(end_time, "%H%M%S")


find_daily_filepaths


### GPM-REFACTOR FILTERING
### USE TROLL SIFT

# MUST ALWAYS LOOK IN THE DAY BEFORE because 00-01-02 could be in previous day
