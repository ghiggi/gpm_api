#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:15:34 2022

@author: ghiggi
"""
import gpm_api
import datetime

base_dir = "/ltenas3/0_Data/GPM"
base_dir = "/home/ghiggi/GPM"

start_time = datetime.datetime.strptime("2020-07-05 00:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-09-01 00:00:00", "%Y-%m-%d %H:%M:%S")

# Load GPM


# Create non-regular timesteps
timesteps = ds_gpm["time"].values
timesteps[5:10] = timesteps[100:105]

# Check if regular timesteps
ds_gpm.gpm_api.has_regular_time

# Retrieve slices of regular timesteps
list_slices = ds_gpm.gpm_api.get_slices_regular_time()
print(list_slices)

# Subset time using time slice
ds_regular = ds_gpm.gpm_api.subset_by_time_slice(slice=list_slices[1])

# Check if regular timesteps
ds_regular.gpm_api.has_regular_time
