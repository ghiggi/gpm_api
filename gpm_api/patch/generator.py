#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:40:12 2022

@author: ghiggi
"""
import numpy as np 
from gpm_api.patch.labels import xr_get_areas_labels
from gpm_api.patch.utils import labels_bbox_slices, extend_row_col_slices


#########################
#### Patch Generator ####
#########################
# TODO: 
# patch centered on max, min labels
# patch on center_of_mass for area, mean, median, sum, sd labels
    
def get_dataset_labels_patches(ds, label, n_patches=None, patch_margin=None, patch_size=None):
    # In memory label array 
    arr = np.asanyarray(ds[label].data)
    
    # Ensure 0 label is set to nan 
    arr[arr == 0] = np.nan
    
    # Get total number of labels 
    n_labels = len(np.unique(arr[~np.isnan(arr)]))
    if n_labels == 0: 
        raise ValueError("No labels available. Only nans (or 0).")
    
    # Define n_patches 
    if n_patches is None: 
        n_patches = len(np.unique(arr[~np.isnan(arr)]))
    else: 
        if n_patches > n_labels:
           n_patches = n_labels
           
    # Retrieve dimensions and shape 
    dims = ds[label].dims
    shape = arr.shape

    # Return xr.Dataset patches  
    for label_id in np.arange(1, n_patches):
        row_slice, col_slice = labels_bbox_slices(arr==label_id)
        row_slice, col_slice = extend_row_col_slices(row_slice, col_slice, shape, margin=patch_margin) 
        subset_isel_dict = {dims[0]: row_slice, dims[1]: col_slice}
        yield ds.isel(subset_isel_dict)


def get_ds_patch_generator(ds,
                           variable, 
                           min_value_threshold=-np.inf, 
                           max_value_threshold= np.inf, 
                           min_area_threshold=1, 
                           max_area_threshold=np.inf,
                           footprint_buffer=None,
                           sort_by="area",
                           sort_decreasing=True,
                           label_name="label",
                           n_patches=None,
                           patch_margin=(48,20)):
    # Check valid variable 
    if variable not in ds.data_vars:
        raise ValueError(f"'{variable}' is not a variable of the GPM xr.Dataset.")
    # Retrieve rainy area labels (if available)
    da_labels, n_labels, values = xr_get_areas_labels(data_array=ds[variable],
                                                      min_value_threshold=min_value_threshold, 
                                                      max_value_threshold=max_value_threshold, 
                                                      min_area_threshold=min_area_threshold, 
                                                      max_area_threshold=max_area_threshold,
                                                      footprint_buffer=footprint_buffer,
                                                      sort_by=sort_by,
                                                      sort_decreasing=sort_decreasing,
                                                      )     
    da_labels = da_labels.where(da_labels > 0)
    
    # Assign label to xr.DataArray  coordinate 
    ds = ds.assign_coords({label_name: da_labels})
     
    # Build a generator returning patches around rainy areas  
    ds_patch_gen = get_dataset_labels_patches(ds, 
                                              label=label_name,
                                              n_patches=n_patches, 
                                              patch_margin=patch_margin)
    return ds_patch_gen


def get_da_patch_generator(data_array,
                           min_value_threshold=-np.inf, 
                           max_value_threshold= np.inf, 
                           min_area_threshold=1, 
                           max_area_threshold=np.inf,
                           footprint_buffer=None,
                           sort_by="area",
                           sort_decreasing=True,
                           label_name="label",
                           n_patches=None,
                           patch_margin=(48,20)):
    
    # Retrieve rainy area labels (if available)
    da_labels, n_labels, values = xr_get_areas_labels(data_array=data_array,
                                                      min_value_threshold=min_value_threshold, 
                                                      max_value_threshold=max_value_threshold, 
                                                      min_area_threshold=min_area_threshold, 
                                                      max_area_threshold=max_area_threshold,
                                                      footprint_buffer=footprint_buffer,
                                                      sort_by=sort_by,
                                                      sort_decreasing=sort_decreasing,
                                                      )     
    da_labels = da_labels.where(da_labels > 0)
    
    # Assign label to xr.DataArray  coordinate 
    data_array = data_array.assign_coords({label_name: da_labels})
     
    # Build a generator returning patches around rainy areas  
    da_patch_gen = get_dataset_labels_patches(data_array, 
                                              label=label_name,
                                              n_patches=n_patches, 
                                              patch_margin=patch_margin)
    return da_patch_gen

