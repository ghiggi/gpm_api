#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:40:12 2022

@author: ghiggi
"""
import numpy as np
import xarray as xr
from gpm_api.patch.labels import label_xarray_object
from gpm_api.patch.utils import labels_bbox_slices, extend_row_col_slices

####--------------------------------------------------------------------------.
#########################
#### Patch Generator ####
#########################
# TODO:
# patch centered on max, min labels
# patch on center_of_mass for area, mean, median, sum, sd labels

# REFACTOR HEAVILY !!! 
# REFACTOR ALSO USING STUFFS IN UTILS !
# if patch_size not specified, it become label area 
# --> choose if to add something around label area if patch_size=None

# _get_labeled_xr_obj_patch_gen used in visualization.labels 


# Note
# - get_patch_generator available as ds.gpm_api.patch_generator


def _get_labeled_xr_obj_patch_gen(
    xr_obj, label_name, n_patches=None, patch_margin=None, patch_size=None
):
    # In memory label array
    arr = np.asanyarray(xr_obj[label_name].data)

    # Ensure 0 label is set to nan
    arr = arr.astype(float) # otherwise if int throw an error when assigning nan
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
    dims = xr_obj[label_name].dims
    shape = arr.shape

    # Return xr.Dataset patches
    for label_id in np.arange(1, n_patches):
        row_slice, col_slice = labels_bbox_slices(arr == label_id)
        row_slice, col_slice = extend_row_col_slices(
            row_slice, col_slice, shape, margin=patch_margin
        )
        subset_isel_dict = {dims[0]: row_slice, dims[1]: col_slice}
        yield xr_obj.isel(subset_isel_dict)


def get_patch_generator(
    xr_obj,
    variable=None,
    min_value_threshold=-np.inf,
    max_value_threshold=np.inf,
    min_area_threshold=1,
    max_area_threshold=np.inf,
    footprint=None,
    sort_by="area",
    sort_decreasing=True,
    n_patches=None,
    patch_margin=None,
):  
    # Retrieve labeled xarray object 
    xr_obj = label_xarray_object(xr_obj, 
                                 variable=variable,
                                 min_value_threshold=min_value_threshold,
                                 max_value_threshold=max_value_threshold,
                                 min_area_threshold=min_area_threshold,
                                 max_area_threshold=max_area_threshold,
                                 footprint=footprint,
                                 sort_by=sort_by,
                                 sort_decreasing=sort_decreasing,
                                 label_name="label",
                                 )
    # Define the patch generator 
    patch_gen = _get_labeled_xr_obj_patch_gen(
        xr_obj, 
        label_name="label", 
        n_patches=n_patches, 
        patch_margin=patch_margin
    )
    
    return patch_gen
