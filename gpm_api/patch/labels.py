#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:34:53 2022

@author: ghiggi
"""
import numpy as np 
import xarray as xr
# import dask_image.ndmeasure
# from dask_image.ndmeasure import as dask_label_image
import dask_image.ndmeasure
from skimage.measure import label as label_image
from skimage.morphology import binary_dilation

####--------------------------------------------------------------------------.
def _vec_translate(arr, my_dict):   
    """Remap array <value> based on the dictionary key-value pairs.
    
    This function is used to redefine label array integer values based on the
    label area_size/max_intensity value.
    
    """
    return np.vectorize(my_dict.__getitem__)(arr)


####--------------------------------------------------------------------------.
#####################
#### Area labels ####
#####################


def _check_array(arr):
    shape = arr.shape 
    if len(shape) != 2: 
        raise ValueError("Expecting a 2D array.")
    if np.any(np.array(shape) == 0):
        raise ValueError("Expecting non-zero dimensions.")
    
    if not isinstance(arr, np.ndarray):
        arr = arr.compute()
    return arr 


def _no_labels_result(arr):
    labels = np.zeros(arr.shape)
    n_labels = 0
    counts = []
    return labels, n_labels, counts


def check_sort_by(value): 
    valid_values =  ["area", "max", "min", "sum", "mean", "median", "sd"]
    if not isinstance(value, str): 
        raise TypeError(f"'sort_by' must be a string. Valid values are: {valid_values} .")
    if value not in valid_values:
        raise TypeError(f"Valid 'sort_by' values are: {valid_values}.")
        

def get_sorting_area_values(arr, label_arr, label_indices, sort_by= "area"):
    """Compute area label values over which to later sort on."""
    # For custom function:
    # https://image.dask.org/en/latest/dask_image.ndmeasure.html#dask_image.ndmeasure.labeled_comprehension
    
    if sort_by == "area":
        values = dask_image.ndmeasure.area(image=arr, 
                                           label_image=label_arr,
                                           index=label_indices)
    elif sort_by == "max":    
         values = dask_image.ndmeasure.maximum(image=arr, 
                                               label_image=label_arr,
                                               index=label_indices)
    elif sort_by == "min":    
         values = dask_image.ndmeasure.minimum(image=arr, 
                                               label_image=label_arr,
                                               index=label_indices)   
    elif sort_by == "mean":
         values =  dask_image.ndmeasure.mean(image=arr, 
                                             label_image=label_arr,
                                             index=label_indices)
    elif sort_by == "median":
         values =  dask_image.ndmeasure.median(image=arr, 
                                               label_image=label_arr,
                                               index=label_indices) 
    elif sort_by == "sum":
         values =  dask_image.ndmeasure.sum_labels(image=arr, 
                                                   label_image=label_arr,
                                                   index=label_indices) 
    elif sort_by == "sd":
         values =  dask_image.ndmeasure.standard_deviation(image=arr, 
                                                            label_image=label_arr,
                                                            index=label_indices)
    else: 
         raise NotImplementedError()
    # Compute values 
    values = values.compute()
    # Return values 
    return values 
 
    
def get_areas_labels(arr, 
                     min_value_threshold=-np.inf, 
                     max_value_threshold= np.inf, 
                     min_area_threshold=1, 
                     max_area_threshold=np.inf,
                     footprint_buffer=None,
                     sort_by="area",
                     sort_decreasing=True):
    # footprint_buffer: The neighborhood expressed as a 2-D array of 1’s and 0’s. 
    #---------------------------------.
    # TODO: this could be extended to work with dask >2D array 
    # - dask_image.ndmeasure.label  https://image.dask.org/en/latest/dask_image.ndmeasure.html
    # - dask_image.ndmorph.binary_dilation https://image.dask.org/en/latest/dask_image.ndmorph.html#dask_image.ndmorph.binary_dilation
    
    #---------------------------------.
    # Check array validity 
    arr = _check_array(arr)

    # Check input arguments 
    check_sort_by(sort_by)
    
    if min_value_threshold == -np.inf and max_value_threshold == np.inf:
        raise ValueError("Specify at least 'min_value_threshold' or 'max_value_threshold'.")
    
    #---------------------------------.
    # Define binary mask 
    mask_native = np.logical_and(arr >= min_value_threshold,
                                 arr <= max_value_threshold,
                                 ~np.isfinite(arr))
    
    mask_native[~np.isfinite(arr)] = True # masked later below 
    #---------------------------------.
    # Apply optional buffering
    if footprint_buffer is not None: 
       mask = binary_dilation(mask_native, footprint_buffer)
    else: 
       mask = mask_native
       
    #---------------------------------.
    # Get area labels
    # - 0 represent the outer area
    label_arr = label_image(mask)           # 0.977-1.37 ms
    
    # mask = mask.astype(int)
    # labels, num_features = dask_label_image(mask) # THIS WORK in ND dimensions
    # %time labels = labels.compute()    # 5-6.5 ms 
    
    #---------------------------------.
    # Count initial label occurence 
    label_indices, label_occurence = np.unique(label_arr, return_counts=True)
    n_initial_labels = len(label_indices)
    if n_initial_labels == 1: # only 0 label
        return _no_labels_result(arr)
    
    #---------------------------------.
    # Set areas outside the native mask to label value 0   
    label_arr[~mask_native] = 0
    
    # Set NaN pixels to label value 0  
    label_arr[~np.isfinite(arr)] = 0
    
    #---------------------------------.
    # Recompute label indices 
    label_indices, label_occurence = np.unique(label_arr, return_counts=True)
    n_initial_labels = len(label_indices)
    
    #---------------------------------.
    # Remove 0 label and associate pixel count
    label_indices = label_indices[1:]
    label_occurence = label_occurence[1:]
    
    #---------------------------------.
    # Filter by area 
    valid_area_indices = np.where(np.logical_and(label_occurence >= min_area_threshold,
                                                 label_occurence <= max_area_threshold))[0]
    if len(valid_area_indices) > 0:
        label_indices = label_indices[valid_area_indices]
    else: 
        return _no_labels_result(arr)
    
    #---------------------------------.
    # Sort labels    
    values = get_sorting_area_values(arr, 
                                     label_arr=label_arr ,
                                     label_indices=label_indices,
                                     sort_by=sort_by)
    if sort_decreasing:
        sort_index = np.argsort(values)[::-1] 
    else: 
        sort_index = np.argsort(values)
        
    # Sort values      
    values = values[sort_index]
    label_indices = label_indices[sort_index]
    
    #---------------------------------.
    # Relabel labels array (from 1 to n_labels)
    n_labels = len(label_indices)
    val_dict = {k: 0 for k in range(0, n_initial_labels+1)} 
    label_indices = label_indices.tolist()
    label_indices_new = np.arange(1, n_labels+1).tolist()
    for k, v in zip(label_indices, label_indices_new):
        val_dict[k] = v
    labels_arr = _vec_translate(label_arr, val_dict)
    
    #---------------------------------.
    # Return infos         
    return labels_arr, n_labels, values


def xr_get_areas_labels(data_array,
                        min_value_threshold=-np.inf, 
                        max_value_threshold= np.inf, 
                        min_area_threshold=1, 
                        max_area_threshold=np.inf,
                        footprint_buffer=None,
                        sort_by="area",
                        sort_decreasing=True,
                        ): 
    # Extract data from DataArray
    if not isinstance(data_array, xr.DataArray): 
        raise TypeError("Expecting xr.DataArray.")
    # Get labels 
    labels_arr, n_labels, values = get_areas_labels(arr=data_array.data, 
                                                    min_value_threshold=min_value_threshold, 
                                                    max_value_threshold=max_value_threshold, 
                                                    min_area_threshold=min_area_threshold, 
                                                    max_area_threshold=max_area_threshold,
                                                    footprint_buffer=footprint_buffer,
                                                    sort_by=sort_by, 
                                                    sort_decreasing=sort_decreasing, 
                                                    )  
     
    # Conversion to DataArray if needed
    da_labels = data_array.copy() 
    da_labels.data = labels_arr 
    da_labels.name = f"labels_{sort_by}"
    return da_labels, n_labels, values

