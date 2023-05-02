#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:34:53 2022

@author: ghiggi
"""
# import dask_image.ndmeasure
# from dask_image.ndmeasure import as dask_label_image
import dask_image.ndmeasure
import numpy as np
import xarray as xr
from skimage.measure import label as label_image
from skimage.morphology import binary_dilation, disk

# TODO:
# - xr_get_label_stats
# - Enable to label in n-dimensions
#   - (2D+VERTICAL) --> CORE PROFILES
#   - (2D+TIME) --> TRACKING

# Future internal renaming:
# - get_areas_labels --> get_labels
# - xr_get_areas_labels --> xr_get_labels

# Note
# - label_xarray_object available as ds.gpm_api.label_object


####--------------------------------------------------------------------------.
#####################
#### Area labels ####
#####################
def _mask_buffer(mask, footprint):
    """Dilate the mask by n pixel in all directions.

    If footprint = 0 or None, no dilation occur.
    If footprint is a positive integer, it create a disk(footprint)
    If footprint is a 2D array, it must represent the neighborhood expressed
    as a 2-D array of 1’s and 0’s.
    For more info: https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.binary_dilation

    """
    # scikitimage > 0.19
    if not isinstance(footprint, (int, np.ndarray, type(None))):
        raise TypeError("`footprint` must be an integer, numpy 2D array or None.")
    if isinstance(footprint, np.ndarray):
        if footprint.ndim != 2:
            raise ValueError("If providing the footprint for dilation as np.array, it must be 2D.")
    if isinstance(footprint, int):
        if footprint < 0:
            raise ValueError("Footprint must be equal or larger than 1.")
        if footprint == 0:
            footprint = None
        else:
            footprint = disk(radius=footprint)
    # Apply dilation
    if footprint is not None:
        mask = binary_dilation(mask, footprint=footprint)
    return mask


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
    valid_values = ["area", "max", "min", "sum", "mean", "median", "sd"]
    if not isinstance(value, str):
        raise TypeError(f"'sort_by' must be a string. Valid values are: {valid_values} .")
    if value not in valid_values:
        raise TypeError(f"Valid 'sort_by' values are: {valid_values}.")


def _get_label_value_stats(arr, label_arr, label_indices=None, stats="area"):
    """Compute label value statistics over which to later sort on.

    If labels_indices is None, it compute the statistic for each label.
    """
    # For custom function:
    # https://image.dask.org/en/latest/dask_image.ndmeasure.html#dask_image.ndmeasure.labeled_comprehension
    # Note:
    # - if label_indices None, by default would return the stats of the entire array
    # - if label_indices is 0, return nan
    # - If label_indices is not inside label_arr, return 0

    if label_indices is None:
        label_indices = np.unique(label_arr)

    if stats == "area":
        values = dask_image.ndmeasure.area(image=arr, label_image=label_arr, index=label_indices)
    elif stats == "max":
        values = dask_image.ndmeasure.maximum(image=arr, label_image=label_arr, index=label_indices)
    elif stats == "min":
        values = dask_image.ndmeasure.minimum(image=arr, label_image=label_arr, index=label_indices)
    elif stats == "mean":
        values = dask_image.ndmeasure.mean(image=arr, label_image=label_arr, index=label_indices)
    elif stats == "median":
        values = dask_image.ndmeasure.median(image=arr, label_image=label_arr, index=label_indices)
    elif stats == "sum":
        values = dask_image.ndmeasure.sum_labels(
            image=arr, label_image=label_arr, index=label_indices
        )
    elif stats == "sd":
        values = dask_image.ndmeasure.standard_deviation(
            image=arr, label_image=label_arr, index=label_indices
        )
    else:
        raise NotImplementedError()
    # Compute values
    values = values.compute()
    # Return values
    return values


def get_labels_stats(arr, label_arr, label_indices=None, stats="area", sort_decreasing=True):
    """Return label and label statistics sorted by statistic value."""
    # Get labels area values
    values = _get_label_value_stats(
        arr, label_arr=label_arr, label_indices=label_indices, stats=stats
    )
    # Get sorting index based on values
    if sort_decreasing:
        sort_index = np.argsort(values)[::-1]
    else:
        sort_index = np.argsort(values)

    # Sort values
    values = values[sort_index]
    label_indices = label_indices[sort_index]

    return label_indices, values


def _vec_translate(arr, my_dict):
    """Remap array <value> based on the dictionary key-value pairs.

    This function is used to redefine label array integer values based on the
    label area_size/max_intensity value.

    """
    return np.vectorize(my_dict.__getitem__)(arr)


def get_labels_with_requested_occurence(label_arr, vmin, vmax):
    "Get label indices with requested occurence."
    # Compute label occurence
    label_indices, label_occurence = np.unique(label_arr, return_counts=True)

    # Remove label 0 and associate pixel count if present
    if label_indices[0] == 0:
        label_indices = label_indices[1:]
        label_occurence = label_occurence[1:]
    # Get index with required occurence
    valid_area_indices = np.where(np.logical_and(label_occurence >= vmin, label_occurence <= vmax))[
        0
    ]
    # Return list of valid label indices
    if len(valid_area_indices) > 0:
        label_indices = label_indices[valid_area_indices]
    else:
        label_indices = []
    return label_indices


def redefine_label_array(label_arr, label_indices=None):
    """Redefine labels of a label array from 0 to len(label_indices).

    If label_indices is None, it takes the unique values of label_arr.
    If label_indices contains a 0, it is discarded !
    If label_indices is not unique, raise an error !

    Native label values not present in label_indices are set to 0.
    The first label in label_indices becomes 1, the second 2, and so on.
    """
    if label_indices is None:
        label_indices = np.unique(label_arr)
    else:
        # Check unique values are provided
        _, c = np.unique(label_indices, return_counts=True)
        if np.any(c > 1):
            raise ValueError("'label_indices' must be uniques.")

    # Remove 0 and nan if present in label_indices
    label_indices = np.delete(label_indices, np.where(label_indices == 0)[0].flatten())
    label_indices = np.delete(label_indices, np.where(np.isnan(label_indices))[0].flatten())

    # Ensure label indices are integer
    label_indices = label_indices.astype(int)

    # Remove nan from label_arr
    label_arr[np.isnan(label_arr)] = 0

    # Compute max label index
    max_label = max(label_indices)

    # Set to 0 labels in label_arr larger than max_label
    # - These are some of the labels that were set to 0 because of mask or area filtering
    label_arr[label_arr > max_label] = 0

    # Initialize dictionary with keys corresponding to all possible labels indices
    val_dict = {k: 0 for k in range(0, max_label + 1)}

    # Update the dictionary keys with the selected label_indices
    # - Assume 0 not in label_indices
    n_labels = len(label_indices)
    label_indices = label_indices.tolist()
    label_indices_new = np.arange(1, n_labels + 1).tolist()
    for k, v in zip(label_indices, label_indices_new):
        val_dict[k] = v

    # Remove keys not in label_arr
    # TODO: to speed up _vec_translate maybe

    # Redefine the id of the labels
    labels_arr = _vec_translate(label_arr, val_dict)

    return labels_arr


####--------------------------------------------------------------------------.
def get_areas_labels(
    arr,
    min_value_threshold=-np.inf,
    max_value_threshold=np.inf,
    min_area_threshold=1,
    max_area_threshold=np.inf,
    footprint=None,
    sort_by="area",
    sort_decreasing=True,
):
    # footprint: The neighborhood expressed as a 2-D array of 1’s and 0’s.
    # ---------------------------------.
    # TODO: this could be extended to work with dask >2D array
    # - dask_image.ndmeasure.label  https://image.dask.org/en/latest/dask_image.ndmeasure.html
    # - dask_image.ndmorph.binary_dilation https://image.dask.org/en/latest/dask_image.ndmorph.html#dask_image.ndmorph.binary_dilation

    # ---------------------------------.
    # Check array validity
    arr = _check_array(arr)

    # Check input arguments
    check_sort_by(sort_by)

    # ---------------------------------.
    # Define masks
    # - mask_native: True when between min and max thresholds
    # - mask_nan: True where is not finite (inf or nan)
    mask_native = np.logical_and(arr >= min_value_threshold, arr <= max_value_threshold)
    mask_nan = ~np.isfinite(arr)
    # ---------------------------------.
    # Dilate (buffer) the native mask
    # - This enable to assign closely connected mask_native areas to the same label
    mask = _mask_buffer(mask_native, footprint=footprint)

    # ---------------------------------.
    # Get area labels
    # - 0 represent the outer area
    label_arr = label_image(mask)  # 0.977-1.37 ms

    # mask = mask.astype(int)
    # labels, num_features = dask_label_image(mask) # THIS WORK in ND dimensions
    # %time labels = labels.compute()    # 5-6.5 ms

    # ---------------------------------.
    # Count initial label occurence
    label_indices, label_occurence = np.unique(label_arr, return_counts=True)
    n_initial_labels = len(label_indices)
    if n_initial_labels == 1:  # only 0 label
        return _no_labels_result(arr)

    # ---------------------------------.
    # Set areas outside the mask_native to label value 0
    label_arr[~mask_native] = 0

    # Set NaN pixels to label value 0
    label_arr[mask_nan] = 0

    # ---------------------------------.
    # Filter label by area
    label_indices = get_labels_with_requested_occurence(
        label_arr=label_arr, vmin=min_area_threshold, vmax=max_area_threshold
    )
    if len(label_indices) == 0:
        return _no_labels_result(arr)

    # ---------------------------------.
    # Sort labels by statistics (i.e. label area, label max value ...)
    label_indices, values = get_labels_stats(
        arr=arr,
        label_arr=label_arr,
        label_indices=label_indices,
        stats=sort_by,
        sort_decreasing=sort_decreasing,
    )

    # ---------------------------------.
    # Relabel labels array (from 1 to n_labels)
    labels_arr = redefine_label_array(label_arr=label_arr, label_indices=label_indices)
    n_labels = len(label_indices)
    # ---------------------------------.
    # Return infos
    return labels_arr, n_labels, values


def xr_get_areas_labels(
    data_array,
    min_value_threshold=-np.inf,
    max_value_threshold=np.inf,
    min_area_threshold=1,
    max_area_threshold=np.inf,
    footprint=None,
    sort_by="area",
    sort_decreasing=True,
):
    # Extract data from DataArray
    if not isinstance(data_array, xr.DataArray):
        raise TypeError("Expecting xr.DataArray.")
    # Get labels
    labels_arr, n_labels, values = get_areas_labels(
        arr=data_array.data,
        min_value_threshold=min_value_threshold,
        max_value_threshold=max_value_threshold,
        min_area_threshold=min_area_threshold,
        max_area_threshold=max_area_threshold,
        footprint=footprint,
        sort_by=sort_by,
        sort_decreasing=sort_decreasing,
    )

    # Conversion to DataArray if needed
    da_labels = data_array.copy()
    da_labels.data = labels_arr
    da_labels.name = f"labels_{sort_by}"
    da_labels.attrs = {}
    return da_labels, n_labels, values


def _check_xr_obj(xr_obj, variable):
    """Check xarray object and variable validity."""
    # Check inputs
    if not isinstance(xr_obj, (xr.Dataset, xr.DataArray)):
        raise TypeError("'xr_obj' must be a xr.Dataset or xr.DataArray.")
    if isinstance(xr_obj, xr.Dataset):
        # Check valid variable is specified
        if variable is None:
            raise ValueError("An xr.Dataset 'variable' must be specified.")
        if variable not in xr_obj.data_vars:
            raise ValueError(f"'{variable}' is not a variable of the xr.Dataset.")
    else:
        if variable is not None:
            raise ValueError("'variable' must not be specified when providing a xr.DataArray.")


def label_xarray_object(
    xr_obj,
    variable=None,
    min_value_threshold=-np.inf,
    max_value_threshold=np.inf,
    min_area_threshold=1,
    max_area_threshold=np.inf,
    footprint=None,
    sort_by="area",
    sort_decreasing=True,
    label_name="label",
):
    # Check xarray input
    _check_xr_obj(xr_obj=xr_obj, variable=variable)

    # Retrieve labels (if available)
    if isinstance(xr_obj, xr.Dataset):
        data_array_to_label = xr_obj[variable]
    else:
        data_array_to_label = xr_obj

    da_labels, n_labels, values = xr_get_areas_labels(
        data_array=data_array_to_label,
        min_value_threshold=min_value_threshold,
        max_value_threshold=max_value_threshold,
        min_area_threshold=min_area_threshold,
        max_area_threshold=max_area_threshold,
        footprint=footprint,
        sort_by=sort_by,
        sort_decreasing=sort_decreasing,
    )
    if n_labels == 0:
        raise ValueError(
            "No patch available. You might want to change the patch generator parameters."
        )

    da_labels = da_labels.where(da_labels > 0)

    # Assign label to xr.DataArray  coordinate
    xr_obj = xr_obj.assign_coords({label_name: da_labels})

    return xr_obj
