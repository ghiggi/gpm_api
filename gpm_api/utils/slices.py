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


# tests for _get_contiguous_true_slices
# bool_arr = np.array([True, False, True, True, True])
# bool_arr = np.array([True, True, True, False, True])
# bool_arr = np.array([True, True, True, True, False])
# bool_arr = np.array([True, False, False, True, True])
# bool_arr = np.array([True, True, True, False, False])
# bool_arr = np.array([False, True, True, True, False])
# bool_arr = np.array([False, False, True, True, True])
# bool_arr = np.array([False])


def get_contiguous_true_slices(
    bool_arr, include_false=True, skip_consecutive_false=True
):
    """Return the slices corresponding to sequences of True in the input arrays.

    If include_false=True, the last element of each slice sequence (except the last) will be False
    If include_false=False, no element in each slice sequence will be False
    If skip_consecutive_false=True (default), the first element of each slice must be a True.
    If skip_consecutive_false=False, it returns also slices of size 1 which selects just the False value.
    Note: if include_false = False, skip_consecutive_false is automatically True.
    """
    # Check the arguments
    if not include_false:
        skip_consecutive_false = True

    # If all True
    if np.all(bool_arr):
        list_slices = [slice(0, len(bool_arr))]
    # If all False
    elif np.all(~bool_arr):
        if skip_consecutive_false:
            list_slices = []
        else:
            list_slices = [slice(i, i + 1) for i in range(0, len(bool_arr))]
    # If True and False
    else:
        # Retrieve indices where False start to occur
        false_indices = np.argwhere(~bool_arr).flatten()
        # Prepend -1 so first start start at 0, if no False at idx 0
        false_indices = np.append(-1, false_indices)
        list_slices = []
        for i in range(1, len(false_indices)):
            idx_before = false_indices[i - 1]
            idx = false_indices[i]
            if skip_consecutive_false:
                if idx - idx_before == 1:
                    continue
            # Define start
            start = idx_before + 1
            # Define stop
            if include_false:
                stop = idx + 1
            else:
                stop = idx
            # Define slice
            slc = slice(start, stop)
            list_slices.append(slc)

        # Includes the last slice (if the last bool_arr element is not False)
        if idx < len(bool_arr) - 1:
            start = idx + 1
            stop = len(bool_arr)
            slc = slice(start, stop)
            list_slices.append(slc)

    # Return list of slices
    return list_slices


def filter_slices_by_size(list_slices, min_size=None, max_size=None):
    """Filter list of slices by size."""
    if min_size is None and max_size is None:
        return list_slices
    # Define min and max size if one is not specified
    min_size = 0 if min_size is None else min_size
    max_size = np.inf if max_size is None else max_size
    # Get list of slice sizes
    sizes = [
        get_slice_size(slc) if isinstance(slc, slice) else 0 for slc in list_slices
    ]
    # Retrieve valid slices
    valid_bool = np.logical_and(
        np.array(sizes) >= min_size, np.array(sizes) <= max_size
    )
    list_slices = np.array(list_slices)[valid_bool].tolist()
    return list_slices


def _get_list_slices_from_indices(indices):
    """Return a list of slices from a list/array of integer indices.
    Example:
        [0,1,2,4,5,8] --> [slices(0,3),slice(4,6), slice(8,9)]
    """
    # TODO ... used in used in get_extent_slices , _replace_0_values --> get ascend/desc_slices
    # Checks
    indices = np.asarray(indices).astype(int)
    indices = sorted(np.unique(indices))
    if np.any(np.sign(indices) < 0):
        raise ValueError("_get_list_slices_from_indices expects only positive"
                         " integer indices.")
    if len(indices) == 1:
        return [slice(indices[0], indices[0] + 1)]
    # Retrieve slices
    # idx_splits = np.where(np.diff(indices) > 1)[0]
    # if len(idx_splits) == 0:
    #     list_slices = [slice(min(indices), max(indices))]
    # else:
    #     list_idx = np.split(indices, idx_splits+1)
    #     list_slices = [slice(x.min(), x.max()+1) for x in list_idx]
    start = indices[0]
    previous = indices[0]
    list_slices = []
    for idx in indices[1:]:
        if idx - previous == 1:
            previous = idx
        else:
            list_slices.append(slice(start, previous + 1))
            start = idx
            previous = idx
    list_slices.append(slice(start, previous + 1))
    return list_slices