#!/usr/bin/env python3
"""
Created on Sat Dec 10 18:46:00 2022

@author: ghiggi
"""
from functools import reduce

import numpy as np

####---------------------------------------------------------------------------.
#### Tools for list_slices


def get_list_slices_from_indices(indices):
    """Return a list of slices from a list/array of integer indices.

    Example:
        [0,1,2,4,5,8] --> [slices(0,3),slice(4,6), slice(8,9)]
    """
    if isinstance(indices, (int, float)):
        indices = [indices]
    # Checks
    if len(indices) == 0:
        list_slices = []
        return list_slices
    indices = np.asarray(indices).astype(int)
    indices = sorted(np.unique(indices))
    if np.any(np.sign(indices) < 0):
        raise ValueError("get_list_slices_from_indices expects only positive" " integer indices.")
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


def get_indices_from_list_slices(list_slices, check_non_intersecting=True):
    """Return a numpy array of indices from a list of slices."""
    if len(list_slices) == 0:
        return np.array([])
    list_indices = [np.arange(slc.start, slc.stop, slc.step) for slc in list_slices]
    indices, counts = np.unique(np.concatenate(list_indices), return_counts=True)
    if check_non_intersecting and np.any(counts > 1):
        raise ValueError("The list of slices contains intersecting slices!")
    return indices


def list_slices_intersection(*args):
    """Return the intersecting slices from multiple list of slices."""
    list_indices = [get_indices_from_list_slices(l_slc) for l_slc in list(args)]
    intersect_indices = reduce(np.intersect1d, list_indices)
    return get_list_slices_from_indices(intersect_indices)


def list_slices_union(*args):
    """Return the union slices from multiple list of slices."""
    list_indices = [get_indices_from_list_slices(l_slc) for l_slc in list(args)]
    union_indices = np.unique(np.concatenate(list_indices))
    return get_list_slices_from_indices(union_indices)


def list_slices_difference(list_slices1, list_slices2):
    """Return the list of slices covered by list_slices1 not intersecting list_slices2."""
    list_indices1 = get_indices_from_list_slices(list_slices1)
    list_indices2 = get_indices_from_list_slices(list_slices2)
    diff_indices = np.setdiff1d(list_indices1, list_indices2)
    list_slices = get_list_slices_from_indices(diff_indices)
    return list_slices


def list_slices_combine(*args):
    """Combine together a list of list_slices, without any additional operation."""
    return [slc for list_slices in args for slc in list_slices]


def list_slices_simplify(list_slices):
    """Simplify list of of sequential slices.

    Example 1: [slice(0,2), slice(2,4)] --> [slice(0,4)]
    """
    if len(list_slices) <= 1:
        return list_slices
    indices = get_indices_from_list_slices(list_slices, check_non_intersecting=False)
    return get_list_slices_from_indices(indices)


def _list_slices_sort(list_slices):
    """Sort a single list of slices."""
    return sorted(list_slices, key=lambda x: x.start)


def list_slices_sort(*args):
    """Sort a single or multiple list of slices by slice.start.

    It output a single list of slices!
    """
    list_slices = list_slices_combine(*args)
    list_slices = _list_slices_sort(list_slices)
    return list_slices


def list_slices_filter(list_slices, min_size=None, max_size=None):
    """Filter list of slices by size."""
    if min_size is None and max_size is None:
        return list_slices
    # Define min and max size if one is not specified
    min_size = 0 if min_size is None else min_size
    max_size = np.inf if max_size is None else max_size
    # Get list of slice sizes
    sizes = [get_slice_size(slc) if isinstance(slc, slice) else 0 for slc in list_slices]
    # Retrieve valid slices
    valid_bool = np.logical_and(np.array(sizes) >= min_size, np.array(sizes) <= max_size)
    list_slices = np.array(list_slices)[valid_bool].tolist()
    return list_slices


def list_slices_flatten(list_slices):
    """Flatten out list of slices with 2 nested level.

    Examples:
    [[slice(1, 7934, None)], [slice(1, 2, None)]] --> [slice(1, 7934, None), slice(1, 2, None)]
    [slice(1, 7934, None), slice(1, 2, None)] --> [slice(1, 7934, None), slice(1, 2, None)]
    """
    flat_list = []
    for sublist in list_slices:
        if isinstance(sublist, list):
            for item in sublist:
                flat_list.append(item)
        else:
            flat_list.append(sublist)
    return flat_list


def get_list_slices_from_bool_arr(bool_arr, include_false=True, skip_consecutive_false=True):
    """Return the slices corresponding to sequences of True in the input arrays.

    If include_false=True, the last element of each slice sequence (except the last) will be False
    If include_false=False, no element in each slice sequence will be False
    If skip_consecutive_false=True (default), the first element of each slice must be a True.
    If skip_consecutive_false=False, it returns also slices of size 1 which selects just the False value.
    Note: if include_false = False, skip_consecutive_false is automatically True.

    Examples:
    If include_false=True and skip_consecutive_false=False:
       --> [False, False] --> [slice(0,1), slice(1,2)]
    If include_false=True and skip_consecutive_false=True:
       --> [False, False] --> []
       --> [False, False, True] --> [slice(2,3)]
       --> [False, False, True, False] --> [slice(2,4)]
    If include_false=False:
        --> [False, False, True, False] --> [slice(2,3)]

    """
    # Check the arguments
    if not include_false:
        skip_consecutive_false = True
    bool_arr = np.array(bool_arr)
    # If all True
    if np.all(bool_arr):
        list_slices = [slice(0, len(bool_arr))]
    # If all False
    elif np.all(~bool_arr):
        list_slices = (
            [] if skip_consecutive_false else [slice(i, i + 1) for i in range(0, len(bool_arr))]
        )
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
            if skip_consecutive_false and idx - idx_before == 1:
                continue
            # Define start
            start = idx_before + 1
            # Define stop
            stop = idx + 1 if include_false else idx
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


# tests for _get_list_slices_from_bool_arr
# bool_arr = np.array([True, False, True, True, True])
# bool_arr = np.array([True, True, True, False, True])
# bool_arr = np.array([True, True, True, True, False])
# bool_arr = np.array([True, False, False, True, True])
# bool_arr = np.array([True, True, True, False, False])
# bool_arr = np.array([False, True, True, True, False])
# bool_arr = np.array([False, False, True, True, True])
# bool_arr = np.array([False])


####----------------------------------------------------------------------------.
#### Tools for slice manipulation


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
    """Get size of the slice.

    Note: The actual slice size must not be representative of the true slice if
    slice.stop is larger than the length of object to be sliced.
    """
    if not isinstance(slc, slice):
        raise TypeError("Expecting slice object")
    size = slc.stop - slc.start
    return size


def get_idx_bounds_from_slice(slc):
    """Get start and end indices of the slice.

    Note: For index based selection, use idx_start:idx_end+1 !
    """
    if not isinstance(slc, slice):
        raise TypeError("Expecting slice object")
    idx_start = slice.start
    idx_end = slice.stop - 1
    return idx_start, idx_end


def get_slice_from_idx_bounds(idx_start, idx_end):
    """Return the slice required to include the idx bounds."""
    return slice(idx_start, idx_end + 1)


def pad_slice(slc, padding, min_start=0, max_stop=np.inf):
    """
    Increase/decrease the slice with the padding argument.

    Does not ensure that all output slices have same size.

    Parameters
    ----------
    slc : slice
        Slice objects.
    padding : int
        Padding to be applied to the slice.
    min_start : int, optional
       The minimum value for the start of the new slice.
       The default is 0.
    max_stop : int
        The maximum value for the stop of the new slice.
        The default is np.inf.
    Returns
    -------
    list_slices : TYPE
        The list of slices after applying padding.
    """

    new_slice = slice(max(slc.start - padding, 0), min(slc.stop + padding, max_stop))
    return new_slice


def pad_slices(list_slices, padding, valid_shape):
    """
    Increase/decrease the list of slices with the padding argument.

    Parameters
    ----------
    list_slices : list
        List of slice objects.
    padding : (int or tuple)
        Padding to be applied on each slice.
    valid_shape : tuple
        The shape of the array which the slices should be valid on.

    Returns
    -------
    list_slices : TYPE
        The list of slices after applying padding.
    """
    # Check the inputs
    if isinstance(padding, int):
        padding = [padding] * len(list_slices)
    if isinstance(valid_shape, int):
        valid_shape = [valid_shape] * len(list_slices)
    if isinstance(padding, (list, tuple)) and len(padding) != len(valid_shape):
        raise ValueError(
            "Invalid padding. The length of padding should be the same as the length of valid_shape."
        )
    # Apply padding
    list_slices = [
        pad_slice(s, padding=p, min_start=0, max_stop=valid_shape[i])
        for i, (s, p) in enumerate(zip(list_slices, padding))
    ]
    return list_slices


# min_size = 10
# min_start = 0
# max_stop = 20
# slc = slice(1, 5)   # left bound
# slc = slice(15, 20) # right bound
# slc = slice(8, 12) # middle


def enlarge_slice(slc, min_size, min_start=0, max_stop=np.inf):
    """
    Enlarge a slice object to have at least a size of min_size.

    The function enforces the left and right bounds of the slice by max_stop and min_start.
    If the original slice size is larger than min_size, the original slice will be returned.

    Parameters
    ----------
    slc : slice
        The original slice object to be enlarged.
    min_size : min_size
        The desired minimum size of the new slice.
    min_start : int, optional
       The minimum value for the start of the new slice.
       The default is 0.
    max_stop : int
        The maximum value for the stop of the new slice.
        The default is np.inf.
    Returns
    -------
    slice
        The new slice object with a size of at least min_size and respecting the left and right bounds.

    """
    # Get slice size
    slice_size = get_slice_size(slc)

    # If min_size is larger than allowable size, raise error
    if min_size > (max_stop - min_start):
        raise ValueError(
            f"'min_size' {min_size} is too large to generate a slice between {min_start} and {max_stop}."
        )

    # If slice size larger than min_size, return the slice
    if slice_size >= min_size:
        return slc

    # Calculate the number of points to add on both sides
    n_indices_to_add = min_size - slice_size
    add_to_left = add_to_right = n_indices_to_add // 2

    # If n_indices_to_add is odd, add + 1 on the left
    if n_indices_to_add % 2 == 1:
        add_to_left += 1

    # Adjust adding for left and right bounds
    naive_start = slc.start - add_to_left
    naive_stop = slc.stop + add_to_right
    if naive_start <= min_start:
        exceeding_left_size = min_start - naive_start
        add_to_right += exceeding_left_size
        add_to_left -= exceeding_left_size
    if naive_stop >= max_stop:
        exceeding_right_size = naive_stop - max_stop
        add_to_right -= exceeding_right_size
        add_to_left += exceeding_right_size

    # Define new slice
    start = slc.start - add_to_left
    stop = slc.stop + add_to_right
    new_slice = slice(start, stop)

    # Check
    assert get_slice_size(new_slice) == min_size

    # Return new slice
    return new_slice


def enlarge_slices(list_slices, min_size, valid_shape):
    """
    Enlarge a list of slice object to have at least a size of min_size.

    The function enforces the left and right bounds of the slice to be between 0 and valid_shape.
    If the original slice size is larger than min_size, the original slice will be returned.

    Parameters
    ----------
    list_slices : list
        List of slice objects.
    min_size : (int or tuple)
        Minimum size of the output slice.
    valid_shape : tuple
        The shape of the array which the slices should be valid on.

    Returns
    -------
    list_slices : list
        The list of slices after enlarging it (if necessary).
    """
    # Check the inputs
    if isinstance(min_size, int):
        min_size = [min_size] * len(list_slices)
    if isinstance(valid_shape, int):
        valid_shape = [valid_shape] * len(list_slices)
    if isinstance(min_size, (list, tuple)) and len(min_size) != len(min_size):
        raise ValueError(
            "Invalid min_size. The length of min_size should be the same as the length of valid_shape."
        )
    # Enlarge the slice
    list_slices = [
        enlarge_slice(slc, min_size=s, min_start=0, max_stop=valid_shape[i])
        for i, (slc, s) in enumerate(zip(list_slices, min_size))
    ]
    return list_slices


###----------------------------------------------------------------------------.
#### Tools for slice creation


def get_slice_around_index(index, size, min_start=0, max_stop=np.inf):
    """
    Get a slice object of `size` around `index` value.

    If size is larger than (max_stop-min_start), raise an error.

    Parameters
    ----------
    index : int
        The index value around which to retrieve the slice.
    min_size : min_size
        The desired size of the slice around the index.
    min_start : int, optional
       The default is np.inf.
       The minimum value for the start of the new slice.
       The default is 0.
    max_stop : int
        The maximum value for the stop of the new slice.

    Returns
    -------
    slice
        The new slice object with a size of at least min_size and respecting the left and right bounds.

    """

    index_slc = slice(index, index + 1)
    slc = enlarge_slice(index_slc, min_size=size, min_start=min_start, max_stop=max_stop)
    if slc == index_slc:
        raise ValueError("'size' {size} is to large to be between {min_start} and {max_stop}.")
    return slc


###----------------------------------------------------------------------------.
#### Tools for slice tiling/sliding
# TODO: implement sliding


def _check_buffer(buffer, slice_size):
    if buffer < 0:
        if abs(buffer) >= int(slice_size / 2):
            raise ValueError(
                "The negative buffer absolute value is larger than half of the slice_size."
            )
    return buffer


def _check_slice_size(slice_size):
    if slice_size <= 0:
        raise ValueError("slice_size must be a positive non-zero integer.")
    return slice_size


def _check_method(method):
    if not isinstance(method, str):
        raise TypeError("'method' must be a string.")
    valid_methods = ["sliding", "tiling"]
    if method not in valid_methods:
        raise ValueError(f"The only valid 'method' are {valid_methods}.")
    return method


def _check_min_start(min_start, start):
    if min_start is None:
        min_start = start
    if min_start > start:
        raise ValueError("'min_start' can not be larger than 'start'.")
    return min_start


def _check_max_stop(max_stop, stop):
    if max_stop is None:
        max_stop = stop
    if max_stop < stop:
        raise ValueError("'max_stop' can not be smaller than 'stop'.")
    return max_stop


def _check_stride(stride, method):
    # TODO: check is an integer !
    if method == "sliding":
        if stride is None:
            stride = 1
        if stride < 1:
            raise ValueError("When sliding, 'stride' must be equal or larger than 1.")
    else:  # tiling
        if stride is None:
            stride = 0
    return stride


def _get_partitioning_idxs(start, stop, stride, slice_size, method):
    if method == "tiling":
        steps = slice_size + stride
    else:  # sliding
        steps = stride
    idxs = np.arange(start, stop + 1, steps)
    return idxs


def get_partitions_slices(
    start,
    stop,
    slice_size,
    method,
    stride=None,
    buffer=0,
    include_last=True,
    ensure_slice_size=False,
    min_start=None,
    max_stop=None,
):
    """
    Create 1D partitioning list of slices.

    Parameters
    ----------
    start : int
        Slice start.
    stop : int
        slice stop.
    slice_size : int
        Slice size.
    method : str
        Whether to retrieve 'tiling' or 'sliding' slices.
        If 'tiling', start slices are separated by stride+slice_size
        If 'sliding', start slices are separated by stride.
    stride : int, optional
        Step size between slices.
        When 'tiling', the default is 0
        When 'sliding', the default is 1.
        When 'tiling', a positive stride make slices to not overlap and not touch,
        while a negative stride make slices to overlap by 'stride' amount. If stride is 0,
        the slices are contiguous (touch).
        When 'sliding', only a positive stride (>= 1) is allowed.
    buffer:
        The default is 0.
        Value by which to enlarge a slice on each side.
        If stride=0 and buffer is positive, it corresponds to
        the amount of overlap between each tile.
        The final slice size should be slice_size + buffer.
        Depending on min_start and max_stop values, buffering might cause
        border slices to not have same sizes.
    include_last : bool, optional
        Whether to include the last slice if not match slice_size.
        The default is True.
    ensure_slice_size : False, optional
        Used only if include_last is True.
        If False, the last slice does not have size 'slice_size'.
        If True,  the last slice is enlarged to have 'slice_size', by
        tentatively expanded the slice on both sides (accounting for min_start and max_stop).
    min_start: int, optional
        The minimum value that the slices start value can have (after i.e. buffering).
        If None (the default), assumed to be equal to start.
    max_stop: int, optional
        Maximum value that the slices stop value can have (after i.e. buffering).
        If None (the default), assumed to be equal to stop.

    Returns
    -------
    slices : list
        List of slices.

    """
    # Check arguments
    slice_size = _check_slice_size(slice_size)
    method = _check_method(method)
    stride = _check_stride(stride, method)
    buffer = _check_buffer(buffer, slice_size)
    min_start = _check_min_start(min_start, start)
    max_stop = _check_max_stop(max_stop, stop)

    # Define slices
    slice_step = 1  # TODO: modify for dilation together with slice_size
    idxs = _get_partitioning_idxs(
        start=start, stop=stop, stride=stride, slice_size=slice_size, method=method
    )
    slices = [slice(idxs[i], idxs[i + 1], slice_step) for i in range(len(idxs) - 1)]

    # Define last slice
    if include_last and idxs[-1] != stop:
        last_slice = slice(idxs[-1], stop)
        if ensure_slice_size:
            last_slice = enlarge_slice(
                last_slice, min_size=slice_size, min_start=min_start, max_stop=max_stop
            )
        slices.append(last_slice)

    # Buffer the slices
    slices = [
        pad_slice(slc, padding=buffer, min_start=min_start, max_stop=max_stop) for slc in slices
    ]

    return slices


def get_nd_partitions_list_slices(
    list_slices, arr_shape, method, kernel_size, stride, buffer, include_last, ensure_slice_size
):
    """Return the n-dimensional partitions list of slices of a initial list of slices."""
    import itertools

    l_iterables = []
    for i in range(len(list_slices)):
        slice_size = kernel_size[i]
        max_stop = arr_shape[i]
        slc = list_slices[i]
        start = slc.start
        stop = slc.stop
        slices = get_partitions_slices(
            start=start,
            stop=stop,
            slice_size=slice_size,
            method=method,
            stride=stride[i],
            buffer=buffer[i],
            include_last=include_last,
            ensure_slice_size=ensure_slice_size,
            min_start=0,
            max_stop=max_stop,
        )
        l_iterables.append(slices)

    tiles_list_slices = list(itertools.product(*l_iterables))
    return tiles_list_slices
