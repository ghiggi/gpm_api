#!/usr/bin/env python3
"""
Created on Wed Oct 19 19:40:12 2022

@author: ghiggi
"""
import random

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from gpm_api.patch.labels import highlight_label, label_xarray_object
from gpm_api.utils.slices import (
    enlarge_slices,
    get_nd_partitions_list_slices,
    get_slice_around_index,
    get_slice_from_idx_bounds,
    pad_slices,
)

#### Note
# - labels_patch_generator available as ds.gpm_api.labels_patch_generator
# - get_patches_from_labels used in gpm_api/visualization.labels

# gpm_api accessor
# - gpm_api.label_object
# - gpm_api.labels_patch_generator

# -----------------------------------------------------------------------------.
#### TODOs
# - Option to bound min_start and max_stop to labels bbox
# - Improve partitions_list_slices
#   --> Change start and stop (if allowed by min_start and max_stop)
#       to be divisible by patch_size + stride ...
# - Add option that returns a flag if the point center is the actual identified one,
#   or was close to the boundary !

# - Check case where tiling kernel_size larger or equal to label_bbox
# - Option: partition_only_on_label_bbox_patch_size_exceedance

# -----------------------------------------------------------------------------.
# - Implement dilate option (to subset pixel within partitions).
#   --> slice(start, stop, step=dilate) ... with patch_size redefined at start to patch_size*dilate
#   --> Need updates of enlarge slcies, pad_slices utilities (but first test current usage !)

# -----------------------------------------------------------------------------.

## Image sliding/tiling reconstruction
# - get_index_overlapping_slices
# - trim: bool, keyword only
#   Whether or not to trim stride elements from each block after calling the map function.
#   Set this to False if your mapping function already does this for you.
#   This for when merging !

####--------------------------------------------------------------------------.


def are_all_integers(arr, negative_allowed=True):
    """
    Check if all values in the input numpy array are integers.

    Parameters
    ----------
    arr : (list, tuple, np.ndarray)
       List, tuple or array of values to be checked.
    negative_allowed: bool, optional
        If False, return True only for integers >=1 (natural numbers)

    Returns
    -------
    bool
        True if all values in the array are integers, False otherwise.

    """
    is_integer = np.isclose(arr, np.round(arr), atol=1e-12, rtol=1e-12)
    if negative_allowed:
        return bool(np.all(is_integer))
    else:
        return bool(np.all(np.logical_and(np.greater(arr, 0), is_integer)))


def are_all_natural_numbers(arr):
    """
    Check if all values in the input numpy array are natural numbers (>1).

    Parameters
    ----------
    arr : (list, tuple, np.ndarray)
       List, tuple or array of values to be checked.

    Returns
    -------
    bool
        True if all values in the array are natural numbers. False otherwise.

    """
    return are_all_integers(arr, negative_allowed=False)


def _ensure_is_dict_argument(arg, dims, arg_name):
    """Ensure argument is a dictionary with same order as dims."""
    if isinstance(arg, (int, float)):
        arg = {dim: arg for dim in dims}
    if isinstance(arg, (list, tuple)):
        if len(arg) != len(dims):
            raise ValueError(f"{arg_name} must match the number of dimensions of the label array.")
        arg = dict(zip(dims, arg))
    if isinstance(arg, dict):
        dict_dims = np.array(list(arg))
        unvalid_dims = dict_dims[np.isin(dict_dims, dims, invert=True)].tolist()
        if len(unvalid_dims) > 0:
            raise ValueError(
                f"{arg_name} must not contain dimensions {unvalid_dims}. It expects only {dims}."
            )
        missing_dims = np.array(dims)[np.isin(dims, dict_dims, invert=True)].tolist()
        if len(missing_dims) > 0:
            raise ValueError(f"{arg_name} must contain also dimensions {missing_dims}")
    else:
        type_str = type(arg)
        raise TypeError(f"Unrecognized type {type_str} for argument {arg_name}.")
    # Reorder as function of dims
    arg = {dim: arg[dim] for dim in dims}
    return arg


def _replace_full_dimension_flag_value(arg, shape):
    """Replace -1 values with the corresponding dimension shape."""
    arg = {dim: shape[i] if value == -1 else value for i, (dim, value) in enumerate(arg.items())}
    return arg


def check_label_arr(label_arr):
    """Check label_arr."""
    # Note: If label array is all zero or nan, labels_id will be []

    # Put label array in memory
    label_arr = np.asanyarray(label_arr)

    # Set 0 label to nan
    label_arr = label_arr.astype(float)  # otherwise if int throw an error when assigning nan
    label_arr[label_arr == 0] = np.nan

    # Check labels_id are natural number >= 1
    valid_labels = np.unique(label_arr[~np.isnan(label_arr)])
    if not are_all_natural_numbers(valid_labels):
        raise ValueError("The label array contains non positive natural numbers.")

    return label_arr


def check_labels_id(labels_id, label_arr):
    """Check labels_id."""
    # Check labels_id type
    if not isinstance(labels_id, (type(None), int, list, np.ndarray)):
        raise TypeError("labels_id must be None or a list or a np.array.")
    if isinstance(labels_id, int):
        labels_id = [labels_id]
    # Get list of valid labels
    valid_labels = np.unique(label_arr[~np.isnan(label_arr)])
    # If labels_id is None, assign the valid_labels
    if isinstance(labels_id, type(None)):
        labels_id = valid_labels
        return labels_id
    # If input labels_id is a list, make it a np.array
    labels_id = np.array(labels_id).astype(int)
    # Check labels_id are natural number >= 1
    if np.any(labels_id == 0):
        raise ValueError("labels id must not contain the 0 value.")
    if not are_all_natural_numbers(labels_id):
        raise ValueError("labels id must be positive natural numbers.")
    # Check labels_id are number present in the label_arr
    unvalid_labels = labels_id[~np.isin(labels_id, valid_labels)]
    if unvalid_labels.size != 0:
        unvalid_labels = unvalid_labels.astype(int)
        raise ValueError(f"The following labels id are not valid: {unvalid_labels}")
    # If no labels, no patch to extract
    n_labels = len(labels_id)
    if n_labels == 0:
        raise ValueError("No labels available.")
    return labels_id


def check_patch_size(patch_size, dims, shape):
    """
    Check the validity of the patch_size argument based on the array shape.

    Parameters
    ----------
    patch_size : (int, list, tuple, dict)
        The size of the patch to extract from the array.
        If int or float, the patch is a hypercube of size patch_size across all dimensions.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, it must have has keys all array dimensions.
        The value -1 can be used to specify the full array dimension shape.
        Otherwise, only positive integers values (>1) are accepted.
    dims : tuple
        The names of the array dimensions.
    shape : tuple
        The shape of the array.

    Returns
    -------
    patch_size : dict
        The shape of the patch.
    """
    patch_size = _ensure_is_dict_argument(patch_size, dims=dims, arg_name="patch_size")
    patch_size = _replace_full_dimension_flag_value(patch_size, shape)
    # Check natural number
    for dim, value in patch_size.items():
        if not are_all_natural_numbers(value):
            raise ValueError(
                "Invalid 'patch_size' values. They must be only positive integer values."
            )
    # Check patch size is smaller than array shape
    idx_valid = [value <= max_value for value, max_value in zip(patch_size.values(), shape)]
    max_allowed_patch_size = {dim: value for dim, value in zip(dims, shape)}
    if not all(idx_valid):
        raise ValueError(f"The maximum allowed patch_size values are {max_allowed_patch_size}")
    return patch_size


def check_kernel_size(kernel_size, dims, shape):
    """
    Check the validity of the kernel_size argument based on the array shape.

    Parameters
    ----------
    kernel_size : (int, list, tuple, dict)
        The size of the kernel to extract from the array.
        If int or float, the kernel is a hypercube of size patch_size across all dimensions.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, it must have has keys all array dimensions.
        The value -1 can be used to specify the full array dimension shape.
        Otherwise, only positive integers values (>1) are accepted.
    dims : tuple
        The names of the array dimensions.
    shape : tuple
        The shape of the array.

    Returns
    -------
    kernel_size : dict
        The shape of the kernel.
    """
    kernel_size = _ensure_is_dict_argument(kernel_size, dims=dims, arg_name="kernel_size")
    kernel_size = _replace_full_dimension_flag_value(kernel_size, shape)
    # Check natural number
    for dim, value in kernel_size.items():
        if not are_all_natural_numbers(value):
            raise ValueError(
                "Invalid 'kernel_size' values. They must be only positive integer values."
            )
    # Check patch size is smaller than array shape
    idx_valid = [value <= max_value for value, max_value in zip(kernel_size.values(), shape)]
    max_allowed_kernel_size = {dim: value for dim, value in zip(dims, shape)}
    if not all(idx_valid):
        raise ValueError(f"The maximum allowed patch_size values are {max_allowed_kernel_size}")
    return kernel_size


def check_buffer(buffer, dims, shape):
    """
    Check the validity of the buffer argument based on the array shape.

    Parameters
    ----------
    buffer : (int, float, list, tuple or dict)
        The size of the buffer to apply to the array.
        If int or float, equal buffer is set on each dimension of the array.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, it must have has keys all array dimensions.
    dims : tuple
        The names of the array dimensions.
    shape : tuple
        The shape of the array.

    Returns
    -------
    buffer : dict
        The buffer to apply on each dimension.
    """
    buffer = _ensure_is_dict_argument(buffer, dims=dims, arg_name="buffer")
    for dim, value in buffer.items():
        if not are_all_integers(value):
            raise ValueError("Invalid 'buffer' values. They must be only integer values.")
    return buffer


def check_padding(padding, dims, shape):
    """
    Check the validity of the padding argument based on the array shape.

    Parameters
    ----------
    padding : (int, float, list, tuple or dict)
        The size of the padding to apply to the array.
        If None, zero padding is assumed.
        If int or float, equal padding is set on each dimension of the array.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, it must have has keys all array dimensions.
    dims : tuple
        The names of the array dimensions.
    shape : tuple
        The shape of the array.

    Returns
    -------
    padding : dict
        The padding to apply on each dimension.
    """
    padding = _ensure_is_dict_argument(padding, dims=dims, arg_name="padding")
    for dim, value in padding.items():
        if not are_all_integers(value):
            raise ValueError("Invalid 'padding' values. They must be only integer values.")
    return padding


def _check_n_patches_per_partition(n_patches_per_partition, centered_on):
    """
    Check the number of patches to extract from each partition.

    It is used only if centered_on is a callable or 'random'

    Parameters
    ----------
    n_patches_per_partition : int
        Number of patches to extract from each partition.
    centered_on : (str, callable)
        Method to extract the patch around a label point.

    Returns
    -------
    n_patches_per_partition: int
       The number of patches to extract from each partition.
    """
    if n_patches_per_partition < 1:
        raise ValueError("n_patches_per_partitions must be a positive integer.")
    if isinstance(centered_on, str):
        if centered_on not in ["random"]:
            if n_patches_per_partition > 1:
                raise ValueError(
                    "Only the pre-implemented centered_on='random' method allow n_patches_per_partition values > 1."
                )
    return n_patches_per_partition


def _check_n_patches_per_label(n_patches_per_label, n_patches_per_partition):
    if n_patches_per_label < n_patches_per_partition:
        raise ValueError("n_patches_per_label must be equal or larger to n_patches_per_partition.")
    return n_patches_per_label


def check_partitioning_method(partitioning_method):
    """Check partitioning method."""
    if not isinstance(partitioning_method, (str, type(None))):
        raise TypeError("'partitioning_method' must be either a string or None.")
    if isinstance(partitioning_method, str):
        valid_methods = ["sliding", "tiling"]
        if partitioning_method not in valid_methods:
            raise ValueError(f"Valid 'partitioning_method' are {valid_methods}")
    return partitioning_method


def check_stride(stride, dims, shape, partitioning_method):
    """
    Check the validity of the stride argument based on the array shape.

    Parameters
    ----------
    stride : (None, int, float, list, tuple, dict)
        The size of the stride to apply to the array.
        If None, no striding is assumed.
        If int or float, equal stride is set on each dimension of the array.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, it must have has keys all array dimensions.
    dims : tuple
        The names of the array dimensions.
    shape : tuple
        The shape of the array.
    partitioning_method: (None, str)
        The optional partitioning method (tiling or sliding) to use.

    Returns
    -------
    stride : dict
        The stride to apply on each dimension.
    """
    if partitioning_method is None:
        return None
    # Set default arguments
    if stride is None:
        if partitioning_method == "tiling":
            stride = 0
        else:  # sliding
            stride = 1
    stride = _ensure_is_dict_argument(stride, dims=dims, arg_name="stride")
    if partitioning_method == "tiling":
        for dim, value in stride.items():
            if not are_all_integers(value):
                raise ValueError("Invalid 'stride' values. They must be only integer values.")
    else:  # sliding
        for dim, value in stride.items():
            if not are_all_natural_numbers(value):
                raise ValueError(
                    "Invalid 'stride' values. They must be only positive integer (>=1) values."
                )
    return stride


def _check_callable_centered_on(centered_on):
    """Check validity of callable centered_on."""
    input_shape = (2, 3)
    arr = np.zeros(input_shape)
    point = centered_on(arr)
    if not isinstance(point, (tuple, type(None))):
        raise ValueError(
            "The 'centered_on' function should return a point coordinates tuple or None."
        )
    if len(point) != len(input_shape):
        raise ValueError(
            "The 'centered_on' function should return point coordinates having same dimensions has input array."
        )
    for c, max_value in zip(point, input_shape):
        if c < 0:
            raise ValueError("The point coordinate must be a positive integer.")
        if c >= max_value:
            raise ValueError("The point coordinate must be inside the array shape.")
        if np.isnan(c):
            raise ValueError("The point coordinate must not be np.nan.")
    try:
        point = centered_on(arr * np.nan)
        if point is not None:
            raise ValueError(
                "The 'centered_on' function should return None if the input array is a np.nan ndarray."
            )
    except:
        raise ValueError("The 'centered_on' function should be able to deal with a np.nan ndarray.")


def check_centered_on(centered_on):
    """Check valid centered_on to identify a point in an array."""
    if not (callable(centered_on) or isinstance(centered_on, str)):
        raise TypeError("'centered_on' must be a string or a function.")
    if isinstance(centered_on, str):
        valid_centered_on = [
            "max",
            "min",
            "centroid",
            "center_of_mass",
            "random",
            "label_bbox",  # unfixed patch_size
        ]
        if centered_on not in valid_centered_on:
            raise ValueError(f"Valid 'centered_on' values are: {valid_centered_on}.")

    if callable(centered_on):
        _check_callable_centered_on(centered_on)
    return centered_on


def _get_variable_arr(xr_obj, variable, centered_on):
    if isinstance(xr_obj, xr.DataArray):
        variable_arr = xr_obj.data
        return variable_arr
    else:
        if centered_on is not None:
            if variable is None and (centered_on in ["max", "min"] or callable(centered_on)):
                raise ValueError("'variable' must be specified if 'centered_on' is specified.")
        if variable is not None:
            variable_arr = xr_obj[variable].data
        else:
            variable_arr = None
    return variable_arr


def _check_variable_arr(variable_arr, label_arr):
    """Check variable array validity."""
    if variable_arr is not None:
        if variable_arr.shape != label_arr.shape:
            raise ValueError(
                "Arrays corresponding to 'variable' and 'label_name' must have same shape."
            )
    return variable_arr


####--------------------------------------------------------------------------.


def _get_point_centroid(arr):
    """Get the coordinate of label bounding box center.

    It assumes that the array has been cropped around the label.
    It returns None if all values are non-finite (i.e. np.nan).
    """
    if np.all(~np.isfinite(arr)):
        return None
    centroid = np.array(arr.shape) / 2.0
    centroid = tuple(centroid.tolist())
    return centroid


def get_point_random(arr):
    """Get random point with finite value."""
    is_finite = np.isfinite(arr)
    if np.all(~is_finite):
        return None
    points = np.argwhere(is_finite)
    random_point = random.choice(points)
    return random_point


def get_point_with_max_value(arr):
    """Get point with maximum value."""
    point = np.argwhere(arr == np.nanmax(arr))
    if len(point) == 0:
        point = None
    else:
        point = tuple(point[0].tolist())
    return point


def get_point_with_min_value(arr):
    """Get point with minimum value."""
    point = np.argwhere(arr == np.nanmin(arr))
    if len(point) == 0:
        point = None
    else:
        point = tuple(point[0].tolist())
    return point


def get_point_center_of_mass(arr, integer_index=True):
    """Get the coordinate of the label center of mass.

    It uses all cells which have finite values.
    If 0 value should be a non-label area, mask before with np.nan.
    It returns None if all values are non-finite (i.e. np.nan).
    """
    indices = np.argwhere(np.isfinite(arr))
    if len(indices) == 0:
        return None
    center_of_mass = np.nanmean(indices, axis=0)
    if integer_index:
        center_of_mass = center_of_mass.astype(int)
    center_of_mass = tuple(center_of_mass.tolist())
    return center_of_mass


def _find_point(arr, centered_on="max"):
    """Find a specific point coordinate of the array.

    If the coordinate can't be find, return None.
    """
    if centered_on == "max":
        point = get_point_with_max_value(arr)
    elif centered_on == "min":
        point = get_point_with_min_value(arr)
    elif centered_on == "centroid":
        point = _get_point_centroid(arr)
    elif centered_on == "center_of_mass":
        point = get_point_center_of_mass(arr)
    elif centered_on == "random":
        point = get_point_random(arr)
    else:  # callable centered_on
        point = centered_on(arr)
    return point


def _get_labels_bbox_slices(arr):
    """
    Compute the bounding box slices of non-zero elements in a n-dimensional numpy array.

    Assume that only one unique non-zero elements values is present in the array.
    Assume that NaN and Inf have been replaced by zeros.

    Other implementations: scipy.ndimage.find_objects

    Parameters
    ----------
    arr : np.ndarray
        n-dimensional numpy array.

    Returns
    -------
    list_slices : list
        List of slices to extract the region with non-zero elements in the input array.
    """
    ndims = arr.ndim
    coords = np.nonzero(arr)
    list_slices = [
        get_slice_from_idx_bounds(np.min(coords[i]), np.max(coords[i])) for i in range(ndims)
    ]
    return list_slices


def _get_patch_list_slices_around_label_point(
    label_arr,
    label_id,
    variable_arr,
    patch_size,
    centered_on,
):
    """Get list_slices to extract patch around a label point.

    Assume label_arr must match variable_arr shape.
    Assume patch_size shape must match variable_arr shape .
    """
    is_label = label_arr == label_id
    if not np.any(is_label):
        return None
    # Subset variable_arr around label
    list_slices = _get_labels_bbox_slices(is_label)
    label_subset_arr = label_arr[tuple(list_slices)]
    variable_subset_arr = variable_arr[tuple(list_slices)]
    variable_subset_arr = np.asarray(variable_subset_arr)  # if dask, make numpy
    # Mask variable arr outside the label
    variable_subset_arr[label_subset_arr != label_id] = np.nan
    # Find point of subset array
    point_subset_arr = _find_point(arr=variable_subset_arr, centered_on=centered_on)
    # Define patch list_slices
    if point_subset_arr is not None:
        # Find point in original array
        point = [slc.start + c for slc, c in zip(list_slices, point_subset_arr)]
        # Find patch list slices
        patch_list_slices = [
            get_slice_around_index(p, size=size, min_start=0, max_stop=shape)
            for p, size, shape in zip(point, patch_size, variable_arr.shape)
        ]
        # TODO: also return a flag if the p midpoint is conserved (by +/- 1) or not
    else:
        patch_list_slices = None
    return patch_list_slices


def _get_patch_list_slices_around_label(label_arr, label_id, padding, min_patch_size):
    """Get list_slices to extract patch around a label."""
    # Get label bounding box slices
    list_slices = _get_labels_bbox_slices(label_arr == label_id)
    # Apply padding to the slices
    list_slices = pad_slices(list_slices, padding=padding, valid_shape=label_arr.shape)
    # Increase slices to match min_patch_size
    list_slices = enlarge_slices(list_slices, min_size=min_patch_size, valid_shape=label_arr.shape)
    return list_slices


def get_patch_list_slices(label_arr, label_id, variable_arr, patch_size, centered_on, padding):
    """Get patch n-dimensional list slices."""
    if not callable(centered_on) and centered_on == "label_bbox":
        list_slices = _get_patch_list_slices_around_label(
            label_arr=label_arr, label_id=label_id, padding=padding, min_patch_size=patch_size
        )
    else:
        list_slices = _get_patch_list_slices_around_label_point(
            label_arr=label_arr,
            label_id=label_id,
            variable_arr=variable_arr,
            patch_size=patch_size,
            centered_on=centered_on,
        )
    return list_slices


def _get_masked_arrays(label_arr, variable_arr, partition_list_slices):
    """Mask labels and variable arrays outside the partitions area."""
    masked_partition_label_arr = np.zeros(label_arr.shape) * np.nan
    masked_partition_label_arr[tuple(partition_list_slices)] = label_arr[
        tuple(partition_list_slices)
    ]
    if variable_arr is not None:
        masked_partition_variable_arr = np.zeros(variable_arr.shape) * np.nan
        masked_partition_variable_arr[tuple(partition_list_slices)] = variable_arr[
            tuple(partition_list_slices)
        ]
    return masked_partition_label_arr, masked_partition_variable_arr


def get_patches_from_partitions_list_slices(
    partitions_list_slices,
    label_arr,
    variable_arr,
    label_id,
    patch_size,
    centered_on,
    n_patches_per_partition,
    padding,
):
    """Return patches list slices from list of partitions list_slices.

    n_patches_per_partition is 1 unless centered_on is 'random' or a callable.
    """
    patches_list_slices = []
    for partition_list_slices in partitions_list_slices:
        masked_label_arr, masked_variable_arr = _get_masked_arrays(
            label_arr=label_arr,
            variable_arr=variable_arr,
            partition_list_slices=partition_list_slices,
        )
        n = 0
        while n < n_patches_per_partition:
            patch_list_slices = get_patch_list_slices(
                label_arr=masked_label_arr,
                variable_arr=masked_variable_arr,
                label_id=label_id,
                patch_size=patch_size,
                centered_on=centered_on,
                padding=padding,
            )
            if patch_list_slices is not None:
                n += 1
                patches_list_slices.append(patch_list_slices)
    return patches_list_slices


def get_list_isel_dicts(patches_list_slices, dims):
    """Return a list with isel dictionaries."""
    list_isel_dicts = []
    for patch_list_slices in patches_list_slices:
        # list_isel_dicts.append(dict(zip(dims, patch_list_slices)))
        list_isel_dicts.append({dim: slc for dim, slc in zip(dims, patch_list_slices)})
    return list_isel_dicts


def _extract_xr_patch(xr_obj, isel_dict, label_name, label_id, highlight_label_id):
    """Extract a xarray patch."""
    # Extract xarray patch around label
    xr_obj_patch = xr_obj.isel(isel_dict)

    # If asked, set label array to 0 except for label_id
    if highlight_label_id:
        xr_obj_patch = highlight_label(xr_obj_patch, label_name=label_name, label_id=label_id)
    return xr_obj_patch


def plot_rectangle_from_list_slices(ax, list_slices, edgecolor="red", facecolor="None", **kwargs):
    """Plot rectangles from 2D patch list slices."""
    if len(list_slices) != 2:
        raise ValueError("Required 2 slices.")
    # Extract the start and stop values from the slice
    y_start, y_stop = (list_slices[0].start, list_slices[0].stop)
    x_start, x_stop = (list_slices[1].start, list_slices[1].stop)
    # Calculate the width and height of the rectangle
    width = x_stop - x_start
    height = y_stop - y_start
    # Plot rectangle
    rectangle = plt.Rectangle(
        (x_start, y_start), width, height, edgecolor=edgecolor, facecolor=facecolor, **kwargs
    )
    ax.add_patch(rectangle)
    return ax


def plot_2d_label_partitions_boundaries(
    partitions_list_slices, label_arr, edgecolor="red", facecolor="None", **kwargs
):
    """Plot partitions from 2D list slices."""
    # Define plot limits
    xmin = min([patch_list_slices[1].start for patch_list_slices in partitions_list_slices])
    xmax = max([patch_list_slices[1].stop for patch_list_slices in partitions_list_slices])
    ymin = min([patch_list_slices[0].start for patch_list_slices in partitions_list_slices])
    ymax = max([patch_list_slices[0].stop for patch_list_slices in partitions_list_slices])

    # Plot patches boundaries
    fig, ax = plt.subplots()
    ax.imshow(label_arr, origin="upper")
    for partition_list_slices in partitions_list_slices:
        _ = plot_rectangle_from_list_slices(
            ax=ax,
            list_slices=partition_list_slices,
            edgecolor=edgecolor,
            facecolor=facecolor,
            **kwargs,
        )
    # Set plot limits
    ax.set_xlim(xmin - 5, xmax + 5)
    ax.set_ylim(ymax + 5, ymin - 5)
    return fig


def _add_label_patches_boundaries(
    fig, patches_list_slices, edgecolor="red", facecolor="None", **kwargs
):

    # Retrieve axis
    ax = fig.axes[0]

    # Define patches limits
    xmin = min([patch_list_slices[1].start for patch_list_slices in patches_list_slices])
    xmax = max([patch_list_slices[1].stop for patch_list_slices in patches_list_slices])
    ymin = min([patch_list_slices[0].start for patch_list_slices in patches_list_slices])
    ymax = max([patch_list_slices[0].stop for patch_list_slices in patches_list_slices])

    # Get current plot axis limits
    plot_xmin, plot_xmax = ax.get_xlim()
    plot_ymin, plot_ymax = ax.get_ylim()

    # Define final plot axis limits
    xmin = min(xmin, plot_xmin)
    xmax = max(xmax, plot_xmax)
    ymin = min(ymin, plot_ymin)
    ymax = max(ymax, plot_ymax)

    # Plot patch boundaries
    for patch_list_slices in patches_list_slices:
        _ = plot_rectangle_from_list_slices(
            ax=ax, list_slices=patch_list_slices, edgecolor=edgecolor, facecolor=facecolor, **kwargs
        )
    # Set plot limits
    ax.set_xlim(xmin - 5, xmax + 5)
    ax.set_ylim(ymax + 5, ymin - 5)

    return fig


def plot_2d_label_patches_boundaries(patches_list_slices, label_arr):
    """Plot patches from  from 2D list slices."""
    # Define plot limits
    xmin = min([patch_list_slices[1].start for patch_list_slices in patches_list_slices])
    xmax = max([patch_list_slices[1].stop for patch_list_slices in patches_list_slices])
    ymin = min([patch_list_slices[0].start for patch_list_slices in patches_list_slices])
    ymax = max([patch_list_slices[0].stop for patch_list_slices in patches_list_slices])

    # Plot patches boundaries
    fig, ax = plt.subplots()
    ax.imshow(label_arr, origin="upper")
    for patch_list_slices in patches_list_slices:
        plot_rectangle_from_list_slices(ax, patch_list_slices)

    # Set plot limits
    ax.set_xlim(xmin - 5, xmax + 5)
    ax.set_ylim(ymax + 5, ymin - 5)

    # Show plot
    plt.show()

    # Return figure
    return fig


####--------------------------------------------------------------------------.
#### TODO: UPDATE TO USE THIS


def _get_patches_isel_dict_generator(
    xr_obj,
    label_name,
    patch_size,
    variable=None,
    # Output options
    n_patches=np.Inf,
    n_labels=None,
    labels_id=None,
    grouped_by_labels_id=False,
    # (Tile) label patch extraction
    padding=0,
    centered_on="max",
    n_patches_per_label=np.Inf,
    n_patches_per_partition=1,
    debug=False,
    # Label Tiling/Sliding Options
    partitioning_method=None,
    n_partitions_per_label=None,
    kernel_size=None,
    buffer=0,
    stride=None,
    include_last=True,
    ensure_slice_size=True,
):
    # Get label array information
    label_arr = xr_obj[label_name].data
    dims = xr_obj[label_name].dims
    shape = label_arr.shape

    # Check input arguments
    if n_labels is not None and labels_id is not None:
        raise ValueError("Specify either n_labels or labels_id.")
    if kernel_size is None:
        kernel_size = patch_size

    patch_size = check_patch_size(patch_size, dims, shape)
    buffer = check_buffer(buffer, dims, shape)
    padding = check_padding(padding, dims, shape)

    partitioning_method = check_partitioning_method(partitioning_method)
    stride = check_stride(stride, dims, shape, partitioning_method)
    kernel_size = check_kernel_size(kernel_size, dims, shape)

    centered_on = check_centered_on(centered_on)
    n_patches_per_partition = _check_n_patches_per_partition(n_patches_per_partition, centered_on)
    n_patches_per_label = _check_n_patches_per_label(n_patches_per_label, n_patches_per_partition)

    label_arr = check_label_arr(label_arr)  # output is np.array !
    labels_id = check_labels_id(labels_id=labels_id, label_arr=label_arr)
    variable_arr = _get_variable_arr(xr_obj, variable, centered_on)  # if required
    variable_arr = _check_variable_arr(variable_arr, label_arr)

    # Define number of labels from which to extract patches
    available_n_labels = len(labels_id)
    n_labels = min(available_n_labels, n_labels) if n_labels else available_n_labels

    # -------------------------------------------------------------------------.
    # Extract patch(es) around the label
    patch_counter = 0
    for label_id in labels_id[0:n_labels]:

        # Subset label_arr around the given label
        label_bbox_slices = _get_labels_bbox_slices(label_arr == label_id)

        # Apply padding to the label bounding box
        label_bbox_slices = pad_slices(
            label_bbox_slices, padding=padding.values(), valid_shape=label_arr.shape
        )

        # --------------------------------------------------------------------.
        # Retrieve partitions list_slices
        if partitioning_method is not None:
            partitions_list_slices = get_nd_partitions_list_slices(
                label_bbox_slices,
                arr_shape=label_arr.shape,
                method=partitioning_method,
                kernel_size=kernel_size,
                stride=stride,
                buffer=buffer,
                include_last=include_last,
                ensure_slice_size=ensure_slice_size,
            )
            if n_partitions_per_label is not None:
                n_to_select = min(len(partitions_list_slices), n_partitions_per_label)
                partitions_list_slices = partitions_list_slices[0:n_to_select]
        else:
            partitions_list_slices = [label_bbox_slices]

        # --------------------------------------------------------------------.
        # If debug=True, plot tile (or label bbox) boundaries
        if debug and label_arr.ndim == 2:
            fig = plot_2d_label_partitions_boundaries(
                partitions_list_slices, label_arr, edgecolor="black"
            )

        # --------------------------------------------------------------------.
        # Retrieve patches list_slices from partitions list slices
        patches_list_slices = get_patches_from_partitions_list_slices(
            partitions_list_slices=partitions_list_slices,
            label_arr=label_arr,
            variable_arr=variable_arr,
            label_id=label_id,
            patch_size=patch_size.values(),
            centered_on=centered_on,
            n_patches_per_partition=n_patches_per_partition,
            padding=padding.values(),
        )

        # If debug=True, plot patches boundaries
        if debug and label_arr.ndim == 2:
            _ = _add_label_patches_boundaries(
                fig=fig, patches_list_slices=patches_list_slices, edgecolor="red"
            )
            plt.show()

        # ---------------------------------------------------------------------.
        # Retrieve patches isel_dictionaries
        patches_isel_dicts = get_list_isel_dicts(patches_list_slices, dims=dims)
        n_to_select = min(len(patches_isel_dicts), n_patches_per_label)
        patches_isel_dicts = patches_isel_dicts[0:n_to_select]

        # ---------------------------------------------------------------------.
        # Return isel_dicts
        if grouped_by_labels_id:
            patch_counter += 1
            if patch_counter > n_patches:
                break
            yield label_id, patches_isel_dicts
        else:
            for isel_dict in patches_isel_dicts:
                patch_counter += 1
                if patch_counter > n_patches:
                    break
                yield label_id, isel_dict

        # ---------------------------------------------------------------------.


def get_patches_isel_dict_from_labels(
    xr_obj,
    label_name,
    patch_size,
    variable=None,
    # Output options
    n_patches=np.Inf,
    n_labels=None,
    labels_id=None,
    # Label Patch Extraction Settings
    centered_on="max",
    padding=0,
    n_patches_per_label=np.Inf,
    n_patches_per_partition=1,
    # Label Tiling/Sliding Options
    partitioning_method=None,
    n_partitions_per_label=None,
    kernel_size=None,
    buffer=0,
    stride=None,
    include_last=True,
    ensure_slice_size=True,
    debug=False,
):
    gen = _get_patches_isel_dict_generator(
        xr_obj=xr_obj,
        label_name=label_name,
        patch_size=patch_size,
        variable=variable,
        n_patches=n_patches,
        n_labels=n_labels,
        labels_id=labels_id,
        grouped_by_labels_id=True,
        # Patch extraction options
        centered_on=centered_on,
        padding=padding,
        n_patches_per_label=n_patches_per_label,
        n_patches_per_partition=n_patches_per_partition,
        # Tiling/Sliding settings
        partitioning_method=partitioning_method,
        n_partitions_per_label=n_partitions_per_label,
        kernel_size=kernel_size,
        buffer=buffer,
        stride=stride,
        include_last=include_last,
        ensure_slice_size=ensure_slice_size,
        debug=debug,
    )
    dict_isel_dicts = {int(label_id): list_isel_dicts for label_id, list_isel_dicts in gen}
    return dict_isel_dicts


def get_patches_from_labels(
    xr_obj,
    label_name,
    patch_size,
    variable=None,
    # Output options
    n_patches=np.Inf,
    n_labels=None,
    labels_id=None,
    highlight_label_id=True,
    # Label Patch Extraction Options
    centered_on="max",
    padding=0,
    n_patches_per_label=np.Inf,
    n_patches_per_partition=1,
    # Label Tiling/Sliding Options
    partitioning_method=None,
    n_partitions_per_label=None,
    kernel_size=None,
    buffer=0,
    stride=None,
    include_last=True,
    ensure_slice_size=True,
    debug=False,
):
    """
    Routines to extract patches around labels.

    Create a generator extracting (from a prelabeled xr.Dataset) a patch around:

    - a label point
    - a label bounding box

    If 'centered_on' is specified, output patches are guaranteed to have equal shape !
    If 'centered_on' is not specified, output patches are guaranteed to have only have a minimum shape !

    If you want to extract the patch around the label bounding box, 'centered_on'
    must not be specified.

    If you want to extract the patch around a label point, the 'centered_on'
    method must be specified. If the identified point is close to an array boundariy,
    the patch is expanded toward the valid directions.

    Tiling or sliding enables to split/slide over each label and extract multiple patch
    for each tile.

    tiling=True
    - centered_on = "centroid" (tiling around labels bbox)
    - centered_on = "center_of_mass" (better coverage around label)

    sliding=True
    - centered_on = "center_of_mass" (better coverage around label) (further data coverage)

    Only one parameter between n_patches and labels_id can be specified.

    Parameters
    ----------
    xr_obj : xr.Dataset
        xr.Dataset with a label array named label_bame.
    label_name : str
        Name of the variable/coordinate representing the label array.
    patch_size : (int, tuple)
        The dimensions of the n-dimensional patch to extract.
        Only positive values (>1) are allowed.
        The value -1 can be used to specify the full array dimension shape.
        If the centered_on method is not 'label_bbox', all output patches
        are ensured to have the ame shape.
        Otherwise, if 'centered_on'='label_bbox', the patch_size argument defines
        defined the minimum n-dimensional shape of the output patches.
        If int, the value is applied to all label array dimensions.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, the dictionary must have has keys the label array dimensions.
    n_patches : int, optional
        Maximum number of patches to extract.
        The default (np.Inf) enable to extract all available patches allowed by the
        specified patch extraction criteria.
    labels_id : list, optional
        List of labels for which to extract the patch.
        If None, it extracts the patches by label order (1, 2, 3, ...)
        The default is None.
    n_labels : int, optional
        The number of labels for which extract patches.
        If None (the default), it extract patches for all labels.
        This argument can be specified only if labels_id is unspecified !
    highlight_label_id : (bool), optional
        If True, the laben_name array of each patch is modified to contain only
        the label_id used to select the patch.
    variable : str, optional
        Dataset variable to use to identify the patch center when centered_on is defined.
        This is required only for centered_on='max', 'min' or the custom function.

    centered_on : (str, callable), optional
        The centered_on method characterize the point around which the patch is extracted.
        Valid pre-implemented centered_on methods are 'label_bbox', 'max', 'min',
        'centroid', 'center_of_mass', 'random'.
        The default method is 'max'.

        If 'label_bbox' it extract the patches around the (padded) bounding box of the label.
        If 'label_bbox',the output patch sizes are only ensured to have a minimum patch_size,
        and will likely be of different size.
        Otherwise, the other methods guarantee that the output patches have a common shape.

        If centered_on is 'max', 'min' or a custom function, the 'variable' must be specified.
        If centered_on is a custom function, it must:
            - return None if all array values are non-finite (i.e np.nan)
            - return a tuple with same length as the array shape.
    padding : (int, tuple, dict), optional
        The padding to apply in each direction around a label prior to
        partitioning (tiling/sliding) or direct patch extraction.
        The default, 0, applies 0 padding in every dimension.
        Negative padding values are allowed !
        If int, the value is applied to all label array dimensions.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, the dictionary must have has keys the label array dimensions.
    n_patches_per_label: int, optional
        The maximum number of patches to extract for each label.
        The default (np.Inf) enables to extract all the available patches per label.
        n_patches_per_label must be larger than n_patches_per_partition !
    n_patches_per_partition, int, optional
        The maximum number of patches to extract from each label partition.
        The default values is 1.
        This method can be specified only if centered_on='random' or a callable.
    method : str
        Whether to retrieve 'tiling' or 'sliding' slices.
        If 'tiling', partition start slices are separated by stride + kernel_size
        If 'sliding', partition start slices are separated by stride.
    stride : (int, tuple, dict), optional
        If partitioning_method is 'sliding'', default stride is set to 1.
        If partitioning_method is 'tiling', default stride is set to 0.
        Step size between slices.
        When 'tiling', a positive stride make partition slices to not overlap and not touch,
        while a negative stride make partition slices to overlap by 'stride' amount.
        If stride is 0, the partition slices are contiguous (no spacing between partitions).
        When 'sliding', only a positive stride (>= 1) is allowed.
        If int, the value is applied to all label array dimensions.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, the dictionary must have has keys the label array dimensions.
    kernel_size: (int, tuple, dict), optional
        The shape of the desired partitions.
        Only positive values (>1) are allowed.
        The value -1 can be used to specify the full array dimension shape.
        If int, the value is applied to all label array dimensions.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, the dictionary must have has keys the label array dimensions.
    buffer: (int, tuple, dict), optional
        The default is 0.
        Value by which to enlarge a partition on each side.
        The final partition size should be kernel_size + buffer.
        If 'tiling' and stride=0, a positive buffer value corresponds to
        the amount of overlap between each partition.
        Depending on min_start and max_stop values, buffering might cause
        border partitions to not have same sizes.
        If int, the value is applied to all label array dimensions.
        If list or tuple, the length must match the number of dimensions of the array.
        If a dict, the dictionary must have has keys the label array dimensions.
    include_last : bool, optional
        Whether to include the last partition if it does not match the kernel_size.
        The default is True.
    ensure_slice_size : False, optional
        Used only if include_last is True.
        If False, the last partition will not have the specified kernel_size.
        If True,  the last partition is enlarged to the specified kernel_size by
        tentatively expandinf it on both sides (accounting for min_start and max_stop).

    Yields
    ------
    (xr.Dataset or xr.DataArray)
        A xarray object patch.

    """
    # Define patches isel dictionary generator
    patches_isel_dicts_gen = _get_patches_isel_dict_generator(
        xr_obj=xr_obj,
        label_name=label_name,
        patch_size=patch_size,
        variable=variable,
        n_patches=n_patches,
        n_labels=n_labels,
        labels_id=labels_id,
        grouped_by_labels_id=False,
        # Label Patch Extraction Options
        centered_on=centered_on,
        padding=padding,
        n_patches_per_label=n_patches_per_label,
        n_patches_per_partition=n_patches_per_partition,
        # Tiling/Sliding Options
        partitioning_method=partitioning_method,
        n_partitions_per_label=n_partitions_per_label,
        kernel_size=kernel_size,
        buffer=buffer,
        stride=stride,
        include_last=include_last,
        ensure_slice_size=ensure_slice_size,
        debug=debug,
    )

    # Extract the patches
    for label_id, isel_dict in patches_isel_dicts_gen:
        xr_obj_patch = _extract_xr_patch(
            xr_obj=xr_obj,
            label_name=label_name,
            isel_dict=isel_dict,
            label_id=label_id,
            highlight_label_id=highlight_label_id,
        )

        # Return the patch around the label
        yield label_id, xr_obj_patch


####--------------------------------------------------------------------------.
#### LABEL + PATCH EXTRACTION WRAPPER


def labels_patch_generator(
    xr_obj,
    patch_size,
    variable=None,
    # Label Options
    min_value_threshold=-np.inf,
    max_value_threshold=np.inf,
    min_area_threshold=1,
    max_area_threshold=np.inf,
    footprint=None,
    sort_by="area",
    sort_decreasing=True,
    # Patch Output options
    n_patches=np.Inf,
    n_labels=None,
    labels_id=None,
    highlight_label_id=True,
    # Label Patch Extraction Options
    centered_on="max",
    padding=0,
    n_patches_per_label=np.Inf,
    n_patches_per_partition=1,
    # Label Tiling/Sliding Options
    partitioning_method=None,
    n_partitions_per_label=None,
    kernel_size=None,
    buffer=0,
    stride=None,
    include_last=True,
    ensure_slice_size=True,
):
    """
    Create a generator extracting patches around sensible regions of an xr.Dataset.

    The function first derives the labels array, and then it extract patches for n_patches labels.

    Parameters
    ----------
    xr_obj : (xr.Dataset or xr.DataArray)
        xarray object.
    variable : str, optional
        Dataset variable to exploit to derive the labels.
        Must be specified only if the input object is an xr.Dataset.
    min_value_threshold : float, optional
        The minimum value to define the interior of a label.
        The default is -np.inf.
    max_value_threshold : float, optional
        The maximum value to define the interior of a label.
        The default is np.inf.
    min_area_threshold : float, optional
        The minimum number of connected pixels to be defined as a label.
        The default is 1.
    max_area_threshold : float, optional
        The maximum number of connected pixels to be defined as a label.
        The default is np.inf.
    footprint : (int, np.ndarray or None), optional
        This argument enables to dilate the mask derived after applying
        min_value_threshold and max_value_threshold.
        If footprint = 0 or None, no dilation occur.
        If footprint is a positive integer, it create a disk(footprint)
        If footprint is a 2D array, it must represent the neighborhood expressed
        as a 2-D array of 1’s and 0’s.
        The default is None (no dilation).
    sort_by : (callable or str), optional
        A function or statistics to define the order of the labels.
        Valid string statistics are "area", "maximum", "minimum", "mean",
        "median", "sum", "standard_deviation", "variance".
        The default is "area".
    sort_decreasing : bool, optional
        If True, sort labels by decreasing 'sort_by' value.
        The default is True.

    TODO: add label patch generator options

    Yields
    ------
    (xr.Dataset or xr.DataArray)
        A xarray object patch.

    """
    # Retrieve labeled xarray object
    xr_obj = label_xarray_object(
        xr_obj,
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
    patch_gen = get_patches_from_labels(
        xr_obj,
        label_name="label",
        patch_size=patch_size,
        variable=variable,
        # Output options
        n_patches=n_patches,
        n_labels=n_labels,
        labels_id=labels_id,
        highlight_label_id=highlight_label_id,
        # Patch extraction Options
        padding=padding,
        centered_on=centered_on,
        n_patches_per_label=n_patches_per_label,
        n_patches_per_partition=n_patches_per_partition,
        # Tiling/Sliding Options
        partitioning_method=partitioning_method,
        n_partitions_per_label=n_partitions_per_label,
        kernel_size=kernel_size,
        buffer=buffer,
        stride=stride,
        include_last=include_last,
        ensure_slice_size=ensure_slice_size,
    )

    return patch_gen
