#!/usr/bin/env python3
"""
Created on Wed Oct 19 19:40:12 2022

@author: ghiggi
"""
import numpy as np

from gpm_api.patch.labels import highlight_label, label_xarray_object
from gpm_api.utils.slices import (
    enlarge_slices,
    get_slice_from_idx_bounds,
    pad_slices,
)

# Note
# - labels_patch_generator available as ds.gpm_api.labels_patch_generator
# - get_labeled_object_patches used in gpm_api/visualization.labels

# gpm_api accessor
# - gpm_api.label_object
# - gpm_api.labels_patch_generator

####--------------------------------------------------------------------------.
#######################################
#### Patch-Around-Label Generator  ####
#######################################


def _is_natural_numbers(arr):
    """
    Check if all values in the input numpy array are natural numbers (positive integers or positive floats)

    Parameters
    ----------
    arr : (list, tuple, np.ndarray)
       List, tuple or array of values to be checked.

    Returns
    -------
    bool
        True if all values in the array are natural numbers, False otherwise..

    """
    return bool(
        np.all(
            np.logical_and(
                np.greater(arr, 0), np.isclose(arr, np.round(arr), atol=1e-12, rtol=1e-12)
            )
        )
    )


def _check_label_arr(label_arr):
    """Check label_arr."""
    # Note: If label array is all zero or nan, labels_id will be []

    # Put label array in memory
    label_arr = np.asanyarray(label_arr)

    # Set 0 label to nan
    label_arr = label_arr.astype(float)  # otherwise if int throw an error when assigning nan
    label_arr[label_arr == 0] = np.nan

    # Check labels_id are natural number >= 1
    valid_labels = np.unique(label_arr[~np.isnan(label_arr)])
    if not _is_natural_numbers(valid_labels):
        raise ValueError("The label array contains non positive natural numbers.")

    return label_arr


def _check_labels_id(labels_id, label_arr):
    """Check labels_id."""
    # Check labels_id type
    if not isinstance(labels_id, (type(None), list, np.ndarray)):
        raise TypeError("labels_id must be None or a list or a np.array.")
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
    if not _is_natural_numbers(labels_id):
        raise ValueError("labels id must be positive natural numbers.")
    # Check labels_id are number present in the label_arr
    unvalid_labels = labels_id[~np.isin(labels_id, valid_labels)]
    if unvalid_labels.size != 0:
        unvalid_labels = unvalid_labels.astype(int)
        raise ValueError(f"The following labels id are not valid: {unvalid_labels}")
    return labels_id


def _check_patch_size(patch_size, array_shape):
    """
    Check the validity of the patch_size argument based on the array_shape.

    Parameters
    ----------
    patch_size : (None, int, float, list or tuple)
        The size of the patch to extract from the array.
        If None, it set a patch size of 2 in all directions.
        If int or float, the patch is a hypercube of size patch_size.
        If list or tuple, the patch has the shape specified by the elements of the list or tuple.
        The patch size must be equal or larger than 2, but smaller than the corresponding array shape.
    array_shape : tuple
        The shape of the array.

    Returns
    -------
    patch_size (None, tuple)
        The shape of the patch.
    """
    n_dims = len(array_shape)
    if patch_size is None:
        patch_size = 2
    elif isinstance(patch_size, (int, float)):
        patch_size = tuple([patch_size] * n_dims)
    elif isinstance(patch_size, (list, tuple)) and len(patch_size) == n_dims:
        if not _is_natural_numbers(patch_size):  # [0,1,..., Inf)
            raise ValueError("Invalid patch size. Must be composed of natural numbers.")
        # Check patch size is smaller than array shape
        idx_valid = [x <= max_val for x, max_val in zip(patch_size, array_shape)]
        if not all(idx_valid):
            raise ValueError(f"The maximum patch size is {array_shape}")
    else:
        raise ValueError(
            f"Invalid patch size. Should be None, int, float, list or tuple of length {n_dims}."
        )
    return patch_size


def _check_padding(padding, array_shape):
    """
    Check the validity of the padding argument based on the array_shape.

    Parameters
    ----------
    padding : (None, int, float, list or tuple)
        The size of the padding to apply to the array.
        If None, zero padding is assumed.
        If int or float, equal padding is set on each dimension of the array.
        If list or tuple, the padding has the shape specified by the elements of the list or tuple.
    array_shape : tuple
        The shape of the array.

    Returns
    -------
    padding tuple
        The padding to apply on each dimension.
    """
    n_dims = len(array_shape)
    if padding is None:
        padding = 0
    elif isinstance(padding, (int, float)):
        padding = tuple([padding] * n_dims)
    elif isinstance(padding, (list, tuple)) and len(padding) == n_dims:
        if not _is_natural_numbers(padding):
            raise ValueError("Invalid padding. Must be composed of natural numbers.")
    else:
        raise ValueError(
            f"Invalid padding. Should be None, int, float, list or tuple of length {n_dims}."
        )
    return padding


def _get_labels_bbox_slices(arr):
    """
    Compute the bounding box slices of non-zero elements in a n-dimensional numpy array.

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


def _get_patch_list_slices(label_arr, label_id, padding, min_patch_size):
    """Get patch subsetting isel dictionary for a given label."""
    # Get label bounding box slices
    list_slices = _get_labels_bbox_slices(label_arr == label_id)
    # Apply padding to the slices
    list_slices = pad_slices(list_slices, padding=padding, valid_shape=label_arr.shape)
    # Increase slices to match min_patch_size
    list_slices = enlarge_slices(list_slices, min_size=min_patch_size, valid_shape=label_arr.shape)
    return list_slices


def get_labeled_object_patches(
    xr_obj,
    label_name,
    n_patches=None,
    labels_id=None,
    padding=None,
    min_patch_size=None,
    highlight_label_id=True,
):
    """
    Create a generator extracting patches around labels (from a prelabeled xr.Dataset).

    Only one parameter between n_patches and labels_id can be specified.
    If n_patches=None and labels_id=None are both None, it returns a patch for each label.
    The patch minimum size is defined by min_patch_size, which default to 2 in all dimensions.
    If the naive label patch size is smaller than min_patch_size, the patch is enlarged to have
    size equal to min_patch_size.

    The output patches are not guaranteed to have equal size !

    Parameters
    ----------
    xr_obj : xr.Dataset
        xr.Dataset with a label array named label_bame.
    label_name : TYPE
        Name of the variable/coordinate representing the label array.
    n_patches : int, optional
        Number of patches to extract. The default is None.
    labels_id : list, optional
        List of labels for which to extract the patch.
        If None, it extracts the patch by label order (1, 2, 3, ...)
        The default is None.
    min_patch_size : (int, tuple), optional
        The minimum size of the patch to extract.
        If None (default) it set a minimum size of 2 in all dimensions.
    padding : (int, tuple), optional
        The padding to apply in each direction.
        If None, it applies 0 padding in every dimension.
        The default is None.
    highlight_label_id : (bool), optional
        If True, the laben_name array of each patch is modified to contain only
        the label_id used to select the patch.

    Yields
    ------
    (xr.Dataset or xr.DataArray)
        A xarray object patch.

    """
    # Get label array information
    label_arr = xr_obj[label_name].data
    dims = xr_obj[label_name].dims
    shape = label_arr.shape

    # Check input arguments
    if n_patches is not None and labels_id is not None:
        raise ValueError("Specify either n_patches or labels_id.")
    label_arr = _check_label_arr(label_arr)  # ouput is np.array
    labels_id = _check_labels_id(labels_id=labels_id, label_arr=label_arr)
    min_patch_size = _check_patch_size(min_patch_size, array_shape=shape)
    padding = _check_padding(padding, array_shape=shape)

    # If no labels, no patch to extract
    n_labels = len(labels_id)
    if n_labels == 0:
        raise ValueError("No labels available.")
        # yield None # TODO: DEFINE CORRECT BEHAVIOUR

    # If n_patches is None --> n_patches = n_labels, else min(n_patches, n_labels)
    n_patches = min(n_patches, n_labels) if n_patches else n_labels

    # Extract patch around the label
    for label_id in labels_id[0:n_patches]:
        # Extract patch list_slice
        list_slices = _get_patch_list_slices(
            label_arr=label_arr, label_id=label_id, padding=padding, min_patch_size=min_patch_size
        )
        # Extract xarray patch around label
        isel_dict = {dim: slc for dim, slc in zip(dims, list_slices)}
        xr_obj_patch = xr_obj.isel(isel_dict)

        # If asked, set label array to 0 except for label_id
        if highlight_label_id:
            xr_obj_patch = highlight_label(xr_obj_patch, label_name=label_name, label_id=label_id)

        # Return the patch around the label
        yield xr_obj_patch


def labels_patch_generator(
    xr_obj,
    variable=None,
    min_value_threshold=-np.inf,
    max_value_threshold=np.inf,
    min_area_threshold=1,
    max_area_threshold=np.inf,
    footprint=None,
    sort_by="area",
    sort_decreasing=True,
    # Patch settings
    n_patches=None,
    padding=None,
    min_patch_size=None,
):
    """
    Create a generator extracting patches around sensible regions of an xr.Dataset.

    The function first derives the labels array, and then it extract patches for n_patches labels.
    If n_patches=None it returns a patch for each label.
    The patch minimum size is defined by min_patch_size, which default to 2 in all dimensions.
    If the naive label patch size is smaller than min_patch_size, the patch is enlarged to have
    size equal to min_patch_size.

    The output patches are not guaranteed to have equal size !

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
        Valid string statistics are "area", "maximum", "mininum", "mean",
        "median", "sum", "standard_deviation", "variance".
        The default is "area".
    sort_decreasing : bool, optional
        If True, sort labels by decreasing 'sort_by' value.
        The default is True.
    n_patches : int, optional
        Number of patches to extract. The default is None (all).
    padding : (int, tuple), optional
        The padding to apply in each direction.
        If None, it applies 0 padding in every dimension.
        The default is None.
    min_patch_size : (int, tuple), optional
        The minimum size of the patch to extract.
        If None (default) it set a minimum size of 2 in all dimensions.

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
    patch_gen = get_labeled_object_patches(
        xr_obj,
        label_name="label",
        n_patches=n_patches,
        padding=padding,
        min_patch_size=min_patch_size,
    )

    return patch_gen
