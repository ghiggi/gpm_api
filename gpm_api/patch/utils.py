#!/usr/bin/env python3
"""
Created on Wed Oct 19 16:51:11 2022

@author: ghiggi
"""
import itertools

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects
from scipy.ndimage.measurements import center_of_mass

from gpm_api.patch.labels import get_areas_labels
from gpm_api.utils.slices import (
    get_slice_size,
)

####--------------------------------------------------------------------------.
# Shapely bounds: (xmin, ymin, xmax, ymax)
# Matlotlib extent: (xmin, xmax, ymin, ymax)
# Cartopy extent: (xmin, xmax, ymin, ymax)
# GPM-API extent: (xmin, xmax, ymin, ymax)

####--------------------------------------------------------------------------.


def get_row_col_slice_centroid(row_slice, col_slice):
    row = int((row_slice.start + row_slice.stop - 1) / 2)
    col = int((col_slice.start + col_slice.stop - 1) / 2)
    return row, col


####--------------------------------------------------------------------------.
#### Patch splitter


def split_large_object_slices(object_slices, patch_size):
    if len(patch_size) == 1:
        patch_size = [patch_size] * len(object_slices)
    if np.any(np.array(patch_size) <= 2):
        raise ValueError("Minimum patch_size should be 2.")
    if len(patch_size) != len(object_slices):
        raise ValueError("Dimensions of patch_size and object_slices do not coincide.")
    l_iterables = []
    for slc, max_size in zip(object_slices, patch_size):
        size = get_slice_size(slc)
        if size > max_size:
            idxs = np.arange(slc.start, slc.stop - 1, max_size)  # start idxs
            l_slc = [slice(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
            l_slc.append(slice(idxs[-1], slc.stop))
            # TODO: optionally remove last slice if size < max_size/2
            # HERE
        else:
            l_slc = [slc]
        l_iterables.append(l_slc)

    list_objects_slices = list(itertools.product(*l_iterables))
    return list_objects_slices


def split_large_objects_slices(objects_slices, patch_size):
    l_object_slices = []
    for object_slices in objects_slices:
        l_slices = split_large_object_slices(object_slices, patch_size)
        l_object_slices.extend(l_slices)
    return l_object_slices


####-------------------------------------------------------------------------.
##########################
#### Patch Extraction ####
##########################

# get_patch_slices_around_point
def get_patch_slices_around(row, col, patch_size, image_shape):
    if len(patch_size) != len(image_shape):
        raise ValueError("patch_size and image shape dimensions do not coincide.")
    # if patch_size[0] < image_shape[0]:
    #     raise ValueError("patch_size on y is larger than image y extent.")
    # if patch_size[1] < image_shape[1]:
    #     raise ValueError("patch_size on x is larger than image x extent.")

    # -------------------------------------
    # Define dx and dy
    dx_left = patch_size[1] // 2 - 1  # -1 to account for  center pixel
    dx_right = patch_size[1] - dx_left - 1  # -1 to account for  center pixel
    dy_up = patch_size[0] // 2 - 1  # -1 to account for  center pixel
    dy_down = patch_size[0] - dy_up - 1  # -1 to account for  center pixel

    # -------------------------------------
    # Define tentative bbox coords
    rmin = row - dy_up
    rmax = row + dy_down
    cmin = col - dx_left
    cmax = col + dx_right

    # -------------------------------------
    # Ensure valid bbox coords
    if rmin < 0:
        rmax = rmax + abs(rmin)
        rmin = 0
    if rmax > (image_shape[0] - 1):
        rmin = rmin - abs(rmax - image_shape[0]) - 1
        rmax = image_shape[0] - 1
    if cmin < 0:
        cmax = cmax + abs(cmin)
        cmin = 0
    if cmax > (image_shape[1] - 1):
        cmin = cmin - abs(cmax - image_shape[1]) - 1
        cmax = image_shape[1] - 1

    # -------------------------------------
    # Build slices
    row_slice = slice(rmin, rmax + 1)
    col_slice = slice(cmin, cmax + 1)
    return row_slice, col_slice


def get_patch_per_label(
    labels,
    intensity,
    patch_size: tuple = (128, 128),
    centered_on="max",
    mask_value=0,
    patch_stats_fun=None,
):
    # Intensity: you might want to mask, or clip intensity in advance
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.find_objects.html
    # ---------------------------------------------------------------------------.
    # ---------------------------------.
    # Check same shape
    if labels.shape != intensity.shape:
        raise ValueError("'labels' and 'intensity' array must have same shape.")

    # Check centered_on
    valid_centered_on = ["min", "max", "centroid", "center_of_mass"]
    if centered_on not in valid_centered_on:
        raise ValueError(f"Valid 'centered_on' argument: {valid_centered_on}")

    # Check patch_size
    if len(patch_size) == 1:
        patch_size = [patch_size] * len(intensity.shape)
    if len(patch_size) != len(intensity.shape):
        raise ValueError("patch_size and image shape dimensions do not coincide.")

    # ---------------------------------.
    # Get data into memory
    if not isinstance(labels, np.ndarray):
        labels = labels.compute()
    if not isinstance(intensity, np.ndarray):
        intensity = intensity.compute()

    # ---------------------------------.
    # Retrieve image shape
    image_shape = labels.shape

    # ---------------------------------.
    # Get bounding box slices for each label
    objects_slices = find_objects(labels)

    # ---------------------------------.
    # If no label, return empty list
    if len(objects_slices) == 0:
        return []

    # ---------------------------------.
    # Loop over each label, and extract patches
    list_patch_slices = []
    # i = 0
    # object_slice = objects_slices[i]
    for i, object_slice in enumerate(objects_slices):
        # If slices larger than patch_size, split into multiple slices and loop over it
        conform_object_slices = split_large_object_slices(object_slice, patch_size=patch_size)
        # r_slice, c_slice = conform_object_slices[0]
        for r_slice, c_slice in conform_object_slices:
            # print(i)
            # ---------------------------------.
            # Define patch label
            tmp_label = labels[r_slice, c_slice]

            # Define patch label mask
            tmp_label_mask = tmp_label == i + 1  # 1 where label, 0 otherwise

            # If no label inside the patch, skip
            if not np.any(tmp_label_mask):
                continue

            # Define patch intensity
            tmp_intensity = intensity[r_slice, c_slice].copy()
            tmp_intensity[~tmp_label_mask] = mask_value

            # If patch is all nan, skip (centered_on min/max would fail)
            if np.all(np.isnan(tmp_intensity)):
                continue

            # ---------------------------------.
            # # DEBUG
            # plt.imshow(tmp_label, cmap="Spectral", vmin=1, interpolation="none")
            # plt.colorbar()
            # plt.show()

            # plt.imshow(tmp_label_mask)
            # plt.colorbar()
            # plt.show()

            # cmap = plt.get_cmap("Spectral").copy()
            # cmap.set_under(color="white")
            # plt.imshow(tmp_intensity, cmap=cmap, vmin=0.1)
            # plt.colorbar()
            # plt.show()

            # ---------------------------------.
            # Retrieve ideal patch center
            if centered_on == "max":
                row, col = np.where(tmp_intensity == np.nanmax(tmp_intensity))
                row = r_slice.start + row[0]
                col = c_slice.start + col[0]
            elif centered_on == "min":
                row, col = np.where(tmp_intensity == np.nanmin(tmp_intensity))
                row = r_slice.start + row[0]
                col = c_slice.start + col[0]
            elif centered_on == "centroid":
                row, col = get_row_col_slice_centroid(row_slice=r_slice, col_slice=c_slice)
            elif centered_on == "center_of_mass":
                row, col = center_of_mass(tmp_label_mask)
                row = int(row)
                col = int(col)
                row = r_slice.start + row
                col = c_slice.start + col
            else:
                raise NotImplementedError("")

            # ---------------------------------.
            # Retrieve actual patch
            row_slice, col_slice = get_patch_slices_around(row, col, patch_size, image_shape)
            # assert get_slice_size(col_slice) == 128
            # print(get_slice_size(col_slice))
            # Append to list
            list_patch_slices.append((row_slice, col_slice))

    # ---------------------------------.
    # Postprocessing list_patch_slices
    # TODO: with shapely?
    # - distance between centroids ---> group ...

    # ---------------------------------.
    # Compute patch statistics
    if callable(patch_stats_fun):
        patch_statistics = [
            patch_stats_fun(intensity[r_slice, c_slice]) for r_slice, c_slice in list_patch_slices
        ]
    else:
        patch_statistics = None
    # ---------------------------------.
    # Return object
    return list_patch_slices, patch_statistics


####--------------------------------------------------------------------------.
#########################
#### Patch Stats Info ###
#########################
# TODO: Inside get_label_stats
# - Add label mask


def patch_stats_fun(patch: np.ndarray):
    width = patch.shape[1]
    height = patch.shape[0]
    patch_area = width * height
    stats_dict = {
        "Max": np.nanmax(patch),
        "Min": np.nanmin(patch),
        "Mean": np.nanmean(patch),
        "Area >= 1": np.nansum(patch >= 1),
        "Area >= 5": np.nansum(patch >= 5),
        "Area >= 20": np.nansum(patch >= 20),
        "Sum": np.nansum(patch),
        "Dry-Wet Area Ratio": np.nansum(patch >= 0.1) / (patch_area),
        "Patch Area": patch_area,
        "HasNaN": np.any(np.isnan(patch)),
    }
    return stats_dict


def get_upper_left_idx_from_str(idx_str):
    if isinstance(idx_str, list):
        return [get_upper_left_idx_from_str(s) for s in idx_str]
    else:
        idx = idx_str.split("-")
        idx = (int(idx[0]), int(idx[1]))
        return idx


def get_patch_info(
    arr,
    min_value_threshold=-np.inf,
    max_value_threshold=np.inf,
    min_area_threshold=1,
    max_area_threshold=np.inf,
    footprint_buffer=None,
    patch_size: tuple = (128, 128),
    centered_on="center_of_mass",
    mask_value=0,
    patch_stats_fun=None,
):
    # Label area
    labels, n_labels, counts = get_areas_labels(
        arr,
        min_value_threshold=min_value_threshold,
        max_value_threshold=max_value_threshold,
        min_area_threshold=min_area_threshold,
        max_area_threshold=max_area_threshold,
        footprint_buffer=footprint_buffer,
    )
    if n_labels > 0:
        # Get patches
        list_patch_slices, patch_statistics = get_patch_per_label(
            labels=labels,
            intensity=arr,
            patch_size=patch_size,
            centered_on=centered_on,
            patch_stats_fun=patch_stats_fun,
        )
        # Found upper left index
        list_patch_upper_left_idx = [
            [slc.start for slc in list_slc] for list_slc in list_patch_slices
        ]
        upper_left_str = [str(row) + "-" + str(col) for row, col in list_patch_upper_left_idx]

        # Define data.frame
        df = pd.DataFrame(patch_statistics)
        df["upper_left_idx"] = upper_left_str
    else:
        df = pd.DataFrame()

    # Hack to return df
    out_str = df.to_json()
    return np.array([out_str], dtype="object")
