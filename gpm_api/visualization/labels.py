#!/usr/bin/env python3
"""
Created on Thu Jan 12 12:04:17 2023

@author: ghiggi
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from gpm_api.patch.labels import redefine_label_array
from gpm_api.patch.labels_patch import get_patches_from_labels, labels_patch_generator
from gpm_api.visualization.plot import plot_image

# TODO:
# - gpm_api.plot_label


def get_label_colorbar_settings(label_indices, cmap="Paired"):
    """Return plot and cbar kwargs to plot properly a label array."""
    # Cast to int the label_indices
    label_indices = label_indices.astype(int)
    # Compute number of required colors
    n_labels = len(label_indices)

    # Get colormap if string
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Extract colors
    color_list = [cmap(i) for i in range(cmap.N)]

    # Create the new colormap
    cmap_new = mpl.colors.LinearSegmentedColormap.from_list("Label Classes", color_list, n_labels)

    # Define the bins and normalize
    bounds = np.linspace(1, n_labels + 1, n_labels + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap_new.N)

    # Define the plot kwargs
    plot_kwargs = {}
    plot_kwargs["cmap"] = cmap_new
    plot_kwargs["norm"] = norm

    # Define colorbar kwargs
    ticks = bounds[:-1] + 0.5
    ticklabels = label_indices
    assert len(ticks) == len(ticklabels)
    cbar_kwargs = {}
    cbar_kwargs["label"] = "Label IDs"
    cbar_kwargs["ticks"] = ticks
    cbar_kwargs["ticklabels"] = ticklabels
    return plot_kwargs, cbar_kwargs


def plot_label(
    da, add_colorbar="True", interpolation="nearest", cmap="Paired", fig_kwargs={}, **plot_kwargs
):
    # Ensure array is numpy downstrema
    arr = da.data
    if hasattr(arr, "chunks"):
        arr = arr.compute()
    # Get label_indices and relabel array from 1 to ... for plotting
    label_indices = np.unique(arr)
    label_indices = np.delete(label_indices, np.where(label_indices == 0)[0].flatten())
    relabeled_arr = redefine_label_array(da.data, label_indices=label_indices)
    da_label = da.copy()
    da_label.data = relabeled_arr
    # Replace 0 with nan
    da_label = da_label.where(da_label > 0)
    # Define appropriate colormap
    plot_kwargs, cbar_kwargs = get_label_colorbar_settings(label_indices, cmap="Paired")
    # Plot image
    p = plot_image(
        da_label,
        interpolation=interpolation,
        add_colorbar=add_colorbar,
        cbar_kwargs=cbar_kwargs,
        fig_kwargs=fig_kwargs,
        **plot_kwargs,
    )
    return p


# TODO: this plot labels ! Rename it !
def plot_label_patches(
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
    # Plot options
    add_colorbar=True,
    interpolation="nearest",
    cmap="Paired",
    fig_kwargs={},
    **plot_kwargs,
):

    # Check plot_kwargs keys
    if "cbar_kwargs" in plot_kwargs:
        raise ValueError("'cbar_kwargs' can not be specified when plotting labels.")

    # Define the patch generator
    patch_gen = get_patches_from_labels(
        xr_obj,
        label_name=label_name,
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
    # Plot patches
    # list_da = list(patch_gen)
    # da = list_da[0]
    for da in patch_gen:
        plot_label(
            da[label_name],
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            cmap=cmap,
            fig_kwargs=fig_kwargs,
            **plot_kwargs,
        )
        plt.show()
    return


# TODO: rename with something like plot_label_patches --> when label not yet defined
def plot_patches(
    xr_obj,
    patch_size,
    variable=None,
    # Label options
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
    # Plot options
    add_colorbar=True,
    interpolation="nearest",
    fig_kwargs={},
    cbar_kwargs={},
    **plot_kwargs,
):
    if isinstance(xr_obj, xr.Dataset):
        if variable is None:
            raise ValueError("'variable' must be specified when plotting xr.Dataset patches.")

    # Define generator
    patch_gen = labels_patch_generator(
        xr_obj=xr_obj,
        patch_size=patch_size,
        variable=variable,
        # Label Options
        min_value_threshold=min_value_threshold,
        max_value_threshold=max_value_threshold,
        min_area_threshold=min_area_threshold,
        max_area_threshold=max_area_threshold,
        footprint=footprint,
        sort_by=sort_by,
        sort_decreasing=sort_decreasing,
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

    # Plot patches
    for label_id, xr_patch in patch_gen:
        if isinstance(xr_patch, xr.Dataset):
            xr_patch = xr_patch[variable]
        try:
            plot_image(
                xr_patch,
                interpolation=interpolation,
                add_colorbar=add_colorbar,
                fig_kwargs=fig_kwargs,
                cbar_kwargs=cbar_kwargs,
                **plot_kwargs,
            )
            plt.show()
        except:
            pass

    return
