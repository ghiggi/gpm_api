#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:04:17 2023

@author: ghiggi
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from gpm_api.patch.labels import redefine_label_array
from gpm_api.patch.labels_patch import get_labeled_object_patches
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
    cmap_new = mpl.colors.LinearSegmentedColormap.from_list(
        'Label Classes', color_list, n_labels)

    # Define the bins and normalize
    bounds = np.linspace(1, n_labels+1, n_labels+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap_new.N)
    
    # Define the plot kwargs 
    plot_kwargs = {}
    plot_kwargs['cmap'] = cmap_new 
    plot_kwargs['norm'] = norm 
    
    # Define colorbar kwargs 
    ticks = bounds[:-1] + 0.5
    ticklabels = label_indices
    assert len(ticks) == len(ticklabels)
    cbar_kwargs = {}
    cbar_kwargs['label'] = "Label IDs"
    cbar_kwargs['ticks'] = ticks
    cbar_kwargs['ticklabels'] = ticklabels 
    return plot_kwargs, cbar_kwargs


def plot_label(da, 
               add_colorbar="True",
               interpolation="nearest", 
               cmap = "Paired",
               fig_kwargs={},
               **plot_kwargs):
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
    p = plot_image(da_label, interpolation=interpolation, add_colorbar=add_colorbar,
                   cbar_kwargs = cbar_kwargs, fig_kwargs=fig_kwargs, **plot_kwargs)
    return p


def plot_label_patches(xr_obj, 
                       label_name,
                       n_patches=None,
                       labels_id=None,
                       padding=None, 
                       min_patch_size=None,
                       add_colorbar=True,
                       interpolation="nearest", 
                       cmap="Paired",
                       fig_kwargs={},
                       **plot_kwargs):
    
    # Check plot_kwargs keys
    if "cbar_kwargs" in plot_kwargs: 
        raise ValueError("'cbar_kwargs' can not be specified when plotting labels.")
    
    # Define the patch generator 
    patch_gen = get_labeled_object_patches(
        xr_obj, 
        label_name=label_name, 
        n_patches=n_patches,
        labels_id=labels_id,
        padding=padding,
        min_patch_size=min_patch_size, 
    )
    # Plot patches
    # list_da = list(patch_gen)
    # da = list_da[0]
    for da in patch_gen:
        p = plot_label(da[label_name], 
                       add_colorbar=add_colorbar,
                       interpolation=interpolation, 
                       cmap = cmap,
                       fig_kwargs=fig_kwargs,
                       **plot_kwargs)
        plt.show()
    return None