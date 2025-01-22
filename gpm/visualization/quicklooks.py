# -----------------------------------------------------------------------------.
# MIT License

# Copyright (c) 2024 GPM-API developers
#
# This file is part of GPM-API.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -----------------------------------------------------------------------------.
"""This module contains utility for generating quicklooks."""
import numpy as np
import xarray as xr

from gpm.utils.slices import get_indices_from_list_slices


def create_quicklooks_dataset(list_ds, spacing=2, total_size=200, concat_dim="along_track"):
    """Concatenate multiple xarray.Dataset objects for quicklook plotting.

    This function merges the Datasets in `list_ds` along the dimension `concat_dim`.
    Between consecutive Datasets, a dummy (NaN-filled) Dataset of width `spacing`
    is inserted to visually separate the events in a plot. After the last Dataset,
    any remaining space to reach `total_size` is also filled with NaNs.

    A coordinate named "spacing_flag" is added to the result, indicating which
    indices in `concat_dim` are actual data (0) versus inserted NaNs (1).

    Parameters
    ----------
    list_ds : list of xarray.Dataset
        List of xarray Datasets to concatenate along the dimension `concat_dim`.
        All Datasets should share the same variables and dimension names,
        except for differences in the size of `concat_dim`.
    spacing : int, optional
        Number of NaN entries to insert between consecutive Datasets.
        Defaults to 2.
    total_size : int, optional
        The desired total size (maximum length) along the `concat_dim`
        dimension in the final "quicklook" dataset. Defaults to 200.
    concat_dim : str, optional
        Name of the dimension along which to concatenate. Typically "along_track".

    Returns
    -------
    xarray.Dataset
        A new Dataset of size `total_size` (or less if the data plus spacing
        exceeds `total_size`) along the `concat_dim` dimension. Between real
        data segments, there are NaNs for visual separation, and an integer
        coordinate "spacing_flag" (1 = NaNs region, 0 = real data).
    """
    # Create dummy dataset of size subplot_size
    ds_template = list_ds[0].isel({concat_dim: [0]}, drop=False)
    ds_template = ds_template.isel({concat_dim: np.zeros(total_size, dtype=int)})
    ds_nan = xr.full_like(ds_template, fill_value=np.nan)

    # Create composite dataset
    # - Original dataset slices are interleaved by NaN dataset of size 'spacing'
    list_ds_quicklook = []
    list_dummy_slices = []
    size = 0
    n_slices = len(list_ds)
    for i, ds in enumerate(list_ds):
        list_ds_quicklook.append(ds)
        size += len(ds[concat_dim])
        # Insert NaN data between slices
        if i == n_slices - 1:  # noqa SIM108
            size_dummy = total_size - size
        else:
            size_dummy = spacing
        if size_dummy > 0:
            list_dummy_slices.append(slice(size, size + size_dummy))
            size += size_dummy
            list_ds_quicklook.append(ds_nan.isel({concat_dim: slice(0, size_dummy)}))

    # Combine slices together
    ds_quicklook = xr.concat(list_ds_quicklook, dim=concat_dim)
    spacing_flag = np.zeros(ds_quicklook[concat_dim].shape)
    spacing_flag[get_indices_from_list_slices(list_dummy_slices)] = 1
    ds_quicklook = ds_quicklook.assign_coords({"spacing_flag": (concat_dim, spacing_flag)})

    # Truncate if we exceeded total_size
    if ds_quicklook.sizes[concat_dim] > total_size:
        ds_quicklook = ds_quicklook.isel({concat_dim: slice(0, total_size)})
    return ds_quicklook


def create_quicklooks_datasets(ds, list_slices, subplot_size=200, spacing=2, n_subplots=4, concat_dim="along_track"):
    """
    Build an array dataset for quicklook plotting of interesting data regions.

    It extracts slices of interest from the input dataset, ensuring each group of slices
    having a total length not exceeding ``subplot_size``.
    Within each group,  slices are concatenated along the "along_track" dimension,
    and a dummy (NaN) dataset of width ``spacing`` is inserted between slices to visually separate them.
    Any leftover space in the subplot is filled with NaNs.

    A new coordinate, "spacing_flag", is added to indicate the indices that correspond
    to these NaN (spacing) regions.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing an "along_track" dimension.
    list_slices : list of slice
        List of valid data segments along the "along_track" dimension.
    subplot_size : int, optional
        Maximum length of each subplot (group of slices) along the "along_track" axis.
        Defaults to 200.
    spacing : int, optional
        Number of NaN points inserted between consecutive slices in a subplot. Defaults to 2.
    n_subplots : int, optional
        Maximum number of subplot groups to produce. Defaults to 4.

    Returns
    -------
    list of xarray.Dataset
        A list of up to ``n_subplots`` datasets, each dataset having:

        * Dimension "along_track" size up to ``subplot_size``.
        * Slices concatenated with NaNs for spacing.
        * A "spacing_flag" coordinate along "along_track" (0 for real data, 1 for NaN spacing).

    """
    # Rerieve dataset slices for subplots
    list_subplot_slices = get_subplot_slices(
        list_slices=list_slices,
        subplot_size=subplot_size,
        spacing=spacing,
        n_subplots=n_subplots,
    )

    list_subplots_ds = [
        create_quicklooks_dataset(
            list_ds=[ds.isel({concat_dim: slc}) for slc in subplot_slices],
            spacing=spacing,
            total_size=subplot_size,
            concat_dim=concat_dim,
        )
        for subplot_slices in list_subplot_slices
    ]
    return list_subplots_ds

    # # Create dummy dataset
    # ds_nan = xr.ones_like(ds.isel({"along_track": slice(0, subplot_size)}))*np.nan

    # # Create composite dataset for each subplot
    # # - Original dataset slices are interleaved by NaN dataset of size 'spacing'
    # list_subplots_datasets = []
    # for subplot_slices in list_subplot_slices:

    #     list_ds_subplot = []
    #     list_dummy_slices = []
    #     size = 0
    #     n_slices = len(subplot_slices)
    #     for i in range(0, n_slices):
    #         # Add slice to list
    #         slc = subplot_slices[i]
    #         list_ds_subplot.append(ds.isel({"along_track":slc}))
    #         size += slc.stop - slc.start
    #         # Insert NaN data between slices
    #         if i == n_slices - 1:
    #             size_dummy = subplot_size - size
    #         else:
    #             size_dummy = spacing
    #         if size_dummy > 0:
    #             list_dummy_slices.append(slice(size, size + size_dummy))
    #             size += size_dummy
    #             list_ds_subplot.append(ds_nan.isel({"along_track": slice(0, size_dummy)}))
    #     # Combine slices together
    #     subplot_dataset = xr.concat(list_ds_subplot, dim="along_track")
    #     spacing_flag = np.zeros(subplot_dataset["along_track"].shape)
    #     spacing_flag[get_indices_from_list_slices(list_dummy_slices)] = 1
    #     subplot_dataset = subplot_dataset.assign_coords({"spacing_flag": ("along_track", spacing_flag)})
    #     # Add subplot dataset to the subplots list
    #     list_subplots_datasets.append(subplot_dataset)
    # return list_subplots_datasets


def get_subplot_slices(list_slices, subplot_size=100, spacing=2, n_subplots=4):
    """
    Group slices into subplots, ensuring each subplot stays within a maximum size.

    This function accumulates slices (plus their inter-slice spacing) until adding
    another slice would exceed ``subplot_size``. It then starts a new group (subplot).
    The number of subplots is capped at ``n_subplots``.

    Parameters
    ----------
    list_slices : list of slice
        List of Python slice objects along an xarray object dimension.
        Each slice has a .start and .stop attribute (integers).
    subplot_size : int, optional
        The maximum allowable sum of slice lengths (plus spacing) for each subplot.
        Defaults to 100.
    spacing : int, optional
        Spacing that is reserved after each slice. Defaults to 2.
        This is accounted for in the total length when deciding whether a slice fits in
        the current subplot.
    n_subplots : int, optional
        Maximum number of subplot groups to return. Defaults to 4.

    Returns
    -------
    list of list of slice
        A list of sub-lists, where each sub-list contains the slices assigned
        to one subplot. Each group's combined length (sum of slices plus spacing)
        does not exceed ``subplot_size`` (except possibly the last group),
        unless truncated by ``n_subplots``.
    """
    # TODO:
    # - ENABLE SPLIT AT BORDER: current_length + length > subplot_size --> create two slices !
    # - DEAL WHEN LENGTH > subplot_size

    subplots_slices = []
    current_group = []
    current_length = 0

    for slc in list_slices:
        length = slc.stop - slc.start
        # TODO: here we remove data slice if larger than subplot size
        if length > subplot_size:
            continue
        # If adding this slice exceeds the available space in the current subplot, start a new one
        if current_length + length > subplot_size:
            subplots_slices.append(current_group)
            current_group = [slc]
            current_length = length + spacing
        else:
            current_group.append(slc)
            current_length += length + spacing

    # Add the last group if not empty
    if current_group:
        subplots_slices.append(current_group)

    # Keep only up to n_subplots subplots (truncate if you have more)
    subplots_slices = subplots_slices[:n_subplots]
    return subplots_slices
