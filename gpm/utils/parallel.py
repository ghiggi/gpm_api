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
"""This module contains utilities for parallel processing."""
import itertools

import dask


def compute_list_delayed(list_delayed, max_concurrent_tasks=None):
    """Compute the list of Dask delayed objects in blocks of max_concurrent_tasks.

    Parameters
    ----------
    list_delayed : list
        List of Dask delayed objects.
    max_concurrent_task : int
        Maximum number of concurrent tasks to execute.

    Returns
    -------
    list
        List of computed results.

    """
    if max_concurrent_tasks is None:
        return dask.compute(*list_delayed)

    max_concurrent_tasks = min(len(list_delayed), max_concurrent_tasks)
    computed_results = []
    for i in range(0, len(list_delayed), max_concurrent_tasks):
        subset_delayed = list_delayed[i : (i + max_concurrent_tasks)]
        computed_results.extend(dask.compute(*subset_delayed))
    return computed_results


def create_group_slices(chunksizes, group_size):
    """
    Create slices by grouping contiguous chunks along a dimension.

    Parameters
    ----------
    chunksizes : list or tuple of int
        Sizes of chunks along the dimension to be grouped.
    group_size : int
        Number of chunks to group together.

    Returns
    -------
    list of slice
        List of slice objects representing the start and stop positions
        of each group of contiguous chunks.
    """
    group_slices = []
    start = 0
    i = 0
    while i < len(chunksizes):
        # Take group_size chunks
        group_chunk_sizes = chunksizes[i : i + group_size]
        stop = start + sum(group_chunk_sizes)
        group_slices.append(slice(start, stop))
        start = stop
        i += group_size
    return group_slices


def get_block_slices(ds, **dim_chunks_kwargs):
    """
    Generate a list of slice dictionaries for grouping chunks in an xarray Dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset for which slices are generated.
    **dim_chunks_kwargs : dict of {str: int}
        Keyword arguments where each key is a dimension name and each value is
        the number of contiguous chunks to group together for that dimension.

    Returns
    -------
    list of dict
        A list of dictionaries where each dictionary maps dimension names to
        slice objects, defining groups of contiguous chunks along the specified
        dimensions.
    """
    # Chunk input is provided
    if len(dim_chunks_kwargs) == 0:
        raise ValueError("Specify at least 1 <dim>=<n_chunks> argument.")

    # Dictionary to store slices for each dimension
    dim_slices = {}
    for dim, group_size in dim_chunks_kwargs.items():
        # Get chunk sizes along this dimension
        chunksizes = ds.chunksizes[dim]
        # Create group slices
        slices = create_group_slices(chunksizes, group_size)
        dim_slices[dim] = slices

    # Generate all combinations of slices across dimensions
    list_of_slices = []
    dims = list(dim_chunks_kwargs.keys())
    slices_lists = [dim_slices[dim] for dim in dims]
    for slices_combination in itertools.product(*slices_lists):
        # Build a dict mapping dimension names to slices
        slice_dict = dict(zip(dims, slices_combination, strict=False))
        list_of_slices.append(slice_dict)
    return list_of_slices
