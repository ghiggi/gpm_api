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
"""This module contains utility to define events."""

import numpy as np


def remove_isolated_indices(indices, neighbor_min_size, neighbor_interval):
    """
    Remove isolated indices that do not have enough neighboring indices within a specified time gap.

    An index is considered isolated (and thus removed) if it does not have at least `neighbor_min_size` other
    indices within the `neighbor_interval` before or after it.
    In other words, for each index, we look for how many other indices fall into the
    index neighborhood defined as [index - neighbor_interval, index + neighbor_interval], excluding it itself.
    If the count of such neighbors is less than `neighbor_min_size`, that index is removed.

    Parameters
    ----------
    indices : array-like of numpy.datetime64
        Sorted or unsorted array of indices.
    neighbor_interval : int or numpy.timedelta64
        The size of the neighborhood.
        Only indices that fall in the [index - neighbor_interval, index + neighbor_interval] are considered neighbors.
    neighbor_min_size : int, optional
        The minimum number of indices required to fall into the neighborhood for an index to be considered non-isolated.
        - If `neighbor_min_size=0,  then no index is considered isolated and no filtering occurs.
        - If `neighbor_min_size=1`, the index must have at least another index within the neighborhood..
        - If `neighbor_min_size=2`, the index must have at least two other indices within the neighborhood.
        Defaults to 1.

    Returns
    -------
    numpy.ndarray
        Array of indices with isolated entries removed.
    """
    # Sort indices
    indices = np.array(indices)
    indices.sort()

    # Do nothing if neighbor_min_size is 0
    if neighbor_min_size == 0:
        return indices

    # Compute the start and end of the interval for each timestep
    neighboorhood_starts = indices - neighbor_interval
    neighboorhood_ends = indices + neighbor_interval

    # Use searchsorted to find the positions where these intervals would be inserted
    # to keep the array sorted. This effectively gives us the bounds of indices
    # within the neighbor interval.
    left_indices = np.searchsorted(indices, neighboorhood_starts, side="left")
    right_indices = np.searchsorted(indices, neighboorhood_ends, side="right")

    # The number of neighbors is the difference in indices minus one (to exclude the index itself)
    n_neighbors = right_indices - left_indices - 1
    valid_mask = n_neighbors >= neighbor_min_size

    non_isolated_indices = indices[valid_mask]
    return non_isolated_indices


def group_indices_into_events(indices, intra_event_max_distance):
    """
    Group indices into events based on intra_event_max_distance.

    Parameters
    ----------
    indices : array-like
        Sorted array of valid indices.
        Accept also datetime64 arrays.
    intra_event_max_distance : int or numpy.timedelta64
        Maximum distance allowed between consecutive indices for them
        to be considered part of the same event.
        If indices are datetime64 arrays, specify intra_event_max_distance as numpy.timedelta64.

    Returns
    -------
    list of numpy.ndarray
        A list of events, where each event is an array of indices.
    """
    # Deal with case with no indices
    if len(indices) == 0:
        return []

    # Ensure indices are sorted
    indices.sort()

    # Compute differences between consecutive indices
    diffs = np.diff(indices)

    # Identify the indices where the gap is larger than intra_event_max_distance
    # These indices represent boundaries between events
    break_indices = np.where(diffs > intra_event_max_distance)[0] + 1

    # Split the indices at the identified break points
    events = np.split(indices, break_indices)
    return events


def get_event_slices(indices, neighbor_min_size, neighbor_interval, intra_event_max_distance):
    indices = remove_isolated_indices(indices, neighbor_min_size=neighbor_min_size, neighbor_interval=neighbor_interval)
    list_events = group_indices_into_events(indices, intra_event_max_distance)
    list_slices = [slice(event_indices[0], event_indices[-1] + 1) for event_indices in list_events]
    return list_slices
