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
"""This module contains utilities for orbit processing."""

import numpy as np
import pandas as pd
import xarray as xr


def adjust_short_sequences(arr, min_size):
    """
    Replace value of short sequences of consecutive identical values.

    The function examines contiguous sequences of identical elements in the input
    array.
    If a sequence is shorter than `min_size`, its values are replaced with
    the value of the adjacent longer sequence, working outward from the first
    valid sequence.

    Parameters
    ----------
    arr : array-like
        The input array of values.
    min_size : int
        The minimum number of consecutive identical elements to not be modified.
        Shorter sequences will be replaced with the previous sequence value.

    Returns
    -------
    arr : list
        The modified array with updated values.

    """
    # Ensure input is a 1D numpy array
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("Input must be a 1D array.")

    # If array is empty or has only one element, return as is
    if len(arr) <= 1:
        return arr

    # Create a copy to modify
    result = arr.copy()

    # Find boundaries of sequences (where values change)
    change_indices = np.where(np.diff(result) != 0)[0]
    sequence_starts = np.concatenate(([0], change_indices + 1))
    sequence_ends = np.concatenate((change_indices + 1, [len(result)]))

    # Find the first valid sequence (length >= min_size)
    first_valid_idx = None
    valid_value = None

    for i, (start, end) in enumerate(zip(sequence_starts, sequence_ends, strict=False)):
        if end - start >= min_size:
            first_valid_idx = i
            valid_value = result[start]
            break

    if first_valid_idx is None:
        # If no valid sequence found, return original array
        return result

    # Process sequences after the first valid sequence
    for i in range(first_valid_idx + 1, len(sequence_starts)):
        start = sequence_starts[i]
        end = sequence_ends[i]
        if end - start < min_size:
            result[start:end] = valid_value
        else:
            valid_value = result[start]

    # Process sequences before the first valid sequence (in reverse)
    valid_value = result[sequence_starts[first_valid_idx]]  # Reset to first valid sequence value
    for i in range(first_valid_idx - 1, -1, -1):
        start = sequence_starts[i]
        end = sequence_ends[i]
        if end - start < min_size:
            result[start:end] = valid_value

    # Return array with replaced values
    return result


def get_orbit_direction(lats, n_tol=1):
    """
    Infer the satellite orbit direction from latitude values.

    This function determines the orbit direction by computing the sign
    of the differences between consecutive latitude values.

    A positive sign (+1) indicates an ascending orbit (increasing latitude),
    while a negative sign (-1) indicates a descending orbit (decreasing latitude).

    Any zero differences are replaced by the nearest nonzero direction.
    Additionally, short sequences of direction changes - those lasting fewer
    than `n_tol` consecutive data points - are adjusted to reduce
    the influence of geolocation errors.

    Parameters
    ----------
    lats : array-like
        1-dimensional array of latitude values corresponding to the satellite's orbit.
    n_tol : int, optional
        The minimum number of consecutive data points required to confirm a change
        in direction.
        Sequences shorter than this threshold will be smoothed. Default is 1.

    Returns
    -------
    numpy.ndarray
        A 1-dimensional array of the same length as `lats` containing
        the inferred orbit direction.
        A value of +1 denotes an ascending orbit and -1 denotes a descending orbit.

    Examples
    --------
    >>> lats = [10, 10.5, 11, 10.8, 10.3, 10, 9.5, 9.8, 10.1]
    >>> get_orbit_direction(lats)
    array([ 1,  1,  1, -1, -1, -1, -1,  1,  1])
    """
    # Get direction (1 for ascending, -1 for descending)
    directions = np.sign(np.diff(lats))
    directions = np.append(directions[0], directions)  # Include starting point
    # Set 0 to NaN and infill values
    directions = directions.astype(float)
    directions[directions == 0] = np.nan
    if np.all(np.isnan(directions)):
        raise ValueError("Invalid orbit.")
    directions = pd.Series(directions).ffill().bfill().to_numpy()
    # Remove short consecutive changes in direction to account for geolocation errors
    directions = adjust_short_sequences(directions, min_size=n_tol)
    # Convert to integer array
    directions = directions.astype(int)
    return directions


def get_orbit_mode(ds):
    # Retrieve latitude coordinates
    if "SClatitude" in ds:
        lats = ds["SClatitude"]
    elif "scLat" in ds:
        lats = ds["scLat"]
    else:
        # Define cross_track idx defining the orbit coordinates
        # - TODO: conically scanning vs cross-track scanning
        # - TODO: Use SClatitude  ?
        # Remove invalid outer cross-track_id if selecting 0 or -1
        # from gpm.visualization.orbit import remove_invalid_outer_cross_track
        # ds, _ = remove_invalid_outer_cross_track(ds)
        idx = int(ds["cross_track"].shape[0] / 2)
        lats = ds["lat"].isel({"cross_track": idx}).to_numpy()
    directions = get_orbit_direction(lats, n_tol=10)
    orbit_mode = xr.DataArray(directions, dims=["along_track"])
    return orbit_mode
