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
"""This module contains functions to analysis bucket archives."""
import datetime

import numpy as np
import pandas as pd

from gpm.utils.xarray import xr_drop_constant_dimension, xr_first


def get_list_overpass_time(timesteps, interval=None):
    """Return a list with (start_time, end_time) of the overpasses.

    This function is typically called on a regional subset of a bucket archive.
    """
    # Check interval
    if interval is None:
        interval = np.array(60, dtype="m8[m]")

    # Check timesteps
    # - Ensure numpy.datetime64
    timesteps = np.unique(timesteps)
    timesteps = np.sort(timesteps)

    # Deal with edge cases
    if len(timesteps) == 0:
        raise ValueError("No timesteps available.")

    if len(timesteps) == 1:
        return [(timesteps[0], timesteps[0])]

    # Compute time difference
    time_diff = np.diff(timesteps)

    # Initialize
    list_time_periods = []
    current_start_time = timesteps[0]
    for i, dt in enumerate(time_diff):
        if i == 0:
            continue
        if dt > interval:
            end_time = timesteps[i]
            time_period = (current_start_time, end_time)
            list_time_periods.append(time_period)
            # Update
            current_start_time = timesteps[i + 1]

    # Add the final group
    list_time_periods.append((current_start_time, timesteps[-1]))
    return list_time_periods


def split_by_overpass(df, interval=None):
    """Split dataframe by overpass."""
    list_time_periods = get_list_overpass_time(timesteps=df["time"], interval=interval)
    list_df = [
        df[np.logical_and(df["time"] >= start_time, df["time"] <= end_time)]
        for start_time, end_time in list_time_periods
    ]
    return list_df


def get_swath_indices(df):
    """Retrieve orbit dimension indices.

    Assumes only two granule id might be present.
    Assumes that if two granule id are present, the max along_track_id of the
    first granule is the real maximum.

    """
    # Split gpm_id into granule_id and along_track_id (make sure they are integers)
    df_along = df["gpm_id"].str.split("-", expand=True).astype(int)
    df_along.columns = ["granule_id", "along_track_id"]

    # We will assign new x indices so that each granule's along-track block is contiguous.
    x_index_list = []
    # Allocate an array to hold the new x value for each row in df.
    x_values = np.empty(len(df), dtype=int)

    offset = 0
    # Process granules in sorted order (you may change this ordering if needed)
    for granule in sorted(df_along["granule_id"].unique()):
        # Create a boolean mask for rows corresponding to this granule
        mask = df_along["granule_id"] == granule
        # Get the along-track ids for this granule
        along_vals = df_along.loc[mask, "along_track_id"]
        # Determine the minimum and maximum along-track id in this granule.
        min_track = along_vals.min()
        max_track = along_vals.max()
        # The number of along-track positions in this granule
        n_tracks = max_track - min_track + 1

        # Build the new x indices for this granule:
        # They will run from "offset" to "offset+n_tracks-1"
        new_x_indices = np.arange(offset, offset + n_tracks)
        # Append these new indices to our full list of x indices
        x_index_list.extend(new_x_indices)

        # Now assign a new x coordinate for every row in this granule.
        # The formula subtracts the minimum (to start at 0 for that granule)
        # and then adds the current offset.
        x_values[mask] = along_vals - min_track + offset

        # Update the offset for the next granule.
        offset += n_tracks

    # The complete set of x indices is just all integers from 0 to offset-1.
    x_index = np.arange(offset)

    # Retrieve y_index and y_values from cross-track IDs
    y_min = df["gpm_cross_track_id"].min()
    y_max = df["gpm_cross_track_id"].max()
    y_index = np.arange(y_min, y_max + 1)
    y_values = df["gpm_cross_track_id"]

    return (x_index, x_values), (y_index, y_values)


def overpass_to_dataset(df_overpass):
    """Reshape an overpass dataframe to a xarray.Dataset.

    The resulting dataset will have missing geolocation for footprints
    that are not included in the df_overpass.
    """
    # Retrieve dimension indices
    (x_index, x_values), (y_index, y_values) = get_swath_indices(df_overpass)
    df_overpass["x_index"] = x_values
    df_overpass["y_index"] = y_values

    # Set index
    df_overpass = df_overpass.set_index(["x_index", "y_index"])

    # Create MultiIndex with all possible combinations
    full_index = pd.MultiIndex.from_product([x_index, y_index], names=["x_index", "y_index"])

    # Reindex to include all interval combinations
    # --> Add nan to rows with inexisitng index combo
    df_swath = df_overpass.reindex(full_index)

    # Convert to dataset
    ds_swath = df_swath.to_xarray()
    ds_swath = ds_swath.rename_dims({"x_index": "along_track", "y_index": "cross_track"})
    ds_swath["time"] = xr_first(ds_swath["time"], dim="cross_track")
    ds_swath["gpm_id"] = xr_first(ds_swath["gpm_id"], dim="cross_track")
    ds_swath["gpm_cross_track_id"] = xr_first(ds_swath["gpm_cross_track_id"], dim="along_track")

    # Drop dimension for variables that are all equals along a dimension
    # - If all values are nan, drop dimension
    ds_swath = xr_drop_constant_dimension(ds_swath)

    # Set coordinates
    candidate_coords = ["lon", "lat", "time", "gpm_id", "gpm_along_track_id", "gpm_cross_track_id"]
    available_coords = [coord for coord in candidate_coords if coord in ds_swath]
    ds_swath = ds_swath.set_coords(available_coords)

    # Drop dummy index
    ds_swath = ds_swath.drop_vars(["x_index", "y_index"])
    return ds_swath


def add_overpass_id(df, interval=None, time="time"):
    if interval is None:
        interval = pd.Timedelta(minutes=2)

    df = df.sort_values(by="time")  # Sort by time
    # TODO: drop column with missing time

    # Initialize
    group_labels = []
    current_group = 0
    group_labels.append(current_group)  # first timestep

    # Compute time difference
    time_diff = df[time].diff().to_numpy()
    # Assign group numbers based on the time intervals
    for dt in time_diff[1:]:
        if dt <= interval:  # if same overpass
            group_labels.append(current_group)
        else:
            current_group += 1
            group_labels.append(current_group)
    df["overpass_id"] = group_labels
    return df


def count_overpass_occurence(df, interval=None, time="time"):
    df = add_overpass_id(df, interval=interval, time=time)
    count_overpass_beams = df.groupby("overpass_id")[df.columns[0]].count()
    count_overpass_beams.name = "count_overpass_occurence"
    df = df.join(count_overpass_beams, on="overpass_id")
    return df


def ensure_start_end_time_interval(start_time, end_time, interval=None):
    from gpm.io.checks import check_time

    # Convert np.datetime64 to datetime if needed
    start_time = check_time(start_time)
    end_time = check_time(end_time)
    if interval is None:
        return start_time, end_time

    # Ensure interval is of type datetime.timedelta
    if not isinstance(interval, datetime.timedelta):
        raise ValueError("Interval must be of type datetime.timedelta")

    # Calculate the current time difference
    time_difference = end_time - start_time

    # If the time difference is less than the desired interval, modify the times
    if time_difference < interval:
        start_time = start_time - interval / 2
        end_time = end_time + interval / 2
    return start_time, end_time
