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


def get_list_overpass_time(timesteps, interval=None):
    """Return a list with (start_time, end_time) of the overpasses.

    This function is typically called on a regional subset of a bucket archive.
    """
    if interval is None:
        interval = np.array(60, dtype="m8[m]")
    timesteps = sorted(timesteps)

    # Compute time difference
    time_diff = np.diff(timesteps)

    # Initialize
    current_start_time = timesteps[0]
    list_time_periods = []

    for i, dt in enumerate(time_diff):
        if i == 0:
            continue
        if dt > interval:
            end_time = timesteps[i]
            time_period = (current_start_time, end_time)
            list_time_periods.append(time_period)
            # Update
            current_start_time = timesteps[i]
    # Last point
    end_time = timesteps[-1]
    time_period = (current_start_time.astype(str), end_time.astype(str))
    list_time_periods.append(time_period)
    return list_time_periods


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
