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
"""This module contains a mix of function to analysis bucket archives."""
import numpy as np


def get_list_overpass_time(timesteps):
    """Return a list with (start_time, end_time) of the overpasses.

    This function is typically called on a regional subset of a bucket archive.
    """
    list_time_periods = []
    start_time = timesteps[0]
    for i in range(1, len(timesteps)):
        if timesteps[i] - timesteps[i - 1] > np.array(60, dtype="m8[m]"):
            end_time = timesteps[i - 1]
            time_period = (start_time, end_time)
            list_time_periods.append(time_period)
            # Update
            start_time = timesteps[i]
    # Last point
    end_time = timesteps[-1]
    time_period = (start_time.astype(str), end_time.astype(str))
    list_time_periods.append(time_period)
    return list_time_periods
