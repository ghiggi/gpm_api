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
import dask


def compute_list_delayed(list_delayed, max_concurrent_tasks=None):
    """Compute the list of Dask delayed objects in blocks of max_concurrent_tasks.

    Parameters
    ----------
    list_results (list): List of Dask delayed objects.
    max_concurrent_task (int): Maximum number of concurrent tasks to execute.

    Returns
    -------
    list: List of computed results.

    """
    if max_concurrent_tasks is None:
        return dask.compute(*list_delayed)

    max_concurrent_tasks = min(len(list_delayed), max_concurrent_tasks)
    computed_results = []
    for i in range(0, len(list_delayed), max_concurrent_tasks):
        subset_delayed = list_delayed[i : (i + max_concurrent_tasks)]
        computed_results.extend(dask.compute(*subset_delayed))
    return computed_results
