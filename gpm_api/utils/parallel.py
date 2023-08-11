#!/usr/bin/env python3
"""
Created on Wed Aug  2 18:15:57 2023

@author: ghiggi
"""
import dask


def compute_list_delayed(list_delayed, max_concurrent_tasks=None):
    """
    Compute the list of Dask delayed objects in blocks of max_concurrent_tasks.

    Parameters:
    list_results (list): List of Dask delayed objects.
    max_concurrent_task (int): Maximum number of concurrent tasks to execute.

    Returns:
    list: List of computed results.
    """
    if max_concurrent_tasks is None:
        computed_results = dask.compute(*list_delayed)
    else:
        computed_results = []
        for i in range(0, len(list_delayed), max_concurrent_tasks):
            subset_delayed = list_delayed[i : (i + max_concurrent_tasks)]
            computed_results.extend(dask.compute(*subset_delayed))

    return computed_results
