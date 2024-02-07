#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:08:19 2024

@author: ghiggi
"""

import dask
from dask import delayed
import pytest
from gpm_api.utils.parallel import compute_list_delayed


# Test function to be used with dask.delayed
def dummy_function(x):
    return x + 1


@pytest.fixture(scope="module")
def dask_client():
    """Fixture for creating and closing a Dask client."""
    with dask.config.set(
        {"scheduler": "threads"}
    ):  # Use threaded scheduler for simplicity in tests
        yield  # No setup or teardown needed for this simple test


def test_compute_list_delayed(dask_client):
    """Check that compute_list_delayed works correctly."""
    values = list(range(10))
    expected_results = [dummy_function(i) for i in values]

    # Create a list of delayed objects
    list_delayed = [delayed(dummy_function)(i) for i in range(10)]

    # Test without max_concurrent_tasks
    results = compute_list_delayed(list_delayed, max_concurrent_tasks=None)
    assert expected_results == list(results)

    # Test with 0 < max_concurrent_tasks < len(list_delayed)
    results = compute_list_delayed(list_delayed, max_concurrent_tasks=5)
    assert expected_results == results

    results = compute_list_delayed(list_delayed, max_concurrent_tasks=1)
    assert expected_results == results

    # Test with max_concurrent_tasks  > len(list_delayed)
    results = compute_list_delayed(list_delayed, max_concurrent_tasks=20)
    assert expected_results == results
