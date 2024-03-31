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
"""This module test the parallel utilities."""


import dask
import pytest
from dask import delayed

from gpm.utils.parallel import compute_list_delayed


# Test function to be used with dask.delayed
def dummy_function(x):
    return x + 1


@pytest.fixture(scope="module")
def _dask_client():
    """Fixture for creating and closing a Dask client."""
    with dask.config.set(
        {"scheduler": "threads"},
    ):  # Use threaded scheduler for simplicity in tests
        yield  # No setup or teardown needed for this simple test


@pytest.mark.usefixtures("_dask_client")
def test_compute_list_delayed():
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
