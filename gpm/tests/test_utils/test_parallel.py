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
import numpy as np
import pytest
import xarray as xr
from dask import delayed

from gpm.utils.parallel import (
    compute_list_delayed,
    create_group_slices,
    get_block_slices,
)


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


def test_create_group_slices():
    # Test with evenly divisible group_size
    chunksizes = [10, 10, 10, 10]
    group_size = 2
    expected_slices = [slice(0, 20), slice(20, 40)]
    result = create_group_slices(chunksizes, group_size)
    assert result == expected_slices, f"Expected {expected_slices}, got {result}"

    # Test with group_size larger than number of chunks
    group_size = 5
    expected_slices = [slice(0, 40)]
    result = create_group_slices(chunksizes, group_size)
    assert result == expected_slices, f"Expected {expected_slices}, got {result}"

    # Test with group_size not evenly dividing chunks
    chunksizes = [10, 15, 20]
    group_size = 2
    expected_slices = [slice(0, 25), slice(25, 45)]
    result = create_group_slices(chunksizes, group_size)
    assert result == expected_slices, f"Expected {expected_slices}, got {result}"

    # Test with group_size of 1
    group_size = 1
    expected_slices = [slice(0, 10), slice(10, 25), slice(25, 45)]
    result = create_group_slices(chunksizes, group_size)
    assert result == expected_slices, f"Expected {expected_slices}, got {result}"


def test_get_block_slices():
    # Create a sample Dataset with known chunksizes
    data = np.zeros((45, 60))
    ds = xr.Dataset({"var": (("x", "y"), data)})
    ds = ds.chunk({"x": (10, 15, 20), "y": (20, 20, 20)})

    # Test grouping along 'x' dimension
    slices = get_block_slices(ds, x=2)
    expected_slices = [
        {"x": slice(0, 25)},
        {"x": slice(25, 45)},
    ]
    assert len(slices) == len(expected_slices), f"Expected {len(expected_slices)} slices, got {len(slices)}"
    for s, e in zip(slices, expected_slices, strict=False):
        assert s == e, f"Expected {e}, got {s}"

    # Test grouping along 'y' dimension
    slices = get_block_slices(ds, y=3)
    expected_slices = [
        {"y": slice(0, 60)},
    ]
    assert len(slices) == len(expected_slices), f"Expected {len(expected_slices)} slices, got {len(slices)}"
    for s, e in zip(slices, expected_slices, strict=False):
        assert s == e, f"Expected {e}, got {s}"

    # Test grouping along both 'x' and 'y' dimensions
    slices = get_block_slices(ds, x=2, y=2)
    expected_slices = [
        {"x": slice(0, 25), "y": slice(0, 40)},
        {"x": slice(0, 25), "y": slice(40, 60)},
        {"x": slice(25, 45), "y": slice(0, 40)},
        {"x": slice(25, 45), "y": slice(40, 60)},
    ]
    assert len(slices) == len(expected_slices), f"Expected {len(expected_slices)} slices, got {len(slices)}"
    for s in slices:
        assert s in expected_slices, f"Slice {s} not in expected slices"

    # Test when group_size is larger than number of chunks
    slices = get_block_slices(ds, x=5)
    expected_slices = [{"x": slice(0, 45)}]
    assert len(slices) == 1, f"Expected 1 slice, got {len(slices)}"
    assert slices[0] == expected_slices[0], f"Expected {expected_slices[0]}, got {slices[0]}"

    # Test raise error when no dimension specified
    with pytest.raises(ValueError):
        get_block_slices(ds)


def test_get_block_slices_invalid_dimension():
    # Create a sample Dataset
    data = np.zeros((10, 10))
    ds = xr.Dataset({"var": (("a", "b"), data)})

    # Test with an invalid dimension
    with pytest.raises(KeyError):
        get_block_slices(ds, x=2)
