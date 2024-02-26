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
"""This module defines pytest fixtures used for the testing of GPM-API Dataset."""

import numpy as np
import pytest


class SaneEqualityArray(np.ndarray):
    """Wrapper class for numpy array allowing deep equality tests on objects containing numpy arrays.

    From https://stackoverflow.com/a/14276901
    """

    def __new__(cls, array):
        """Create a new SaneEqualityArray from array only (instead of shape + type + array)."""

        if isinstance(array, list):  # No need to wrap regular lists
            return array

        return np.asarray(array).view(cls)

    def __eq__(self, other):
        # Only use equal_nan for floats dtypes
        equal_nan = np.issubdtype(self.dtype, np.floating)
        return (
            isinstance(other, np.ndarray)
            and self.shape == other.shape
            and np.array_equal(self, other, equal_nan=equal_nan)
        )


def pytest_configure():
    pytest.SaneEqualityArray = SaneEqualityArray
