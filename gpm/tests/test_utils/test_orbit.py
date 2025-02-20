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
"""This module tests the dataframe utilities functions."""
import numpy as np
import pytest
from gpm.utils.orbit import adjust_short_sequences


class TestAdjustShortSequences:
    """Test cases for the adjust_short_sequences function."""

    def test_constant_array(self):
        """Return unchanged array when all elements are identical."""
        arr = [5, 5, 5, 5]
        result = adjust_short_sequences(arr, min_size=2)
        expected = np.array([5, 5, 5, 5])
        np.testing.assert_array_equal(result, expected)

    def test_no_change_for_long_sequences(self):
        """Return unchanged array when all sequences meet min_size."""
        arr = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        result = adjust_short_sequences(arr, min_size=2)
        expected = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        np.testing.assert_array_equal(result, expected)

    def test_replace_short_sequence_middle(self):
        """Replace a short middle sequence with the previous sequence value."""
        # Here the sequence [2] is of length 1 (< min_size=2) and should be replaced by 1.
        arr = [1, 1, 2, 3, 3, 3]
        result = adjust_short_sequences(arr, min_size=2)
        expected = np.array([1, 1, 1, 3, 3, 3])
        np.testing.assert_array_equal(result, expected)

    def test_replace_short_sequence_start(self):
        """Replace a short starting sequence with the following sequence value."""
        # The first sequence [2] is short (< min_size=2) and should be replaced by 3.
        arr = [2, 3, 3, 3]
        result = adjust_short_sequences(arr, min_size=2)
        expected = np.array([3, 3, 3, 3])
        np.testing.assert_array_equal(result, expected)

    def test_replace_short_sequence_end(self):
        """Replace a short ending sequence with the previous sequence value."""
        # The ending sequence [2] is short (< min_size=2) and should be replaced by 1.
        arr = [1, 1, 1, 2]
        result = adjust_short_sequences(arr, min_size=2)  
        expected = np.array([1, 1, 1, 1])
        np.testing.assert_array_equal(result, expected)

    def test_min_size_one(self):
        """Return unchanged array when min_size is 1 (all sequences allowed)."""
        arr = [1, 2, 3, 4]
        result = adjust_short_sequences(arr, min_size=1)  
        np.testing.assert_array_equal(result, arr)

    def test_non_1d_input(self):
        """Raise ValueError for non 1D array input."""
        arr = [[1, 1, 1], [2, 2, 2]]
        with pytest.raises(ValueError):
            adjust_short_sequences(arr, min_size=2)
        
    @pytest.mark.parametrize("arr,expected", [
        ( 
            [1, -1, -1, -1, 1, 1, 1, 1, -1], # Short sequences at the edges
            [-1, -1, -1, -1, 1, 1, 1, 1, 1]
        ), 
        ( 
            [1, -1, -1, -1, 1, 1, 1, 1, -1, -1], # Short sequence at start  
            [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1]
        ),
        (
            [1, 1, -1, 1, 1, 1, 1, 1, -1], #Short sequence in the middle and at the end 
            [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ),
    ])
    def test_edge_cases(self, arr, expected):
        """Test various edge cases with short sequences at the edges."""
        result = adjust_short_sequences(arr, min_size=2)
        np.testing.assert_array_equal(result, np.array(expected))
