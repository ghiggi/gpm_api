import pytest
import numpy as np

from gpm_api.utils import slices as gpm_slices


def test_get_list_slices_from_indices() -> None:
    """Test get_list_slices_from_indices"""

    # Check expected behavior
    indices = [0, 1, 2, 4, 5, 8]
    expected_slices = [slice(0, 3), slice(4, 6), slice(8, 9)]
    returned_slices = gpm_slices.get_list_slices_from_indices(indices)
    assert returned_slices == expected_slices

    # Test with single index
    assert gpm_slices.get_list_slices_from_indices(0) == [slice(0, 1)]
    assert gpm_slices.get_list_slices_from_indices(0.0) == [slice(0, 1)]

    # Test with empty list
    assert gpm_slices.get_list_slices_from_indices([]) == []

    # Test negative indices
    indices = [-1, 0]
    with pytest.raises(ValueError):
        gpm_slices.get_list_slices_from_indices(indices)


def test_get_indices_from_list_slices() -> None:
    """Test get_indices_from_list_slices"""

    # Check expected behavior
    slices = [slice(0, 3), slice(4, 6), slice(8, 9)]
    expected_indices = [0, 1, 2, 4, 5, 8]
    returned_indices = gpm_slices.get_indices_from_list_slices(slices)
    assert np.array_equal(returned_indices, expected_indices)

    # Test with intersecting slices
    slices = [slice(0, 3), slice(1, 6), slice(8, 9)]
    expected_indices = [0, 1, 2, 3, 4, 5, 8]
    returned_indices = gpm_slices.get_indices_from_list_slices(slices, check_non_intersecting=False)
    assert np.array_equal(returned_indices, expected_indices)

    # Test with intersection check
    with pytest.raises(ValueError):
        gpm_slices.get_indices_from_list_slices(slices, check_non_intersecting=True)

    # Test with empty list
    assert gpm_slices.get_indices_from_list_slices([]).size == 0


def test_list_slices_intersection() -> None:
    """Test list_slices_intersection"""

    slices = [
        [slice(0, 3), slice(4, 7)],  # 0 1 2|  4 5 6|
        [slice(2, 5), slice(5, 8)],  #     2 3 4|5 6 7
    ]  #                 intersection:     2|3 4|5 6|
    # Common indices: 2, 4, 5, 6
    expected_list = [slice(2, 3), slice(4, 5), slice(5, 7)]
    returned_list = gpm_slices.list_slices_intersection(*slices)
    assert returned_list == expected_list

    # Hole in one list: should not be patched
    slices = [
        [slice(0, 10)],
        [slice(0, 5), slice(5, 10)],
    ]
    expected_list = [slice(0, 5), slice(5, 10)]
    returned_list = gpm_slices.list_slices_intersection(*slices)
    assert returned_list == expected_list

    # Repeated slices
    slices = [
        [slice(0, 3), slice(3, 6)],
        [slice(0, 3), slice(3, 6)],
    ]
    expected_list = [slice(0, 3), slice(3, 6)]
    returned_list = gpm_slices.list_slices_intersection(*slices)
    assert returned_list == expected_list

    # Non-intersecting lists
    slices = [
        [slice(0, 3)],
        [slice(4, 7)],
    ]
    expected_list = []
    returned_list = gpm_slices.list_slices_intersection(*slices)
    assert returned_list == expected_list

    # Test with empty list
    assert gpm_slices.list_slices_intersection() == []


def test_list_slices_union() -> None:
    """Test list_slices_union"""

    slices = [
        [slice(0, 3), slice(4, 7)],
        [slice(4, 7), slice(7, 10)],
    ]
    expected_list = [slice(0, 3), slice(4, 10)]
    returned_list = gpm_slices.list_slices_union(*slices)
    assert returned_list == expected_list

    # Test with one empty list
    slices[1] = []
    expected_list = [slice(0, 3), slice(4, 7)]
    returned_list = gpm_slices.list_slices_union(*slices)
    assert returned_list == expected_list


def test_list_slices_difference() -> None:
    """Test list_slices_difference"""

    # Base cases
    assert gpm_slices.list_slices_difference([slice(3, 6)], [slice(1, 2)]) == [slice(3, 6)]
    assert gpm_slices.list_slices_difference([slice(3, 6)], [slice(1, 3)]) == [slice(3, 6)]
    assert gpm_slices.list_slices_difference([slice(3, 6)], [slice(1, 4)]) == [slice(4, 6)]
    assert gpm_slices.list_slices_difference([slice(3, 6)], [slice(1, 8)]) == []
    assert gpm_slices.list_slices_difference([slice(3, 6)], [slice(5, 8)]) == [slice(3, 5)]
    assert gpm_slices.list_slices_difference([slice(3, 6)], [slice(6, 8)]) == [slice(3, 6)]
    assert gpm_slices.list_slices_difference([slice(3, 6)], [slice(7, 8)]) == [slice(3, 6)]
    assert gpm_slices.list_slices_difference([slice(3, 6)], [slice(4, 5)]) == [
        slice(3, 4),
        slice(5, 6),
    ]

    # List of slices
    slices = [
        [slice(0, 3), slice(4, 7)],
        [slice(4, 7), slice(7, 10)],
    ]
    expected_list = [slice(0, 3)]
    returned_list = gpm_slices.list_slices_difference(*slices)
    assert returned_list == expected_list

    # Test with one empty list
    slices = [
        [slice(0, 3), slice(4, 7)],
        [],
    ]
    expected_list = [slice(0, 3), slice(4, 7)]
    returned_list = gpm_slices.list_slices_difference(*slices)
    assert returned_list == expected_list

    slices = [
        [],
        [slice(0, 3)],
    ]
    expected_list = []
    returned_list = gpm_slices.list_slices_difference(*slices)
    assert returned_list == expected_list

    # Hole in one list: should not be patched
    slices = [
        [slice(0, 10)],
        [slice(0, 5), slice(5, 10)],
    ]
    expected_list = [slice(5, 5)]
    returned_list = gpm_slices.list_slices_difference(*slices)


def test_list_slices_combine() -> None:
    """Test list_slices_combine"""

    slices = [
        [slice(0, 3), slice(4, 7)],
        [slice(4, 7), slice(7, 10)],
    ]
    expected_list = [slice(0, 3), slice(4, 7), slice(4, 7), slice(7, 10)]
    returned_list = gpm_slices.list_slices_combine(*slices)
    assert returned_list == expected_list

    # Test with one empty list
    slices[1] = []
    expected_list = [slice(0, 3), slice(4, 7)]
    returned_list = gpm_slices.list_slices_combine(*slices)
    assert returned_list == expected_list


def test_list_slices_simplify() -> None:
    """Test list_slices_simplify"""

    slices = [slice(0, 3), slice(4, 7), slice(4, 7), slice(7, 10)]
    expected_list = [slice(0, 3), slice(4, 10)]
    returned_list = gpm_slices.list_slices_simplify(slices)
    assert returned_list == expected_list

    # Test with single slice
    slices = [slice(0, 3)]
    expected_list = [slice(0, 3)]
    returned_list = gpm_slices.list_slices_simplify(slices)
    assert returned_list == expected_list


def test_list_slices_sort() -> None:
    """Test list_slices_sort"""

    slices = [
        [slice(1, 3)],
        [slice(0, 10), slice(2, 4)],
    ]
    expected_list = [slice(0, 10), slice(1, 3), slice(2, 4)]
    returned_list = gpm_slices.list_slices_sort(*slices)
    assert returned_list == expected_list


def test_list_slices_filter() -> None:
    """Test list_slices_filter"""

    slices = [slice(0, 1), slice(0, 10), slice(0, 20)]

    # Test without filter
    expected_list = slices
    returned_list = gpm_slices.list_slices_filter(slices)
    assert returned_list == expected_list

    # Test minimum length
    min_size = 10
    expected_list = [slice(0, 10), slice(0, 20)]
    returned_list = gpm_slices.list_slices_filter(slices, min_size=min_size)
    assert returned_list == expected_list

    # Test maximum length
    max_size = 10
    expected_list = [slice(0, 1), slice(0, 10)]
    returned_list = gpm_slices.list_slices_filter(slices, max_size=max_size)
    assert returned_list == expected_list


def test_list_slices_flatten() -> None:
    """Test list_slices_flatten"""

    slices = [[slice(1, 7934)], [slice(1, 2), slice(3, 4)]]
    expected_list = [slice(1, 7934), slice(1, 2), slice(3, 4)]
    returned_list = gpm_slices.list_slices_flatten(slices)
    assert returned_list == expected_list

    # Test with already flat list
    slices = expected_list
    returned_list = gpm_slices.list_slices_flatten(slices)
    assert returned_list == expected_list


def test_get_list_slices_from_bool_arr() -> None:
    """Test get_list_slices_from_bool_arr"""

    #             0      1     2     3      4      5     6
    bool_array = [False, True, True, False, False, True, False]

    # Test default behavior
    returned_list = gpm_slices.get_list_slices_from_bool_arr(
        bool_array, include_false=True, skip_consecutive_false=True
    )
    expected_list = [slice(1, 4), slice(5, 7)]
    assert returned_list == expected_list

    # Test without skipping consecutive False, creating one empty slice per False value
    returned_list = gpm_slices.get_list_slices_from_bool_arr(
        bool_array, include_false=True, skip_consecutive_false=False
    )
    expected_list = [slice(0, 1), slice(1, 4), slice(4, 5), slice(5, 7)]
    assert returned_list == expected_list

    # Test without including final False values, making slices shorter by one
    returned_list = gpm_slices.get_list_slices_from_bool_arr(bool_array, include_false=False)
    expected_list = [slice(1, 3), slice(5, 6)]
    assert returned_list == expected_list

    # With final element True
    #             0      1     2     3      4      5
    bool_array = [False, True, True, False, False, True]
    returned_list = gpm_slices.get_list_slices_from_bool_arr(
        bool_array, include_false=True, skip_consecutive_false=True
    )
    expected_list = [slice(1, 4), slice(5, 6)]
    assert returned_list == expected_list

    # Test all True values
    bool_array = [True, True, True]
    returned_list = gpm_slices.get_list_slices_from_bool_arr(bool_array)
    expected_list = [slice(0, 3)]
    assert returned_list == expected_list

    # Test all False values
    bool_array = [False, False, False]
    returned_list = gpm_slices.get_list_slices_from_bool_arr(
        bool_array, skip_consecutive_false=True
    )
    expected_list = []
    assert returned_list == expected_list

    returned_list = gpm_slices.get_list_slices_from_bool_arr(
        bool_array, skip_consecutive_false=False
    )
    expected_list = [slice(0, 1), slice(1, 2), slice(2, 3)]
    assert returned_list == expected_list


def test_ensure_is_slice() -> None:
    """Test ensure_is_slice"""

    # Test case for an integer input
    assert gpm_slices.ensure_is_slice(5) == slice(5, 6)

    # Test case for a slice input
    assert gpm_slices.ensure_is_slice(slice(2, 4)) == slice(2, 4)

    # Test case for a list with one element input
    assert gpm_slices.ensure_is_slice([3]) == slice(3, 4)

    # Test case for a tuple with one element input
    assert gpm_slices.ensure_is_slice((7,)) == slice(7, 8)

    # Test case for a numpy array with single element input
    assert gpm_slices.ensure_is_slice(np.array([10])) == slice(10, 11)

    # Test case for unsupported input types
    with pytest.raises(ValueError):
        gpm_slices.ensure_is_slice("unsupported")

    with pytest.raises(ValueError):
        gpm_slices.ensure_is_slice([1, 2, 3])


def test_get_slice_size() -> None:
    """Test get_slice_size"""

    assert gpm_slices.get_slice_size(slice(1, 10)) == 9

    # Test non-slice input
    with pytest.raises(TypeError):
        gpm_slices.get_slice_size(5)


def test_pad_slice():
    """Test pad_slice"""

    # Always step = 1

    test_slice = slice(2, 8)
    pad = 2
    expected_slice = slice(0, 10)
    assert gpm_slices.pad_slice(test_slice, pad) == expected_slice

    # With min and max set
    min_start = 1
    max_stop = 9
    expected_slice = slice(1, 9)
    assert gpm_slices.pad_slice(test_slice, pad, min_start, max_stop) == expected_slice


def test_pad_slices():
    """Test pad_slices"""

    test_slices = [slice(2, 8), slice(1, 9)]

    # Integer padding and shape
    pad = 2
    shape = 20
    expected_slices = [slice(0, 10), slice(0, 11)]
    assert gpm_slices.pad_slices(test_slices, pad, shape) == expected_slices

    # Truncating with integer shape
    shape = 10
    expected_slices = [slice(0, 10), slice(0, 10)]
    assert gpm_slices.pad_slices(test_slices, pad, shape) == expected_slices

    # Tuple padding
    pad = (2, 3)
    shape = 20
    expected_slices = [slice(0, 10), slice(0, 12)]
    assert gpm_slices.pad_slices(test_slices, pad, shape) == expected_slices

    # Tuple shape
    pad = 2
    shape = (9, 10)
    expected_slices = [slice(0, 9), slice(0, 10)]
    assert gpm_slices.pad_slices(test_slices, pad, shape) == expected_slices

    # Invalid tuple sizes
    pad = (2, 3, 4)
    shape = 10
    with pytest.raises(ValueError):
        gpm_slices.pad_slices(test_slices, pad, shape)

    pad = 2
    shape = (9, 10, 11)
    with pytest.raises(ValueError):
        gpm_slices.pad_slices(test_slices, pad, shape)


def test_enlarge_slice():
    """Test enlarge_slice"""

    test_slice = slice(3, 5)
    min_start = 1
    max_stop = 10

    # No change
    # 1 2 3 4 5 6 7 8 9 10
    #     |---|
    min_size = 1
    expected_slice = slice(3, 5)
    assert gpm_slices.enlarge_slice(test_slice, min_size, min_start, max_stop) == expected_slice

    # Enlarge one side
    # 1 2 3 4 5 6 7 8 9 10
    #     |---|
    #   |-----|
    min_size = 3
    expected_slice = slice(2, 5)
    assert gpm_slices.enlarge_slice(test_slice, min_size, min_start, max_stop) == expected_slice

    # Enlarge both sides
    # 1 2 3 4 5 6 7 8 9 10
    #     |---|
    #   |-------|
    min_size = 4
    expected_slice = slice(2, 6)
    assert gpm_slices.enlarge_slice(test_slice, min_size, min_start, max_stop) == expected_slice

    # Enlarge reaching min_start
    # 1 2 3 4 5 6 7 8 9 10
    #     |---|
    # |---------------|
    min_size = 8
    expected_slice = slice(1, 9)
    assert gpm_slices.enlarge_slice(test_slice, min_size, min_start, max_stop) == expected_slice

    # Enlarge reaching max_stop
    # 1 2 3 4 5 6 7 8 9 10
    #           |---|
    #   |---------------|
    test_slice = slice(6, 8)
    expected_slice = slice(2, 10)
    assert gpm_slices.enlarge_slice(test_slice, min_size, min_start, max_stop) == expected_slice

    # Enlarge too much
    min_size = 20
    with pytest.raises(ValueError):
        gpm_slices.enlarge_slice(test_slice, min_size, min_start, max_stop)


def test_enlarge_slices():
    """Test enlarge_slices"""

    test_slices = [slice(3, 5), slice(6, 8)]

    # Integer min_size and shape
    min_size = 4
    shape = 10
    expected_slices = [slice(2, 6), slice(5, 9)]
    assert gpm_slices.enlarge_slices(test_slices, min_size, shape) == expected_slices

    # Capping with integer shape
    shape = 8
    expected_slices = [slice(2, 6), slice(4, 8)]
    assert gpm_slices.enlarge_slices(test_slices, min_size, shape) == expected_slices

    # Tuple min_size
    min_size = (4, 6)
    shape = 10
    expected_slices = [slice(2, 6), slice(4, 10)]
    assert gpm_slices.enlarge_slices(test_slices, min_size, shape) == expected_slices

    # Tuple shape
    min_size = 4
    shape = (5, 8)
    expected_slices = [slice(1, 5), slice(4, 8)]
    assert gpm_slices.enlarge_slices(test_slices, min_size, shape) == expected_slices

    # Invalid tuple sizes
    min_size = (4, 5, 6)
    shape = 10
    with pytest.raises(ValueError):
        gpm_slices.enlarge_slices(test_slices, min_size, shape)

    min_size = 4
    shape = (8, 9, 10)
    with pytest.raises(ValueError):
        gpm_slices.enlarge_slices(test_slices, min_size, shape)
