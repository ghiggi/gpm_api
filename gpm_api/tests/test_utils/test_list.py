import pytest

from gpm_api.utils.list import flatten_list


def test_flatten_list() -> None:
    """Test flattening nested lists into lists"""

    assert flatten_list([["single item"]]) == ["single item"]
    assert flatten_list([["double", "item"]]) == ["double", "item"]
    assert flatten_list([]) == [], "Empty list should return empty list"
    assert flatten_list(["single item"]) == ["single item"], "Flat list should return same list"
    assert flatten_list(["double", "item"]) == [
        "double",
        "item",
    ], "Flat list should return same list"
