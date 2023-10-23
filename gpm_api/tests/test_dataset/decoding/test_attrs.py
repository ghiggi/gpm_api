from gpm_api.dataset.decoding.dataarray_attrs import (
    convert_string_to_number,
    ensure_dtype_name,
)
import pytest
import numpy as np


def test_convert_string_to_number() -> None:
    """Test that a string is converted to a number"""

    assert convert_string_to_number("1") == 1
    assert convert_string_to_number("1.0") == 1.0
    assert convert_string_to_number("1.0e-3") == 1.0e-3
    assert convert_string_to_number("1.0e3") == 1.0e3
    assert convert_string_to_number("1.0e+3") == 1.0e3
    assert convert_string_to_number("1.0e+03") == 1.0e3
    assert convert_string_to_number("-999") == -999

    with pytest.raises(ValueError):
        assert convert_string_to_number("notanumber")


def test_ensure_dtype_name() -> None:
    """Test that a dtype is returned as a string name"""

    # Test with dtype
    assert ensure_dtype_name(np.dtype("float32")) == "float32"
    assert ensure_dtype_name(np.dtype("int32")) == "int32"
    assert ensure_dtype_name(np.dtype("uint8")) == "uint8"

    # Test normal string
    assert ensure_dtype_name("notadtype") == "notadtype"

    # Test not a dtype
    with pytest.raises(TypeError):
        assert ensure_dtype_name(np.dtype("float31"))
