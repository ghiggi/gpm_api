from gpm_api.dataset.decoding import attrs as at
import pytest
import numpy as np


def test_convert_string_to_number() -> None:
    """Test that a string is converted to a number"""

    assert at.convert_string_to_number("1") == 1
    assert at.convert_string_to_number("1.0") == 1.0
    assert at.convert_string_to_number("1.0e-3") == 1.0e-3
    assert at.convert_string_to_number("1.0e3") == 1.0e3
    assert at.convert_string_to_number("1.0e+3") == 1.0e3
    assert at.convert_string_to_number("1.0e+03") == 1.0e3
    assert at.convert_string_to_number("-999") == -999

    with pytest.raises(ValueError):
        assert at.convert_string_to_number("notanumber")


def test_ensure_dtype_name() -> None:
    """Test that a dtype is returned as a string name"""

    # Test with dtype
    assert at.ensure_dtype_name(np.dtype("float32")) == "float32"
    assert at.ensure_dtype_name(np.dtype("int32")) == "int32"
    assert at.ensure_dtype_name(np.dtype("uint8")) == "uint8"

    # Test normal string
    assert at.ensure_dtype_name("notadtype") == "notadtype"

    # Test not a dtype
    with pytest.raises(TypeError):
        assert at.ensure_dtype_name(np.dtype("float31"))
