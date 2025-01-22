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
"""This module test the PMW utilities."""
import numpy as np
import pytest

from gpm.utils.pmw import PMWFrequency, find_polarization_pairs


def test_pmwfrequency_init():
    """Test initialization from direct numeric/str arguments."""
    freq = PMWFrequency(center_frequency=10.65, polarization="V", offset=None)
    assert freq.center_frequency == 10.65
    assert freq.polarization == "V"
    assert freq.offset is None


def test_pmwfrequency_init_offset():
    """Test initialization with an offset."""
    freq = PMWFrequency(183, "H", 3)
    assert freq.center_frequency == 183
    assert freq.polarization == "H"
    assert freq.offset == 3


def test_pmwfrequency_invalid_polarization():
    """Test initialization raises ValueError for invalid polarization."""
    with pytest.raises(ValueError) as excinfo:
        PMWFrequency(center_frequency=89, polarization="RCP", offset=5)
    assert "Invalid polarization 'RCP'" in str(excinfo.value)


def test_pmwfrequency_from_string_invalid():
    """Test classmethod from_string with invalid input."""
    with pytest.raises(TypeError):
        PMWFrequency.from_string(1)  # invalid type

    with pytest.raises(ValueError):
        PMWFrequency.from_string("notAValidString")  # invalid string


def test_pmwfrequency_from_string_with_numpy_str():
    """Test classmethod from_string with numpy str type."""
    string = np.array(["10.65V", "10.65V"], dtype=str)[0]
    freq = PMWFrequency.from_string(string)
    assert freq.center_frequency == pytest.approx(10.65, abs=1e-9)
    assert freq.polarization == "V"
    assert freq.offset is None


def test_pmwfrequency_from_string_no_offset():
    """Test classmethod from_string with no offset."""
    freq = PMWFrequency.from_string("10.65V")
    assert freq.center_frequency == pytest.approx(10.65, abs=1e-9)
    assert freq.polarization == "V"
    assert freq.offset is None


def test_pmwfrequency_from_string_with_offset():
    """Test classmethod from_string with offset."""
    freq = PMWFrequency.from_string("183H3")
    assert freq.center_frequency == 183
    assert freq.polarization == "H"
    assert freq.offset == 3


def test_pmwfrequency_from_string_qv():
    """Test parsing QV/QH polarization."""
    freq = PMWFrequency.from_string("89QV7.5")
    assert freq.center_frequency == pytest.approx(89, abs=1e-9)
    assert freq.polarization == "QV"
    assert freq.offset == pytest.approx(7.5, abs=1e-9)


def test_title_no_offset():
    """Test title method without offset."""
    freq = PMWFrequency(10.65, "V")
    assert freq.title() == "10.65 GHz (V)"


def test_title_with_offset():
    """Test title method with offset."""
    freq = PMWFrequency(183, "H", 3)
    assert freq.title() == "183 ± 3 GHz (H)"


def test_opposite_polarization_standard():
    """Test opposite_polarization for V/H."""
    freq_v = PMWFrequency(10.65, "V")
    freq_h = freq_v.opposite_polarization()
    assert freq_h is not None
    assert freq_h.center_frequency == pytest.approx(10.65)
    assert freq_h.polarization == "H"
    assert freq_h.offset is None


def test_opposite_polarization_qv_qh():
    """Test opposite_polarization for QV/QH."""
    freq_qv = PMWFrequency(89, "QV", 7.5)
    freq_qh = freq_qv.opposite_polarization()
    assert freq_qh is not None
    assert freq_qh.center_frequency == 89
    assert freq_qh.polarization == "QH"
    assert freq_qh.offset == 7.5


def test_has_same_center_frequency_true():
    """Test has_same_center_frequency returns True within tolerance."""
    freq1 = PMWFrequency(10.6500000001, "V")
    freq2 = PMWFrequency(10.6500000002, "H")
    assert freq1.has_same_center_frequency(freq2, tol=1e-6)  # well within 1e-6


def test_has_same_center_frequency_false():
    """Test has_same_center_frequency returns False outside tolerance."""
    freq1 = PMWFrequency(10.65, "V")
    freq2 = PMWFrequency(10.66, "V")
    assert not freq1.has_same_center_frequency(freq2, tol=1e-6)


def test_wavelength_property():
    """Test wavelength property calculation."""
    freq = PMWFrequency(10.0, "H")  # 10 GHz
    # Wavelength = c / f = 3e8 / (10 * 1e9) = 0.03 m
    assert freq.wavelength == pytest.approx(0.03, abs=1e-9)


def test_equality_basic():
    """Test __eq__ for two identical frequencies (no offset)."""
    freq1 = PMWFrequency(10.65, "V")
    freq2 = PMWFrequency(10.65, "V")
    assert freq1 == freq2


def test_equality_different_init():
    freq1 = PMWFrequency(10.65, "V")
    freq2 = PMWFrequency.from_string("10.65V")
    assert freq1 == freq2


def test_equality_offset():
    """Test __eq__ with offset, within tolerance."""
    freq1 = PMWFrequency(183.0, "V", 3.0)
    freq2 = PMWFrequency(183.0, "V", 3.0000000001)
    assert freq1 == freq2


def test_equality_polarization_mismatch():
    """Test __eq__ returns False for polarization mismatch."""
    freq1 = PMWFrequency(10.65, "V")
    freq2 = PMWFrequency(10.65, "H")
    assert freq1 != freq2


def test_equality_offset_none_vs_zero():
    """Test __eq__ treats offset=None as offset=0.0."""
    freq1 = PMWFrequency(10.65, "V", None)
    freq2 = PMWFrequency(10.65, "V", 0.0)
    assert freq1 == freq2


def test_repr_no_offset():
    """Test __repr__ output with no offset."""
    freq = PMWFrequency(10.65, "H")
    r = repr(freq)
    assert "<PMWFrequency: 10.65 GHz (H)>" in r


def test_repr_with_offset():
    """Test __repr__ output with offset."""
    freq = PMWFrequency(183, "V", 3)
    r = repr(freq)
    assert "<PMWFrequency: 183 ± 3 GHz (V)>" in r


def test_to_string_no_offset():
    """Test to_string method without offset."""
    freq = PMWFrequency(10.65, "V")
    assert freq.to_string() == "10.65V"


def test_to_string_with_integer_offset():
    """Test to_string method with integer offset."""
    freq = PMWFrequency(183, "H", 3)
    assert freq.to_string() == "183H3"


def test_to_string_with_float_offset():
    """Test to_string method with float offset."""
    freq = PMWFrequency(89, "QV", 7.5)
    assert freq.to_string() == "89QV7.5"


def test_to_string_zero_offset():
    """Test to_string method with offset zero."""
    freq = PMWFrequency(165, "V", 0.0)
    assert freq.to_string() == "165V"


def test_to_string_no_decimal():
    """Test to_string method with no decimal in frequency."""
    freq = PMWFrequency(23, "H")
    assert freq.to_string() == "23H"


def test_to_string_decimal_frequency():
    """Test to_string method with decimal frequency."""
    freq = PMWFrequency(36.5, "H")
    assert freq.to_string() == "36.5H"


def test_to_string_offset_none_vs_zero():
    """Test to_string method consider offset None or zero equivalent."""
    freq_with_none = PMWFrequency(165, "V", None)
    freq_with_zero = PMWFrequency(165, "V", 0.0)
    assert freq_with_none.to_string() == "165V"
    assert freq_with_zero.to_string() == "165V"


def test_to_string_leading_trailing_zeros():
    """Test to_string method removes unnecessary trailing zeros."""
    freq = PMWFrequency(183.0, "V", 3.0)
    assert freq.to_string() == "183V3"

    freq = PMWFrequency(10.6500, "H")
    assert freq.to_string() == "10.65H"


def test_to_string_complex_polarization():
    """Test to_string method with complex polarization."""
    freq = PMWFrequency(183.31, "QH", 7.5)
    assert freq.to_string() == "183.31QH7.5"


####----------------------------------------------------------------------------------------------------.


def test_find_polarization_pairs():
    """Test identify channels with polarization pairs."""
    pmw_frequencies_str = ["10.65V", "10.65H", "36.5QH", "36.5QV", "18.7V", "183V3", "183H7"]
    pmw_frequencies = [PMWFrequency.from_string(freq) for freq in pmw_frequencies_str]
    dict_pairs = find_polarization_pairs(pmw_frequencies)
    # Assert valid couples are found
    assert "10.65" in dict_pairs
    assert "36.5" in dict_pairs
    assert len(dict_pairs) == 2
    # Assert first freq is vertically polarized
    assert dict_pairs["10.65"][0].polarization == "V"
    assert dict_pairs["36.5"][0].polarization == "QV"
