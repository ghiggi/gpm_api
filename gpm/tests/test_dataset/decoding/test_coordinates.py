import pytest

from gpm.dataset.decoding.coordinates import get_pmw_frequency_corra
from gpm.utils.pmw import get_pmw_frequency, get_pmw_frequency_dict


def test_get_pmw_frequency_dict() -> None:
    """Test that a dictionary is returned."""
    res = get_pmw_frequency_dict()

    assert isinstance(res, dict), "Dictionary not returned"


def test_get_pmw_frequency_corra(
    products: list[str],
) -> None:
    # Try GPM CORRA Product
    res = get_pmw_frequency_corra("2B-GPM-CORRA")
    assert len(res) > 0
    assert res == (get_pmw_frequency("GMI", scan_mode="S1") + get_pmw_frequency("GMI", scan_mode="S2"))

    # Try TRMM CORRA Product
    res = get_pmw_frequency_corra("2B-TRMM-CORRA")
    assert len(res) > 0
    assert res == (
        get_pmw_frequency("TMI", scan_mode="S1")
        + get_pmw_frequency("TMI", scan_mode="S2")
        + get_pmw_frequency("TMI", scan_mode="S3")
    )

    # Test other non-corra products fail
    for product in products:
        if "corra" not in product.lower():
            with pytest.raises(ValueError):
                get_pmw_frequency_corra(product)
