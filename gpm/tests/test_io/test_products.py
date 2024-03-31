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
"""This module test the product info routines."""


import pytest

from gpm.io.products import (
    _get_sensor_satellite_names,
    available_product_categories,
    available_product_levels,
    available_products,
    available_satellites,
    available_sensors,
    available_versions,
    get_info_dict,
    get_info_dict_subset,
    get_product_category,
    get_product_info,
    get_product_level,
    is_gpm_api_product,
    is_trmm_product,
)


def test_get_product_category(
    products: list[str],
    product_categories: list[str],
) -> None:
    """Test that the product category is in the list of product categories."""
    for product in products:
        assert get_product_category(product) in product_categories

    # Add value to info dict to force a ValueError on None return
    get_info_dict()["fake_product"] = {"product_category": None}
    with pytest.raises(ValueError):
        get_product_category("fake_product")

    get_info_dict().pop("fake_product")  # Remove fake value


def test_get_product_info():
    """Test get_product_info."""

    # Test with non string input
    with pytest.raises(TypeError):
        get_product_info(123)

    # Test with an invalid product.
    with pytest.raises(ValueError):
        get_product_info("invalid_product")

    # Test return the info dictionary
    assert isinstance(get_product_info("2A-DPR"), dict)


@pytest.mark.parametrize("full", [True, False])
def test_get_product_level(full):
    """Get the product_level of a GPM product."""

    results_dict = {
        "1B-Ka": {"full": "1B", "short": "1B"},
        "1C-GMI-R": {"full": "1C-R", "short": "1C"},
        "2A-DPR": {"full": "2A", "short": "2A"},
        "2A-GPM-SLH": {"full": "2A", "short": "2A"},
        "2A-ENV-Ka": {"full": "2A-ENV", "short": "2A"},
        "2A-GMI-CLIM": {"full": "2A-CLIM", "short": "2A"},
        "2B-GPM-CORRA": {"full": "2B", "short": "2B"},
        "IMERG-ER": {"full": "3B-HHR-E", "short": "3B"},
        "IMERG-LR": {"full": "3B-HHR-L", "short": "3B"},
        "IMERG-FR": {"full": "3B-HHR", "short": "3B"},
        #   3B-HH3, 3B-DAY
    }
    for product, results in results_dict.items():
        expected_product_level = results["full"] if full else results["short"]
        product_level = get_product_level(product, full=full)
        assert product_level == expected_product_level


class TestGetInfoDictSubset:
    def test_list_of_values(
        self,
        sensors,
        satellites,
        product_categories,
        product_types,
        versions,
        full_product_levels,
        product_levels,
    ):
        """Test get_info_dict_subset with a list of values."""
        # Test individual arguments (list of values)
        list_kwargs = [
            {"sensors": sensors},
            {"satellites": satellites},
            {"product_categories": product_categories},
            {"product_types": product_types},
            {"versions": versions},
            {"full_product_levels": full_product_levels},
            {"product_levels": product_levels},
        ]
        for kwargs in list_kwargs:
            result = get_info_dict_subset(**kwargs)
            assert isinstance(result, dict)

    def test_single_value(
        self,
        sensors,
        satellites,
        product_categories,
        product_types,
        versions,
        full_product_levels,
        product_levels,
    ):
        """Test get_info_dict_subset with a single value."""
        list_kwargs = [
            {"sensors": sensors},
            {"satellites": satellites},
            {"product_categories": product_categories},
            {"product_types": product_types},
            {"versions": versions},
            {"full_product_levels": full_product_levels},
            {"product_levels": product_levels},
        ]
        list_kwargs = [{key: values[0]} for kwargs in list_kwargs for key, values in kwargs.items()]
        for kwargs in list_kwargs:
            result = get_info_dict_subset(**kwargs)
            assert isinstance(result, dict)

    def test_bad_value(self):
        """Test get_info_dict_subset with bad arguments"""
        list_kwargs = [
            {"sensors": "BAD"},
            {"satellites": "BAD"},
            {"product_categories": "BAD"},
            {"product_types": "BAD"},
            {"versions": -1},
            {"full_product_levels": "BAD"},
            {"product_levels": "BAD"},
        ]
        for kwargs in list_kwargs:
            with pytest.raises(ValueError):
                get_info_dict_subset(**kwargs)


def test_get_sensor_satellite_names():
    """Test get_sensor_satellite_names function."""
    info_dict = {
        "2A-SSMIS-F18-CLIM": {
            "pattern": "2A-CLIM.F18.SSMIS.*",
            "sensor": "SSMIS",
            "satellite": "F18",
        },
    }

    assert ["SSMIS"] == _get_sensor_satellite_names(info_dict, key="sensor", combine_with=None)
    assert ["F18"] == _get_sensor_satellite_names(info_dict, key="satellite", combine_with=None)
    assert ["SSMIS-F18"] == _get_sensor_satellite_names(
        info_dict,
        key="satellite",
        combine_with="sensor",
    )
    assert ["SSMIS-F18"] == _get_sensor_satellite_names(
        info_dict,
        key="sensor",
        combine_with="satellite",
    )


@pytest.mark.parametrize("full", [True, False])
def test_available_product_levels(full):
    """Test available_product_levels() function."""
    result = available_product_levels(full=full)
    assert isinstance(result, list)
    assert len(result) > 0


@pytest.mark.parametrize("prefix_with_sensor", [True, False])
def test_available_satellites(prefix_with_sensor):
    """Test available_satellites() function."""
    result = available_satellites(prefix_with_sensor=prefix_with_sensor)
    assert isinstance(result, list)
    assert len(result) > 0


@pytest.mark.parametrize("suffix_with_satellite", [True, False])
def test_available_sensors(suffix_with_satellite):
    """Test available_sensors() function."""
    result = available_sensors(suffix_with_satellite=suffix_with_satellite)
    assert isinstance(result, list)
    assert len(result) > 0


def test_available_product_categories():
    """Test available_product_categories() function."""
    result = available_product_categories()
    assert isinstance(result, list)
    assert len(result) > 0


def test_available_versions():
    """Test available_versions() function."""
    result = available_versions()
    assert isinstance(result, list)
    assert len(result) > 0


def test_available_products():
    """Test available_products() function."""
    result = available_products()
    assert isinstance(result, list)
    assert len(result) > 0


def test_is_trmm_product():
    """Test is_trmm_product() function."""
    assert is_trmm_product("2A-PR")
    assert not is_trmm_product("2A-DPR")
    assert not is_trmm_product("BAD")


def test_is_gpm_api_product():
    """Test is_gpm_api_product() function."""
    assert is_gpm_api_product("2A-DPR")
    assert not is_gpm_api_product("2A-PR")
    assert not is_gpm_api_product("BAD")
