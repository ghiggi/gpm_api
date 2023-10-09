#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 12:35:27 2023

@author: ghiggi
"""
import pytest
from typing import List
from gpm_api.io.products import get_product_category, get_info_dict


def test_get_product_category(
    products: List[str],
    product_categories: List[str],
) -> None:
    """Test that the product category is in the list of product categories."""
    for product in products:
        assert get_product_category(product) in product_categories

    # Add value to info dict to force a ValueError on None return
    get_info_dict()["fake_product"] = {"product_category": None}
    with pytest.raises(ValueError):
        get_product_category("fake_product")

    get_info_dict().pop("fake_product")  # Remove fake value
