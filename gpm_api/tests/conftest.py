import pytest
from gpm_api.io.products import get_info_dict, available_products


@pytest.fixture
def product_types() -> list[str]:
    ''' Return a list of all product types from the info dict'''
    product_types = []
    for product, props in get_info_dict().items():
        product_types += props['product_types']

    product_types = list(set(product_types))  # Dedup list

    return product_types


@pytest.fixture
def product_categories() -> list[str]:
    ''' Return a list of product categories from the info dict'''

    return list(
        set([props['product_category'] for props in get_info_dict().values()])
    )


@pytest.fixture
def product_levels() -> list[str]:
    ''' Return a list of product levels from the info dict'''

    # Available in gpm_api.io.checks.check_product_level()
    return ['1A', '1B', '1C', '2A', '2B']


@pytest.fixture
def versions() -> list[int]:
    ''' Return a list of versions '''

    # Available in gpm_api.io.checks.check_version()
    return [4, 5, 6, 7]


@pytest.fixture
def products() -> list[str]:
    ''' Return a list of all products regardless of type '''

    return available_products()
