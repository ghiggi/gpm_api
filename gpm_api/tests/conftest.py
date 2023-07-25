import pytest
import datetime
from typing import Any, List, Dict
from gpm_api.io.products import get_info_dict, available_products


@pytest.fixture
def product_types() -> List[str]:
    """Return a list of all product types from the info dict"""
    product_types = []
    for product, props in get_info_dict().items():
        product_types += props["product_types"]

    product_types = list(set(product_types))  # Dedup list

    return product_types


@pytest.fixture
def product_categories() -> List[str]:
    """Return a list of product categories from the info dict"""

    return list(set([props["product_category"] for props in get_info_dict().values()]))


@pytest.fixture
def product_levels() -> List[str]:
    """Return a list of product levels from the info dict"""

    # Available in gpm_api.io.checks.check_product_level()
    return ["1A", "1B", "1C", "2A", "2B"]


@pytest.fixture
def versions() -> List[int]:
    """Return a list of versions"""

    # Available in gpm_api.io.checks.check_version()
    return [4, 5, 6, 7]


@pytest.fixture
def products() -> List[str]:
    """Return a list of all products regardless of type"""

    return available_products()


@pytest.fixture
def username() -> str:
    """Return a username

    GPM uses an email address as username
    """

    return "testuser@example.com"


@pytest.fixture
def password() -> str:
    """Return a password

    GPM password is the username
    """

    return "testuser@example.com"


@pytest.fixture
def server_paths() -> Dict[str, Dict[str, Any]]:
    """Return a list of probable GPM server paths"""

    # Not validated to be real paths but follow the structure
    return {
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S170044-E183317.036092.V07A.HDF5": {
            "year": 2020,
            "month": 7,
            "day": 5,
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 17, 0, 44),
            "end_time": datetime.datetime(2020, 7, 5, 18, 33, 17),
            "version": 7,
        },
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S183318-E200550.036093.V07A.HDF5": {
            "year": 2020,
            "month": 7,
            "day": 5,
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 18, 33, 18),
            "end_time": datetime.datetime(2020, 7, 5, 20, 5, 50),
            "version": 7,
        },
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S200551-E213823.036094.V07A.HDF5": {
            "year": 2020,
            "month": 7,
            "day": 5,
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 20, 5, 51),
            "end_time": datetime.datetime(2020, 7, 5, 21, 38, 23),
            "version": 7,
        },
        # Include non-ftps folders
        "ftp://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S213824-E231056.036095.V07A.HDF5": {
            "year": 2020,
            "month": 7,
            "day": 5,
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 21, 38, 24),
            "end_time": datetime.datetime(2020, 7, 5, 23, 10, 56),
            "version": 7,
        },
        "ftp://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S231057-E004329.036096.V07A.HDF5": {
            "year": 2020,
            "month": 7,
            "day": 5,
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 23, 10, 57),
            "end_time": datetime.datetime(2020, 7, 6, 0, 43, 29),
            "version": 7,
        },
        "ftp://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S004330-E021602.036097.V07A.HDF5": {
            "year": 2020,
            "month": 7,
            "day": 5,
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 0, 43, 30),
            "end_time": datetime.datetime(2020, 7, 5, 2, 16, 2),
            "version": 7,
        },
        # TODO: Add more products with varying attributes ...
    }
