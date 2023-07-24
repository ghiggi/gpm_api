import pytest
from typing import Any
from gpm_api.io.products import get_info_dict, available_products


@pytest.fixture
def product_types() -> list[str]:
    """Return a list of all product types from the info dict"""
    product_types = []
    for product, props in get_info_dict().items():
        product_types += props["product_types"]

    product_types = list(set(product_types))  # Dedup list

    return product_types


@pytest.fixture
def product_categories() -> list[str]:
    """Return a list of product categories from the info dict"""

    return list(set([props["product_category"] for props in get_info_dict().values()]))


@pytest.fixture
def product_levels() -> list[str]:
    """Return a list of product levels from the info dict"""

    # Available in gpm_api.io.checks.check_product_level()
    return ["1A", "1B", "1C", "2A", "2B"]


@pytest.fixture
def versions() -> list[int]:
    """Return a list of versions"""

    # Available in gpm_api.io.checks.check_version()
    return [4, 5, 6, 7]


@pytest.fixture
def products() -> list[str]:
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
def server_paths() -> list[str]:
    """Return a list of probable GPM server paths"""

    return [  # Not validated to be real paths but follow the structure
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S170044-E183317.036092.V07A.HDF5",
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S183318-E200550.036093.V07A.HDF5",
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S200551-E213823.036094.V07A.HDF5",
        # Include non-ftps folders
        "ftp://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S213824-E231056.036095.V07A.HDF5",
        "ftp://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S231057-E004329.036096.V07A.HDF5",
        "ftp://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S004330-E021602.036097.V07A.HDF5",
    ]
