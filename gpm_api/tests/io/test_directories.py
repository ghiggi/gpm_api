import pytest
import datetime
import os
from typing import List
from pytest_mock import MockerFixture
from gpm_api.io import directories as dir
from gpm_api.io.products import available_products, get_info_dict


def test_get_product_category(
    products: List[str],
    product_categories: List[str],
) -> None:
    """Test that the product category is in the list of product categories."""
    for product in products:
        assert dir.get_product_category(product) in product_categories

    # If product_category is None, raise ValueError
    def return_none():
        yield None

    # Add value to info dict to force a ValueError on None return
    get_info_dict()["fake_product"] = {"product_category": None}
    with pytest.raises(ValueError):
        dir.get_product_category("fake_product")

    get_info_dict().pop("fake_product")  # Remove fake value


def test_get_disk_dir_pattern(
    products: List[str],
    product_types: List[str],
    versions: List[int],
) -> None:
    """Test that the disk directory pattern is correct."""

    # Go through all available options and check that the pattern is correct
    for product in products:
        for product_type in product_types:
            for version in versions:
                dir_pattern = dir.get_disk_dir_pattern(
                    product,
                    product_type,
                    version,
                )

                # Work only on product if product_type are compatible
                if product in available_products(product_type=product_type):
                    product_category = dir.get_product_category(product)
                    if product_type == "NRT":
                        assert "V0" not in dir_pattern
                        assert dir_pattern == os.path.join(
                            "GPM", product_type, product_category, product
                        )
                    elif product_type == "RS":
                        assert str(version) in dir_pattern
                        # Literal
                        assert dir_pattern == os.path.join(
                            "GPM", product_type, f"V0{version}", product_category, product
                        )


def test_get_disk_directory(
    products: List[str],
    product_types: List[str],
    versions: List[int],
    tmpdir,
) -> None:
    """Test that the disk directory is correct."""

    date = datetime.datetime.strptime("2021-01-01", "%Y-%m-%d").date()

    base_dir = os.path.join(tmpdir, "gpm_api_data")

    for product in products:
        for product_type in product_types:
            for version in versions:
                dir_path = dir.get_disk_directory(
                    base_dir,
                    product,
                    product_type,
                    version,
                    date,
                )

                # Work only on product if product_type are compatible
                if product in available_products(product_type=product_type):
                    product_category = dir.get_product_category(product)
                    if product_type == "NRT":
                        assert "V0" not in dir_path
                        assert dir_path == os.path.join(
                            base_dir,
                            os.path.join("GPM", product_type, product_category, product),
                            date.strftime("%Y"),
                            date.strftime("%m"),
                            date.strftime("%d"),
                        )
                    elif product_type == "RS":
                        assert str(version) in dir_path
                        # Literal
                        assert dir_path == os.path.join(
                            base_dir,
                            os.path.join(
                                "GPM", product_type, f"V0{version}", product_category, product
                            ),
                            date.strftime("%Y"),
                            date.strftime("%m"),
                            date.strftime("%d"),
                        )


def test_get_pps_nrt_product_dir(products: List[str]) -> None:
    """Test NRT product type folder name

    Depends on gpm_api.io.directories._get_pps_nrt_product_folder_name()
    """

    date = datetime.datetime(2021, 1, 1).date()

    for product in products:
        # Only work on NRT products
        if product in available_products(product_type="NRT"):
            # Dependent on dir forming private function
            foldername = dir._get_pps_nrt_product_folder_name(product)

            res = dir._get_pps_nrt_product_dir(product, date)
            if product in available_products(
                product_type="NRT",
                product_category="IMERG",
            ):
                assert res == os.path.join(
                    foldername,
                    date.strftime("%Y%m"),
                )
            else:
                assert res == foldername


# def test_get_pps_directory(
#     products: List[str],
#     product_types: List[str],
# ) -> None:
#     for product in products:
#         for product_type in product_types:
#             # Only work on NRT products
#             if product in available_products(product_type=product_type):
#                 # Dependent on dir forming private function
#                 foldername = dir._get_pps_nrt_product_folder_name(product)

#                 res = dir.get_pps_directory(product, product_type)
#                 assert res == foldername
#     pass
