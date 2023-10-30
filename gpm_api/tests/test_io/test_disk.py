import datetime
import os
from typing import List
from gpm_api.io.products import available_products, get_product_category
from gpm_api.io import local


def test__get_local_dir_pattern(
    products: List[str],
    product_types: List[str],
    versions: List[int],
) -> None:
    """Test that the disk directory pattern is correct."""

    # Go through all available options and check that the pattern is correct
    for product in products:
        for product_type in product_types:
            for version in versions:
                dir_pattern = local._get_local_dir_pattern(
                    product,
                    product_type,
                    version,
                )

                # Work only on product if product_type are compatible
                if product in available_products(product_type=product_type):
                    product_category = get_product_category(product)
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


def test_get_local_product_base_directory(
    products: List[str],
    product_types: List[str],
    versions: List[int],
    tmpdir,
) -> None:
    """Test that the disk directory is correct."""

    date = datetime.datetime.strptime("2021-01-01", "%Y-%m-%d").date()

    # TODO: define in gpm_api.config !!!
    base_dir = os.path.join(tmpdir, "gpm_api_data")

    for product in products:
        for product_type in product_types:
            for version in versions:
                dir_path = local.get_local_product_directory(
                    base_dir=base_dir,
                    product=product,
                    product_type=product_type,
                    version=version,
                    date=date,
                )

                # Work only on product if product_type are compatible
                if product in available_products(product_type=product_type):
                    product_category = get_product_category(product)
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
