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
"""This module test the local file search routines."""

import datetime
import os

import gpm
from gpm.io import local
from gpm.io.products import available_products, get_product_category


def create_fake_file(
    base_dir,
    filename="dummy.HDF5",
    product="2A-DPR",
    product_type="RS",
    date=datetime.date(2022, 1, 1),
    version=7,
):
    from gpm.io.checks import check_base_dir
    from gpm.io.local import get_local_product_directory

    base_dir = check_base_dir(base_dir)
    product_dir_path = get_local_product_directory(
        base_dir=base_dir,
        product=product,
        product_type=product_type,
        version=version,
        date=date,
    )
    # Create directory
    os.makedirs(product_dir_path, exist_ok=True)
    # Define filepath
    filepath = os.path.join(product_dir_path, filename)
    # Create fake file
    with open(filepath, "w") as f:
        f.write("Hello World")
    return filepath


def test_get_local_filepaths(tmp_path):
    """Test get_local_filepaths returns a list of the available filepaths."""
    # Create GPM base directory
    base_dir = tmp_path / "GPM"
    base_dir.mkdir(parents=True)

    product = "2A-DPR"
    product_type = "RS"
    version = 7

    with gpm.config.set({"base_dir": base_dir}):
        # Test with non-existent files
        returned_filepaths = local.get_local_filepaths(
            product=product,
            product_type=product_type,
            version=version,
        )

        assert returned_filepaths == []

        # Create fake_files
        filepath1 = create_fake_file(
            base_dir=base_dir,
            filename="file1.HDF5",
            product=product,
            product_type=product_type,
            version=version,
        )
        filepath2 = create_fake_file(
            base_dir=base_dir,
            filename="file2.HDF5",
            product=product,
            product_type=product_type,
            version=version,
        )
        expected_filepaths = [filepath1, filepath2]

        # Test it retrieve the available (fake) files
        returned_filepaths = local.get_local_filepaths(
            product=product,
            product_type=product_type,
            version=version,
        )
        assert returned_filepaths == sorted(expected_filepaths)


def test__get_local_dir_pattern(
    products: list[str],
    product_types: list[str],
    versions: list[int],
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
                if product in available_products(product_types=product_type):
                    product_category = get_product_category(product)
                    if product_type == "NRT":
                        assert "V0" not in dir_pattern
                        assert dir_pattern == os.path.join(
                            "GPM",
                            product_type,
                            product_category,
                            product,
                        )
                    elif product_type == "RS":
                        assert str(version) in dir_pattern
                        # Literal
                        assert dir_pattern == os.path.join(
                            "GPM",
                            product_type,
                            f"V0{version}",
                            product_category,
                            product,
                        )


def test_get_local_product_base_directory(
    products: list[str],
    product_types: list[str],
    versions: list[int],
    tmpdir,
) -> None:
    """Test that the disk directory is correct."""
    date = datetime.datetime.strptime("2021-01-01", "%Y-%m-%d").date()

    base_dir = os.path.join(tmpdir, "gpm_data")
    with gpm.config.set({"base_dir": base_dir}):
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
                    if product in available_products(product_types=product_type):
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
                                    "GPM",
                                    product_type,
                                    f"V0{version}",
                                    product_category,
                                    product,
                                ),
                                date.strftime("%Y"),
                                date.strftime("%m"),
                                date.strftime("%d"),
                            )
