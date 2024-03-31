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
"""This module test the NASA PPS file search routines."""
import datetime

import pytest
from pytest_mock.plugin import MockerFixture

import gpm
from gpm.io import pps
from gpm.io.products import available_products


def test_get_pps_nrt_product_dir(products: list[str]) -> None:
    """Test NRT product type folder name

    Depends on gpm.io.pps._get_pps_nrt_product_folder_name()
    """
    date = datetime.datetime(2021, 1, 1).date()
    for product in products:
        # Only work on NRT products
        if product in available_products(product_types="NRT"):
            # Dependent on dir forming private function
            foldername = pps._get_pps_nrt_product_folder_name(product)

            res = pps._get_pps_nrt_product_dir(product, date)
            if product in available_products(
                product_types="NRT",
                product_categories="IMERG",
            ):
                assert res == f"{foldername}/{date.strftime('%Y%m')}"
            else:
                assert res == foldername


@pytest.mark.parametrize("product_type", ["RS", "NRT"])
@pytest.mark.parametrize("filepaths", [[], ["pps_filepath1", "pps_filepath2"]])
def test_find_first_pps_granule_filepath(mocker: MockerFixture, filepaths, product_type) -> None:
    """Test find_first_pps_granule_filepath function."""
    from gpm.io import find

    product = "2A-DPR"
    version = 7

    # Mock find_filepaths
    mocker.patch.object(find, "find_filepaths", autospec=True, return_value=filepaths)

    # Test if no files available on PPS raise error
    if len(filepaths) == 0:
        with pytest.raises(ValueError):
            pps.find_first_pps_granule_filepath(
                version=version,
                product=product,
                product_type=product_type,
            )
    # Test return first filepath  (sorted alphabetically)
    else:
        pps_filepath = pps.find_first_pps_granule_filepath(
            product=product, product_type=product_type, version=version
        )

        assert pps_filepath == "pps_filepath1"


class TestGetPPSFileList:
    def test_try_get_pps_file_list(self, mocker: MockerFixture):
        url = "http://example.com/products/"

        # Mock subprocess.Popen to simulate curl command success
        mock_process = mocker.MagicMock()
        mock_process.communicate.return_value = (b"file1.txt\nfile2.txt", b"")
        mocker.patch("subprocess.Popen", return_value=mock_process, autospec=True)

        with gpm.config.set({"username_pps": "user", "password_pps": "pass"}):
            filepaths = pps._try_get_pps_file_list(url)
            assert filepaths == [
                "file1.txt",
                "file2.txt",
            ], "File paths do not match expected output"

    def test_try_get_pps_file_list_unavailable_server(self, mocker: MockerFixture):
        url = "http://example.com/products/"

        # Mock subprocess.Popen to simulate server unavailability
        mock_process = mocker.MagicMock()
        mock_process.communicate.return_value = (b"", b"")
        mocker.patch("subprocess.Popen", return_value=mock_process, autospec=True)

        with gpm.config.set({"username_pps": "user", "password_pps": "pass"}):
            with pytest.raises(ValueError) as excinfo:
                pps._try_get_pps_file_list(url)
            assert "The PPS server is currently unavailable." in str(
                excinfo.value
            ), "Expected ValueError not raised for unavailable server"

    def test_try_get_pps_file_list_no_data_found(self, mocker: MockerFixture):
        url = "http://example.com/products/"

        # Mock subprocess.Popen to simulate no data found on PPS
        mock_process = mocker.MagicMock()
        mock_process.communicate.return_value = (b"<html></html>", b"")
        mocker.patch("subprocess.Popen", return_value=mock_process, autospec=True)

        with gpm.config.set({"username_pps": "user", "password_pps": "pass"}):
            with pytest.raises(ValueError) as excinfo:
                pps._try_get_pps_file_list(url)
            assert "No data found on PPS." in str(
                excinfo.value
            ), "Expected ValueError not raised for no data found"

    @pytest.mark.parametrize("verbose", [True, False])
    def test_get_pps_file_list(self, mocker: MockerFixture, verbose):
        url = "http://example.com/products/"
        product = "2A-DPR"
        version = 6
        date = datetime.date(2020, 7, 5)
        expected_filepaths = [
            "/gpmdata/2020/07/05/radar/file1.HDF5",
            "/gpmdata/2020/07/05/radar/file2.HDF5",
        ]
        mocker.patch.object(pps, "_try_get_pps_file_list", return_value=expected_filepaths)

        filepaths = pps._get_pps_file_list(url, product, date, version, verbose=verbose)
        assert filepaths == expected_filepaths, "File paths do not match expected output"

    def test_no_data_found_verbose(self, mocker: MockerFixture, capsys):
        """Test the 'No data found on PPS.' scenario with verbose=True."""

        url = "http://example.com/products/"
        product = "2A-DPR"
        version = 6
        date = datetime.date(2020, 7, 5)

        mocker.patch.object(
            pps, "_try_get_pps_file_list", side_effect=ValueError("No data found on PPS.")
        )

        filepaths = pps._get_pps_file_list(url, product, date, version, verbose=True)
        assert filepaths == [], "Expected empty list for no data found"

        captured = capsys.readouterr()
        assert (
            f"No data found on PPS on date {date} for product {product}" in captured.out
        ), "Expected verbose message not printed"

    def test_unavailable_server(self, mocker: MockerFixture):
        """Test the 'The PPS server is currently unavailable.' error."""

        url = "BAD URL"
        product = "2A-DPR"
        version = 6
        date = datetime.date(2020, 7, 5)

        mocker.patch.object(
            pps,
            "_try_get_pps_file_list",
            side_effect=ValueError(
                "The PPS server is currently unavailable. Sorry for the inconvenience."
            ),
        )
        with pytest.raises(ValueError) as excinfo:
            pps._get_pps_file_list(url, product, date, version)
        assert "The PPS server is currently unavailable." in str(
            excinfo.value
        ), "Expected ValueError not raised for unavailable server"

    def test_undefined_error(self, mocker: MockerFixture):
        """Test undefined error handling."""

        url = "http://example.com/products/"
        product = "2A-DPR"
        version = 6
        date = datetime.date(2020, 7, 5)

        mocker.patch.object(pps, "_try_get_pps_file_list", side_effect=Exception("Some new error."))
        with pytest.raises(ValueError) as excinfo:
            pps._get_pps_file_list(url, product, date, version)
        assert "Undefined error." in str(
            excinfo.value
        ), "Expected ValueError not raised for an undefined error"
