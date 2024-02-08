import datetime
import pytest
from pytest_mock.plugin import MockerFixture
from typing import List
from gpm_api.io import pps
from gpm_api.io.products import available_products
import gpm_api.configs as cfg


def test_get_pps_nrt_product_dir(products: List[str]) -> None:
    """Test NRT product type folder name

    Depends on gpm_api.io.pps._get_pps_nrt_product_folder_name()
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
    from gpm_api.io import find

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


def TestGetPPSFileList():
    def test__get_pps_file_list_success(self, mocker: MockerFixture):
        # Mock subprocess.Popen to simulate curl command success
        mock_process = mocker.MagicMock()
        mock_process.communicate.return_value = (b"file1.txt\nfile2.txt", b"")
        mocker.patch("subprocess.Popen", return_value=mock_process, autospec=True)

        # Mock get_pps_username and get_pps_password to return dummy values
        mocker.patch.object(cfg, "get_pps_username", return_value="user", autospec=True)
        mocker.patch.object(cfg, "get_pps_password", return_value="pass", autospec=True)

        filepaths = pps.__get_pps_file_list("http://example.com/products/")
        assert filepaths == ["file1.txt", "file2.txt"], "File paths do not match expected output"

    @pytest.mark.parametrize("verbose", [True, False])
    def test_get_pps_file_list_success(mocker, verbose):
        expected_filepaths = [
            "/gpmdata/2020/07/05/radar/file1.HDF5",
            "/gpmdata/2020/07/05/radar/file2.HDF5",
        ]
        mocker.patch("pps.__get_pps_file_list", return_value=expected_filepaths)

        filepaths = pps._get_pps_file_list(
            "http://example.com/products/", "GPM", datetime(2020, 7, 5), "06", verbose=verbose
        )
        assert filepaths == expected_filepaths, "File paths do not match expected output"

    def test__get_pps_file_list_unavailable_server(self, mocker: MockerFixture):
        # Mock subprocess.Popen to simulate server unavailability
        mock_process = mocker.MagicMock()
        mock_process.communicate.return_value = (b"", b"")
        mocker.patch("subprocess.Popen", return_value=mock_process, autospec=True)

        mocker.patch.object(cfg, "get_pps_username", return_value="user", autospec=True)
        mocker.patch.object(cfg, "get_pps_password", return_value="pass", autospec=True)

        with pytest.raises(ValueError) as excinfo:
            pps.__get_pps_file_list("http://example.com/products/")
        assert "The PPS server is currently unavailable." in str(
            excinfo.value
        ), "Expected ValueError not raised for unavailable server"

    def test_no_data_found(self, mocker: MockerFixture):
        # Mock subprocess.Popen to simulate no data found on PPS
        mock_process = mocker.MagicMock()
        mock_process.communicate.return_value = (b"<html></html>", b"")
        mocker.patch("subprocess.Popen", return_value=mock_process, autospec=True)

        mocker.patch.object(cfg, "get_pps_username", return_value="user", autospec=True)
        mocker.patch.object(cfg, "get_pps_password", return_value="pass", autospec=True)

        with pytest.raises(ValueError) as excinfo:
            pps.__get_pps_file_list("http://example.com/products/")
        assert "No data found on PPS." in str(
            excinfo.value
        ), "Expected ValueError not raised for no data found"

    def test_no_data_found_verbose(self, mocker, capsys):
        """Test the 'No data found on PPS.' scenario with verbose=True."""
        mocker.patch.object(
            pps, "__get_pps_file_list", side_effect=ValueError("No data found on PPS.")
        )

        date = datetime(2020, 7, 5)
        product = "GPM"
        version = "06"
        filepaths = pps._get_pps_file_list(
            "http://example.com/products/", product, date, version, verbose=True
        )
        assert filepaths == [], "Expected empty list for no data found"

        captured = capsys.readouterr()
        assert (
            f"No data found on PPS on date {date} for product {product} (V006)" in captured.out
        ), "Expected verbose message not printed"

    def test_unavailable_server(self, mocker):
        """Test the 'The PPS server is currently unavailable.' error."""
        mocker.patch.object(
            pps,
            "get_pps_file_list",
            side_effect=ValueError(
                "The PPS server is currently unavailable. Sorry for the inconvenience."
            ),
        )
        with pytest.raises(ValueError) as excinfo:
            pps._get_pps_file_list("http://wrong.url/", "GPM", datetime(2020, 7, 5), "06")
        assert "The PPS server is currently unavailable." in str(
            excinfo.value
        ), "Expected ValueError not raised for unavailable server"

    def test_undefined_error(self, mocker):
        """Test undefined error handling."""
        mocker.patch.object(pps, "__get_pps_file_list", side_effect=Exception("Some new error."))
        with pytest.raises(ValueError) as excinfo:
            pps._get_pps_file_list(
                "http://example.com/products/", "GPM", datetime(2020, 7, 5), "06"
            )
        assert "Undefined error." in str(
            excinfo.value
        ), "Expected ValueError not raised for an undefined error"
