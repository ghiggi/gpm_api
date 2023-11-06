from datetime import datetime
import os

from pytest_mock.plugin import MockerFixture

from gpm_api.io import find
from gpm_api.io.products import available_products


def test_get_local_daily_filepaths(
    mock_configuration: dict[str, str],
    mocker: MockerFixture,
    product_info: dict[str, dict],
):
    """Test _get_all_daily_filepaths for "local" storage"""

    storage = "local"
    date = datetime(2020, 12, 31)

    # Test with non-existent files
    returned_filepaths = find._get_all_daily_filepaths(
        storage=storage,
        date=date,
        product="1C-GMI",
        product_type="RS",
        version=7,
        verbose=True,
    )
    assert returned_filepaths == []

    # Mock os.listdir to return a list of filenames
    mock_filenames = [
        "file1.HDF5",
        "file2.HDF5",
    ]
    mocker.patch("gpm_api.io.local.os.listdir", return_value=mock_filenames)
    mocker.patch("gpm_api.io.local.os.path.exists", return_value=True)

    # Test with existing files (mocked)
    for product_type in ["RS", "NRT"]:
        for product in available_products(product_type=product_type):
            info = product_info[product]
            version = info["available_versions"][-1]
            product_category = info["product_category"]

            returned_filepaths = find._get_all_daily_filepaths(
                storage=storage,
                date=date,
                product=product,
                product_type=product_type,
                version=version,
                verbose=True,
            )

            expected_filepath_elements = [
                mock_configuration["gpm_base_dir"],
                "GPM",
                product_type,
            ]

            if product_type == "RS":
                expected_filepath_elements.append(f"V0{version}")

            expected_filepath_elements.extend(
                [
                    product_category,
                    product,
                    date.strftime("%Y"),
                    date.strftime("%m"),
                    date.strftime("%d"),
                ]
            )

            expected_filepaths = [
                os.path.join(*expected_filepath_elements, filename) for filename in mock_filenames
            ]

            assert returned_filepaths == expected_filepaths


def test_get_pps_daily_filepaths(
    mocker: MockerFixture,
    product_info: dict[str, dict],
):
    """Test _get_all_daily_filepaths for "pps" storage"""

    stoarge = "pps"
    date = datetime(2020, 12, 31)

    # Mock gpm_api.io.pps.__get_pps_file_list, which uses curl to get a list of files
    mock_filenames = [
        "file1.HDF5",
        "file2.HDF5",
    ]

    def mock_get_pps_file_list(url_product_dir):
        # Remove the base URL, assuming they have the followgin format:
        # RS: https://arthurhouhttps.pps.eosdis.nasa.gov/text/...
        # NRT: https://jsimpsonhttps.pps.eosdis.nasa.gov/text/...
        url_without_base = url_product_dir.split("/text")[1]
        return [f"{url_without_base}/{filename}" for filename in mock_filenames]

    mocker.patch("gpm_api.io.pps.__get_pps_file_list", side_effect=mock_get_pps_file_list)

    # Test RS version 7
    product_type = "RS"
    version = 7
    for product in available_products(product_type=product_type, version=version):
        info = product_info[product]
        pps_dir = info["pps_rs_dir"]

        returned_filepaths = find._get_all_daily_filepaths(
            storage=stoarge,
            date=date,
            product=product,
            product_type=product_type,
            version=version,
            verbose=True,
        )
        base_url = f"ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/{date.strftime('%Y/%m/%d')}/{pps_dir}/"
        expected_filepaths = [f"{base_url}{filename}" for filename in mock_filenames]
        assert returned_filepaths == expected_filepaths

    # Test RS lower version
    product_type = "RS"
    for product in available_products(product_type=product_type):
        info = product_info[product]
        pps_dir = info["pps_rs_dir"]

        for version in info["available_versions"]:
            if version == 7:
                continue

            returned_filepaths = find._get_all_daily_filepaths(
                storage=stoarge,
                date=date,
                product=product,
                product_type=product_type,
                version=version,
                verbose=True,
            )
            base_url = f"ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmallversions/V0{version}/{date.strftime('%Y/%m/%d')}/{pps_dir}/"
            expected_filepaths = [f"{base_url}{filename}" for filename in mock_filenames]
            assert returned_filepaths == expected_filepaths

    # Test NRT
    product_type = "NRT"
    for product in available_products(product_type=product_type):
        info = product_info[product]
        if info["product_category"] == "IMERG":
            continue

        version = info["available_versions"][-1]
        pps_dir = info["pps_nrt_dir"]

        returned_filepaths = find._get_all_daily_filepaths(
            storage=stoarge,
            date=date,
            product=product,
            product_type=product_type,
            version=version,
            verbose=True,
        )
        base_url = f"ftps://jsimpsonftps.pps.eosdis.nasa.gov/data/{pps_dir}/"
        expected_filepaths = [f"{base_url}{filename}" for filename in mock_filenames]
        assert returned_filepaths == expected_filepaths

    # Test NRT IMERG
    product_type = "NRT"
    product_category = "IMERG"
    for product in available_products(product_type=product_type, product_category=product_category):
        info = product_info[product]
        version = info["available_versions"][-1]
        pps_dir = info["pps_nrt_dir"]

        returned_filepaths = find._get_all_daily_filepaths(
            storage=stoarge,
            date=date,
            product=product,
            product_type=product_type,
            version=version,
            verbose=True,
        )
        base_url = (
            f"ftps://jsimpsonftps.pps.eosdis.nasa.gov/data/{pps_dir}/{date.strftime('%Y%m')}/"
        )
        expected_filepaths = [f"{base_url}{filename}" for filename in mock_filenames]
        assert returned_filepaths == expected_filepaths
