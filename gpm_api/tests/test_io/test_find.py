from datetime import datetime
import os

from pytest_mock.plugin import MockerFixture

from gpm_api.io import find


def test_get_local_daily_filepaths(
    mock_configuration: dict[str, str],
    mocker: MockerFixture,
):
    """Test _get_all_daily_filepaths for "local" storage"""

    kwargs = {
        "storage": "local",
        "date": datetime(2020, 12, 31),
        "product": "1C-GMI",
        "product_type": "RS",
        "version": 7,
        "verbose": True,
    }

    product_category = "PMW"

    # Test with non-existent files
    returned_filepaths = find._get_all_daily_filepaths(**kwargs)
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
        kwargs["product_type"] = product_type

        returned_filepaths = find._get_all_daily_filepaths(**kwargs)

        expected_filepath_elements = [
            mock_configuration["gpm_base_dir"],
            "GPM",
            product_type,
        ]

        if product_type == "RS":
            expected_filepath_elements.append(f"V0{kwargs['version']}")

        expected_filepath_elements.extend(
            [
                product_category,
                kwargs["product"],
                kwargs["date"].strftime("%Y"),
                kwargs["date"].strftime("%m"),
                kwargs["date"].strftime("%d"),
            ]
        )

        expected_filepaths = [
            os.path.join(*expected_filepath_elements, filename) for filename in mock_filenames
        ]

        assert returned_filepaths == expected_filepaths


def test_get_pps_daily_filepaths(
    mocker: MockerFixture,
):
    """Test _get_all_daily_filepaths for "pps" storage"""

    kwargs = {
        "storage": "pps",
        "date": datetime(2020, 12, 31),
        "product": "1C-GMI",
        "product_type": None,
        "version": 7,
        "verbose": True,
    }

    pps_rs_dir = "1C"
    pps_nrt_dir = "1C/GMI"

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
    kwargs["product_type"] = "RS"
    kwargs["version"] = 7
    returned_filepaths = find._get_all_daily_filepaths(**kwargs)
    base_url = f"ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/{kwargs['date'].strftime('%Y/%m/%d')}/{pps_rs_dir}/"
    expected_filepaths = [f"{base_url}{filename}" for filename in mock_filenames]
    assert returned_filepaths == expected_filepaths

    # Test RS lower version
    kwargs["product_type"] = "RS"
    kwargs["version"] = 5
    returned_filepaths = find._get_all_daily_filepaths(**kwargs)
    base_url = f"ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmallversions/V0{kwargs['version']}/{kwargs['date'].strftime('%Y/%m/%d')}/{pps_rs_dir}/"
    expected_filepaths = [f"{base_url}{filename}" for filename in mock_filenames]
    assert returned_filepaths == expected_filepaths

    # Test NRT
    kwargs["product_type"] = "NRT"
    kwargs["version"] = 7
    returned_filepaths = find._get_all_daily_filepaths(**kwargs)
    base_url = f"ftps://jsimpsonftps.pps.eosdis.nasa.gov/data/{pps_nrt_dir}/"
    expected_filepaths = [f"{base_url}{filename}" for filename in mock_filenames]
    assert returned_filepaths == expected_filepaths

    # Test NRT IMERG
    kwargs["product"] = "IMERG-ER"
    kwargs["product_type"] = "NRT"
    pps_nrt_dir = "imerg/early"
    returned_filepaths = find._get_all_daily_filepaths(**kwargs)
    base_url = f"ftps://jsimpsonftps.pps.eosdis.nasa.gov/data/{pps_nrt_dir}/{kwargs['date'].strftime('%Y%m')}/"
    expected_filepaths = [f"{base_url}{filename}" for filename in mock_filenames]
    assert returned_filepaths == expected_filepaths
