import os
from typing import Dict

from pytest_mock import MockerFixture

from gpm_api.io import local


def test_get_local_filepaths(
    mock_configuration: Dict[str, str],
    mocker: MockerFixture,
):
    product = "2A-DPR"
    product_category = "RADAR"
    product_type = "RS"
    version = 7

    # Test with non-existent files
    returned_filepaths = local.get_local_filepaths(
        product=product,
        product_type=product_type,
        version=version,
    )

    assert returned_filepaths == []

    # Mock glob.glob to return a list of filepaths
    mock_filenames = [
        "file1.HDF5",
        "file2.HDF5",
    ]

    def mock_glob(pattern):
        return [os.path.join(pattern.rstrip("*"), filename) for filename in mock_filenames]

    mocker.patch("gpm_api.io.local.glob.glob", side_effect=mock_glob)
    mocker.patch("gpm_api.io.local.os.path.exists", return_value=True)

    returned_filepaths = local.get_local_filepaths(
        product=product,
        product_type=product_type,
        version=version,
    )

    expected_filepaths = [
        os.path.join(
            mock_configuration["gpm_base_dir"],
            "GPM",
            product_type,
            f"V0{version}",
            product_category,
            product,
            "*",
            "*",
            "*",
            filename,
        )
        for filename in mock_filenames
    ]
    assert returned_filepaths == expected_filepaths
