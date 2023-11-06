from datetime import datetime
import os

from pytest_mock.plugin import MockerFixture

from gpm_api.io import find


def test_get_local_daily_filepaths(
    mock_configuration: dict[str, str],
    mocker: MockerFixture,
):
    """Test _get_all_daily_filepaths for "local" storage"""

    storage = "local"
    date = datetime(2020, 12, 31)
    product = "2A-DPR"
    product_category = "RADAR"
    product_type = "RS"
    version = 7
    verbose = True

    # Test with non-existent files
    returned_filepaths = find._get_all_daily_filepaths(
        storage=storage,
        date=date,
        product=product,
        product_type=product_type,
        version=version,
        verbose=verbose,
    )
    assert returned_filepaths == []

    # Mock os.listdir to return a list of filenames
    mock_filenames = [
        "file1.HDF5",
        "file2.HDF5",
    ]
    mocker.patch("gpm_api.io.local.os.listdir", return_value=mock_filenames)
    mocker.patch("gpm_api.io.local.os.path.exists", return_value=True)

    storage = "local"
    date = datetime(2020, 12, 31)
    product = "2A-DPR"
    product_category = "RADAR"
    version = 7
    verbose = True

    for product_type in ["RS", "NRT"]:
        returned_filepaths = find._get_all_daily_filepaths(
            storage=storage,
            date=date,
            product=product,
            product_type=product_type,
            version=version,
            verbose=verbose,
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
