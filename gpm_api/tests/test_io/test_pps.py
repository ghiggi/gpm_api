from typing import Dict, Any, List
from pytest_mock import MockerFixture
from gpm_api.io import pps
from gpm_api.io.products import available_products, available_versions


def test_find_pps_daily_filepaths_private(
    mocker: MockerFixture,
    product_types: List[str],
    server_paths: Dict[str, Any],
) -> None:
    """Test the find_pps_daily_filepaths function."""

    # Mock server call, with a return of empty data
    mocker.patch.object(pps, "_get_pps_daily_filepaths", return_value=[])

    for product_type in product_types:
        for product in available_products(product_type=product_type):
            for version in available_versions(product=product):
                pps._find_pps_daily_filepaths(
                    date="2021-01-01",
                    product=product,
                    version=version,
                    product_type=product_type,
                )

    # Return the curated server_path list
    mocker.patch.object(
        pps,
        "_get_pps_daily_filepaths",
        return_value=list(server_paths),
    )

    for product_type in product_types:
        for product in available_products(product_type=product_type):
            pps._find_pps_daily_filepaths(
                date="2021-01-01",
                product=product,
                version=None,
                product_type=product_type,
            )


def test_find_pps_filepaths(
    product_types: List[str],
    mocker: MockerFixture,
    server_paths: Dict[str, Any],
) -> None:
    """Test the find_pps_filepaths function."""

    sftp_paths = [x for x in list(server_paths) if x.split("://")[0] == "sftp"]
    mocker.patch.object(
        pps,
        "_find_pps_daily_filepaths",
        autospec=True,
        return_value=(sftp_paths, []),
    )

    for product_type in product_types:
        for product in available_products(product_type=product_type):
            assert (
                pps.find_pps_filepaths(
                    product=product,
                    product_type=product_type,
                    start_time="2021-01-01",
                    end_time="2021-01-01",
                )
                == sftp_paths
            )

            # Non-parallel
            assert (
                pps.find_pps_filepaths(
                    product=product,
                    product_type=product_type,
                    start_time="2021-01-01",
                    end_time="2021-01-01",
                    parallel=False,
                )
                == sftp_paths
            )
