import datetime
import os
from typing import Dict, Any, List
from pytest_mock import MockerFixture
from gpm_api.io import pps
from gpm_api.io import find
from gpm_api.io.products import available_products, available_versions


def test_get_pps_nrt_product_dir(products: List[str]) -> None:
    """Test NRT product type folder name

    Depends on gpm_api.io.pps._get_pps_nrt_product_folder_name()
    """

    date = datetime.datetime(2021, 1, 1).date()

    for product in products:
        # Only work on NRT products
        if product in available_products(product_type="NRT"):
            # Dependent on dir forming private function
            foldername = pps._get_pps_nrt_product_folder_name(product)

            res = pps._get_pps_nrt_product_dir(product, date)
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


# def test_get_pps_product_directory(
#     products: List[str],
#     product_types: List[str],
# ) -> None:
#     for product in products:
#         for product_type in product_types:
#             # Only work on NRT products
#             if product in available_products(product_type=product_type):
#                 # Dependent on dir forming private function
#                 foldername = pps._get_pps_nrt_product_folder_name(product)

#                 res = pps.get_pps_product_directory(product, product_type, date, version)
#                 assert res == foldername
#     pass


def test_find_pps_daily_filepaths_private(
    mocker: MockerFixture,
    product_types: List[str],
    remote_filepaths: Dict[str, Any],
) -> None:
    """Test the find_pps_daily_filepaths function."""

    # Mock server call, with a return of empty data
    mocker.patch.object(pps, "get_pps_daily_filepaths", return_value=[])

    for product_type in product_types:
        for product in available_products(product_type=product_type):
            for version in available_versions(product=product):
                find.find_daily_filepaths(
                    storage="pps",
                    date="2021-01-01",
                    product=product,
                    version=version,
                    product_type=product_type,
                )

    # Return the curated remote_filepath list
    mocker.patch.object(
        pps,
        "get_pps_daily_filepaths",
        return_value=list(remote_filepaths),
    )

    for product_type in product_types:
        for product in available_products(product_type=product_type):
            find.find_daily_filepaths(
                storage="pps",
                date="2021-01-01",
                product=product,
                version=None,
                product_type=product_type,
            )


def test_find_pps_filepaths(
    product_types: List[str],
    mocker: MockerFixture,
    remote_filepaths: Dict[str, Any],
) -> None:
    """Test the PPS find_filepaths function."""

    sftp_paths = [x for x in list(remote_filepaths) if x.split("://")[0] == "sftp"]
    mocker.patch.object(
        find,
        "find_daily_filepaths",
        autospec=True,
        return_value=(sftp_paths, []),
    )

    for product_type in product_types:
        for product in available_products(product_type=product_type):
            assert (
                find.find_filepaths(
                    storage="pps",
                    product=product,
                    product_type=product_type,
                    start_time="2021-01-01",
                    end_time="2021-01-01",
                )
                == sftp_paths
            )

            # Non-parallel
            assert (
                find.find_filepaths(
                    storage="pps",
                    product=product,
                    product_type=product_type,
                    start_time="2021-01-01",
                    end_time="2021-01-01",
                    parallel=False,
                )
                == sftp_paths
            )
