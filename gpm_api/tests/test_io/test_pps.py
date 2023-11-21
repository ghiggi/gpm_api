import datetime
from typing import List
from gpm_api.io import pps
from gpm_api.io.products import available_products


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
                assert res == f"{foldername}/{date.strftime('%Y%m')}"
            else:
                assert res == foldername
