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
"""This module test the file search routines."""
import datetime
import os
from typing import Any

import pytest
from pytest_mock.plugin import MockerFixture

import gpm
from gpm.io import find
from gpm.io.find import (
    _check_correct_version,
    _get_all_daily_filepaths,
    find_daily_filepaths,
    find_filepaths,
)
from gpm.io.products import available_products
from gpm.utils.warnings import GPMDownloadWarning


class TestGetDailyFilepaths:
    """Test _get_all_daily_filepaths."""

    date = datetime.datetime(2020, 12, 31)
    mock_filenames = [
        "file1.HDF5",
        "file2.HDF5",
    ]

    def test_local_non_existent_files(
        self,
    ) -> None:
        """Test _get_all_daily_filepaths for "LOCAL" storage with non-existent files."""
        storage = "LOCAL"

        returned_filepaths = _get_all_daily_filepaths(
            storage=storage,
            date=self.date,
            product="1C-GMI",
            product_type="RS",
            version=7,
            verbose=True,
        )
        assert returned_filepaths == []

    def test_local_existing_files(
        self,
        check,  # For non-failing asserts
        mocker: MockerFixture,
        product_info: dict[str, dict],
    ) -> None:
        """Test _get_all_daily_filepaths for "LOCAL" storage with existing (mocked) files."""
        base_dir = "dummy/path/to/base_dir"
        storage = "LOCAL"

        # Mock os.listdir to return a list of filenames
        mocker.patch("gpm.io.local.os.listdir", return_value=self.mock_filenames)
        mocker.patch("gpm.io.local.os.path.exists", return_value=True)

        with gpm.config.set({"base_dir": base_dir}):
            # Test with existing files (mocked)
            for product_type in ["RS", "NRT"]:
                for product in available_products(product_types=product_type):
                    info = product_info[product]
                    version = info["available_versions"][-1]
                    product_category = info["product_category"]

                    returned_filepaths = _get_all_daily_filepaths(
                        storage=storage,
                        date=self.date,
                        product=product,
                        product_type=product_type,
                        version=version,
                        verbose=True,
                    )

                    expected_filepath_elements = [
                        base_dir,
                        "GPM",
                        product_type,
                    ]

                    if product_type == "RS":
                        expected_filepath_elements.append(f"V0{version}")

                    expected_filepath_elements.extend(
                        [
                            product_category,
                            product,
                            self.date.strftime("%Y"),
                            self.date.strftime("%m"),
                            self.date.strftime("%d"),
                        ],
                    )

                    expected_filepaths = [
                        os.path.join(*expected_filepath_elements, filename) for filename in self.mock_filenames
                    ]

                    with check:
                        assert returned_filepaths == expected_filepaths

    @pytest.fixture
    def _mock_get_pps_file_list(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Mock gpm.io.pps._try_get_pps_file_list, which uses curl to get a list of files."""

        def mocked_get_pps_file_list(url_product_dir: str) -> list[str]:
            # Remove the base URL, assuming they have the following format:
            # RS: https://arthurhouhttps.pps.eosdis.nasa.gov/text/...
            # NRT: https://jsimpsonhttps.pps.eosdis.nasa.gov/text/...
            url_without_base = url_product_dir.split("/text")[1]
            return [f"{url_without_base}/{filename}" for filename in self.mock_filenames]

        mocker.patch("gpm.io.pps._try_get_pps_file_list", side_effect=mocked_get_pps_file_list)

    @pytest.mark.usefixtures("_mock_get_pps_file_list")
    def test_pps_rs_version_7(
        self,
        check,  # For non-failing asserts
        product_info: dict[str, dict],
    ) -> None:
        """Test _get_all_daily_filepaths for "PPS" storage with RS version 7 products."""
        storage = "PPS"
        product_type = "RS"
        version = 7

        for product in available_products(product_types=product_type, versions=version):
            info = product_info[product]
            pps_dir = info["pps_rs_dir"]

            returned_filepaths = _get_all_daily_filepaths(
                storage=storage,
                date=self.date,
                product=product,
                product_type=product_type,
                version=version,
                verbose=True,
            )
            base_url = f"ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/{self.date.strftime('%Y/%m/%d')}/{pps_dir}/"
            expected_filepaths = [f"{base_url}{filename}" for filename in self.mock_filenames]
            with check:
                assert returned_filepaths == expected_filepaths

    @pytest.mark.usefixtures("_mock_get_pps_file_list")
    def test_pps_rs_lower_version(
        self,
        check,  # For non-failing asserts
        product_info: dict[str, dict],
    ) -> None:
        """Test _get_all_daily_filepaths for "PPS" storage with RS lower version products."""
        storage = "PPS"
        product_type = "RS"

        for product in available_products(product_types=product_type):
            info = product_info[product]
            pps_dir = info["pps_rs_dir"]

            for version in info["available_versions"]:
                if version == 7:
                    continue

                returned_filepaths = _get_all_daily_filepaths(
                    storage=storage,
                    date=self.date,
                    product=product,
                    product_type=product_type,
                    version=version,
                    verbose=True,
                )
                base_url = f"ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmallversions/V0{version}/{self.date.strftime('%Y/%m/%d')}/{pps_dir}/"
                expected_filepaths = [f"{base_url}{filename}" for filename in self.mock_filenames]
                with check:
                    assert returned_filepaths == expected_filepaths

    @pytest.mark.usefixtures("_mock_get_pps_file_list")
    def test_pps_nrt(
        self,
        check,  # For non-failing asserts
        product_info: dict[str, dict],
    ) -> None:
        """Test _get_all_daily_filepaths for "PPS" storage with NRT products (except IMERG)."""
        storage = "PPS"
        product_type = "NRT"

        for product in available_products(product_types=product_type):
            info = product_info[product]
            if info["product_category"] == "IMERG":
                continue

            version = info["available_versions"][-1]
            pps_dir = info["pps_nrt_dir"]

            returned_filepaths = _get_all_daily_filepaths(
                storage=storage,
                date=self.date,
                product=product,
                product_type=product_type,
                version=version,
                verbose=True,
            )
            base_url = f"ftps://jsimpsonftps.pps.eosdis.nasa.gov/data/{pps_dir}/"
            expected_filepaths = [f"{base_url}{filename}" for filename in self.mock_filenames]
            with check:
                assert returned_filepaths == expected_filepaths

    @pytest.mark.usefixtures("_mock_get_pps_file_list")
    def test_pps_nrt_imerg(
        self,
        check,  # For non-failing asserts
        product_info: dict[str, dict],
    ) -> None:
        """Test _get_all_daily_filepaths for "PPS" storage with NRT IMERG products."""
        storage = "PPS"
        product_type = "NRT"
        product_category = "IMERG"

        for product in available_products(
            product_types=product_type,
            product_categories=product_category,
        ):
            info = product_info[product]
            version = info["available_versions"][-1]
            pps_dir = info["pps_nrt_dir"]

            returned_filepaths = _get_all_daily_filepaths(
                storage=storage,
                date=self.date,
                product=product,
                product_type=product_type,
                version=version,
                verbose=True,
            )
            base_url = f"ftps://jsimpsonftps.pps.eosdis.nasa.gov/data/{pps_dir}/{self.date.strftime('%Y%m')}/"
            expected_filepaths = [f"{base_url}{filename}" for filename in self.mock_filenames]
            with check:
                assert returned_filepaths == expected_filepaths

    def test_pps_missing_pps_product_dir(
        self,
        product_info: dict[str, dict],
        mocker: MockerFixture,
    ) -> None:
        storage = "PPS"
        product = "1A-GMI"
        version = 7
        info = product_info[product]

        # Mock missing dirs
        del info["pps_rs_dir"]
        del info["pps_nrt_dir"]
        mocker.patch("gpm.io.products.get_product_info", return_value=info)

        for product_type in ["RS", "NRT"]:
            with pytest.raises(ValueError):
                _get_all_daily_filepaths(
                    storage=storage,
                    date=self.date,
                    product=product,
                    product_type=product_type,
                    version=version,
                    verbose=True,
                )

    @pytest.fixture
    def _mock_get_ges_disc_list_path(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Mock gpm.io.ges_disc._get_gesc_disc_list_path, which uses wget to get a list of files."""

        def mocked_get_ges_disc_list_path(url: str) -> list[str]:
            return [f"{url}/{filename}" for filename in self.mock_filenames]

        mocker.patch(
            "gpm.io.ges_disc._get_ges_disc_list_path",
            side_effect=mocked_get_ges_disc_list_path,
        )

    @pytest.mark.usefixtures("_mock_get_ges_disc_list_path")
    def test_ges_disc(
        self,
        check,  # For non-failing asserts
        product_info: dict[str, dict],
    ) -> None:
        """Test _get_all_daily_filepaths for "GES_DISC" storage."""
        storage = "GES_DISC"
        version = 7

        for product, info in product_info.items():
            version = info["available_versions"][-1]
            ges_disc_dir = info["ges_disc_dir"]
            if ges_disc_dir is None:
                continue

            returned_filepaths = _get_all_daily_filepaths(
                storage=storage,
                date=self.date,
                product=product,
                product_type=None,
                version=version,
                verbose=True,
            )

            subdomain = "disc2" if "TRMM" in ges_disc_dir else "gpm2"

            base_url = f"https://{subdomain}.gesdisc.eosdis.nasa.gov/data/{ges_disc_dir}.0{version}/{self.date.strftime('%Y/%j')}"
            expected_filepaths = [f"{base_url}/{filename}" for filename in self.mock_filenames]
            with check:
                assert returned_filepaths == expected_filepaths

    def test_invalid_storage(self) -> None:
        """Test _get_all_daily_filepaths for invalid "storage" argument."""
        storage = "invalid"
        product = "1C-GMI"
        product_type = "RS"
        version = 7

        with pytest.raises(ValueError):
            _get_all_daily_filepaths(
                storage=storage,
                date=self.date,
                product=product,
                product_type=product_type,
                version=version,
                verbose=True,
            )


@pytest.mark.filterwarnings("error")
def test_check_correct_version() -> None:
    """Test _check_correct_version."""
    product = "2A-DPR"
    version = 7
    filepath_template = "2A.GPM.DPR.V9-20211125.20200705-S170044-E183317.036092.V0{}A.HDF5"

    # Test correct version
    files_version = [7] * 3
    filepaths = [filepath_template.format(v) for v in files_version]
    returned_filepaths, returned_version = _check_correct_version(
        filepaths=filepaths,
        product=product,
        version=version,
    )
    assert returned_filepaths == filepaths
    assert returned_version == version

    # Test incorrect version
    files_version = [6] * 3
    filepaths = [filepath_template.format(v) for v in files_version]
    with pytest.raises(GPMDownloadWarning):
        _check_correct_version(filepaths=filepaths, product=product, version=version)

    # Test multiple versions
    files_version = [6, 7, 7]
    filepaths = [filepath_template.format(v) for v in files_version]
    with pytest.raises(ValueError):
        _check_correct_version(filepaths=filepaths, product=product, version=version)

    # Test empty list
    filepaths = []
    _, returned_version = _check_correct_version(
        filepaths=filepaths,
        product=product,
        version=version,
    )
    assert returned_version == version


def test_find_daily_filepaths(
    mocker: MockerFixture,
) -> None:
    """Test find_daily_filepaths."""
    storage = "storage"
    date = datetime.datetime(2020, 12, 31)
    product = "product"
    product_type = "product-type"
    version = 7
    start_time = datetime.datetime(2020, 12, 31, 1, 2, 3)
    end_time = datetime.datetime(2020, 12, 31, 4, 5, 6)

    date_checked = datetime.datetime(1900, 1, 1)
    mocker.patch.object(find, "check_date", autospec=True, return_value=date_checked)

    # Mock _get_all_daily_filepaths, already tested above
    def mock_get_all_daily_filepaths(**kwargs: Any) -> list[str]:
        base_filepath = "_".join([f"{key}:{value}" for key, value in kwargs.items()])
        return [f"{base_filepath}_{i}" for i in range(3)]

    patch_get_all_daily_filepaths = mocker.patch.object(
        find,
        "_get_all_daily_filepaths",
        autospec=True,
        side_effect=mock_get_all_daily_filepaths,
    )

    # Mock filter_filepaths, already tested in test_filter.py
    def mock_filter_filepaths(filepaths, **kwargs: Any) -> list[str]:
        suffix = "_".join([f"{key}-filtered:{value}" for key, value in kwargs.items()])
        return [f"{filepath}_{suffix}" for filepath in filepaths]

    patch_filter_filepaths = mocker.patch.object(
        find,
        "filter_filepaths",
        autospec=True,
        side_effect=mock_filter_filepaths,
    )

    # Mock _check_correct_version, already tested above
    def mock_check_correct_version(filepaths, **kwargs: Any) -> tuple[list[str], int]:
        suffix = "_".join([f"{key}-version-checked:{value}" for key, value in kwargs.items()])
        return [f"{filepath}_{suffix}" for filepath in filepaths], version

    mocker.patch.object(
        find,
        "_check_correct_version",
        autospec=True,
        side_effect=mock_check_correct_version,
    )

    kwargs = {
        "storage": storage,
        "date": date,
        "product": product,
        "product_type": product_type,
        "version": version,
        "start_time": start_time,
        "end_time": end_time,
        "verbose": True,
    }

    returned_filepaths, returned_versions = find_daily_filepaths(**kwargs)

    assert returned_versions[0] == version
    returned_filepath = returned_filepaths[0]

    # Check date checked
    assert str(date_checked) in returned_filepath

    # Check all _get_all_daily_filepaths kwargs passed
    assert f"storage:{storage}" in returned_filepath
    assert f"product:{product}" in returned_filepath
    assert f"product_type:{product_type}" in returned_filepath
    assert f"date:{date_checked}" in returned_filepath
    assert f"version:{version}" in returned_filepath
    assert "verbose:True" in returned_filepath

    # Check all filter_filepaths kwargs passed
    assert f"product-filtered:{product}" in returned_filepath
    assert f"product_type-filtered:{product_type}" in returned_filepath
    assert "version-filtered:None" in returned_filepath
    assert f"start_time-filtered:{start_time}" in returned_filepath
    assert f"end_time-filtered:{end_time}" in returned_filepath

    # Check all _check_correct_version kwargs passed
    assert f"product-version-checked:{product}" in returned_filepath
    assert f"version-version-checked:{version}" in returned_filepath

    # Check empty filtered filepaths list
    patch_filter_filepaths.side_effect = lambda filepaths, **kwargs: []
    returned_filepaths, returned_versions = find_daily_filepaths(**kwargs)
    assert returned_filepaths == []
    assert returned_versions == []

    # Check empty filepaths list
    patch_get_all_daily_filepaths.side_effect = lambda **kwargs: []
    kwargs["storage"] = "LOCAL"
    returned_filepaths, returned_versions = find_daily_filepaths(**kwargs)
    assert returned_filepaths == []
    assert returned_versions == []


def test_find_filepaths(
    mocker: MockerFixture,
) -> None:
    """Test find_filepaths.

    Since find_filepaths relies on find_daily_filepaths, we can mock
    find_daily_filepaths and only test some cases here.
    """
    storage = "PPS"
    version = 7
    product = "2A-DPR"
    product_type = "RS"
    start_time = datetime.datetime(2020, 12, 29, 8, 0, 0)
    end_time = datetime.datetime(2020, 12, 31, 8, 0, 0)
    verbose = True

    n_filepath_per_day = 3

    # Mock find_daily_filepaths
    def mock_find_daily_filepaths(**kwargs: Any) -> tuple[list[str], list[int]]:
        base_filepath = "_".join([f"{key}:{value}" for key, value in kwargs.items()])
        return [f"{base_filepath}_{i}" for i in range(n_filepath_per_day)], [
            version,
        ] * n_filepath_per_day

    mocker.patch.object(
        find,
        "find_daily_filepaths",
        autospec=True,
        side_effect=mock_find_daily_filepaths,
    )

    kwargs = {
        "storage": storage,
        "product": product,
        "product_type": product_type,
        "start_time": start_time,
        "end_time": end_time,
        "verbose": verbose,
    }

    returned_filepaths = find_filepaths(**kwargs, parallel=False)
    returned_filepaths_parallel = find_filepaths(**kwargs, parallel=True)
    assert returned_filepaths == returned_filepaths_parallel

    # Check all find_daily_filepaths kwargs passed
    returned_filepath = returned_filepaths[-1]

    # Take last filepath, because "verbose" is not passed to first date
    assert f"storage:{storage}" in returned_filepath
    assert f"version:{version}" in returned_filepath
    assert f"product:{product}" in returned_filepath
    assert f"product_type:{product_type}" in returned_filepath
    assert f"start_time:{start_time}" in returned_filepath
    assert f"end_time:{end_time}" in returned_filepath
    assert f"verbose:{verbose}" in returned_filepath

    # Check that date goes from (start_time - 1) day to end_time
    start_date = datetime.datetime(
        start_time.year,
        start_time.month,
        start_time.day,
    ) - datetime.timedelta(days=1)
    end_date = datetime.datetime(end_time.year, end_time.month, end_time.day)
    n_days = (end_date - start_date).days + 1  # Include last day
    assert len(returned_filepaths) == n_days * n_filepath_per_day, "More days than expected"

    for date in [start_date + datetime.timedelta(days=i) for i in range(n_days)]:
        filtered = list(filter(lambda fp: f"date:{date}" in fp, returned_filepaths))
        assert len(filtered) == n_filepath_per_day, "Date is missing"

    # Test NRT products: single date
    product_type = "NRT"
    kwargs["product_type"] = product_type
    returned_filepaths = find_filepaths(**kwargs, parallel=False)
    assert len(returned_filepaths) == n_filepath_per_day, "More days than expected"
