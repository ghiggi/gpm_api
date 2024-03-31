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
"""This module test the GES DISC file search routines."""
import datetime

import pytest
from pytest_mock.plugin import MockerFixture

from gpm.io import ges_disc


def test_get_ges_disc_list_path():
    """Test _get_ges_disc_list_path"""
    # Empty directory
    url = "https://gpm2.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.07/"
    with pytest.raises(ValueError) as excinfo:
        list_path = ges_disc._get_ges_disc_list_path(url)
    assert "directory is empty" in str(excinfo.value)

    # Year directory
    url = "https://gpm2.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2ADPR.07/"
    list_path = ges_disc._get_ges_disc_list_path(url)
    assert len(list_path) > 0

    # File directory
    url = "https://gpm2.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2ADPR.07/2019/006/"
    list_path = ges_disc._get_ges_disc_list_path(url)
    assert len(list_path) > 0

    # Wrong URL
    url = "BAD_URL"
    with pytest.raises(ValueError) as excinfo:
        list_path = ges_disc._get_ges_disc_list_path(url)
    assert f"The requested url {url} was not found on the GES DISC server." == str(excinfo.value)

    # Unexisting directory
    url = "https://gpm2.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHHE.07/20020"
    with pytest.raises(ValueError) as excinfo:
        list_path = ges_disc._get_ges_disc_list_path(url)
    assert f"The requested url {url} was not found on the GES DISC server." == str(excinfo.value)


def test_define_ges_disc_filepath():
    # Test request of NRT but not IMERG
    with pytest.raises(ValueError):
        ges_disc.define_ges_disc_filepath(
            product="2A-DPR",
            product_type="NRT",
            date="DUMMY",
            version=1,
            filename="BIDDIBIBBODIBIBU",
        )
    # Test for valid name
    expected_url = "https://gpm2.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2ADPR.07/2019/006/2A.GPM.DPR.V9-20211125.20190106-S020627-E033859.027589.V07A.HDF5"
    filename = "2A.GPM.DPR.V9-20211125.20190106-S020627-E033859.027589.V07A.HDF5"
    version = 7
    product_type = "RS"
    date = datetime.date(2019, 1, 6)
    url = ges_disc.define_ges_disc_filepath(
        product="2A-DPR",
        product_type=product_type,
        date=date,
        version=version,
        filename=filename,
    )
    assert url == expected_url


class TestGESDISCFileList:
    def test_success(self, mocker: MockerFixture):
        url = "http://example.com/products/"
        date = datetime.date(2020, 7, 5)
        product = "2A-DPR"
        version = "07"

        expected_filepaths = ["https://example.com/file1.hdf5", "https://example.com/file2.hdf5"]

        mocker.patch.object(
            ges_disc, "_get_ges_disc_list_path", autospec=True, return_value=expected_filepaths
        )

        filepaths = ges_disc._get_ges_disc_file_list(url, product, date, version, verbose=True)
        assert filepaths == expected_filepaths, "File paths do not match expected output"

    def test_not_found_error(self, mocker: MockerFixture):
        url = "http://wrong.url/"
        date = datetime.date(2020, 7, 5)
        product = "2A-DPR"
        version = "07"

        mocker.patch.object(
            ges_disc,
            "_get_ges_disc_list_path",
            side_effect=Exception("was not found on the GES DISC server"),
        )

        with pytest.raises(Exception) as excinfo:
            ges_disc._get_ges_disc_file_list(url, product, date, version)
        assert "was not found on the GES DISC server" in str(
            excinfo.value
        ), "Expected exception not raised for server not found"

    def test_no_data_verbose(self, mocker: MockerFixture, capsys):
        url = "http://example.com/products/"
        date = datetime.date(2020, 7, 5)
        product = "2A-DPR"
        version = "07"

        mocker.patch.object(
            ges_disc,
            "_get_ges_disc_list_path",
            autospec=True,
            side_effect=Exception("some other error"),
        )

        filepaths = ges_disc._get_ges_disc_file_list(url, product, date, version, verbose=True)
        assert filepaths == [], "Expected empty list for no data found"

        captured = capsys.readouterr()
        assert "No data found on GES DISC" in captured.out, "Expected verbose message not printed"

    def test_no_data_not_verbose(self, mocker: MockerFixture, capsys):
        url = "http://example.com/products/"
        date = datetime.date(2020, 7, 5)
        product = "2A-DPR"
        version = "07"

        mocker.patch.object(
            ges_disc,
            "_get_ges_disc_list_path",
            autospec=True,
            side_effect=Exception("some other error"),
        )

        filepaths = ges_disc._get_ges_disc_file_list(url, product, date, version, verbose=False)
        assert filepaths == [], "Expected empty list for no data found"

        captured = capsys.readouterr()
        assert captured.out == "", "No output expected when verbose is False"
