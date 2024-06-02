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
"""This module test the info extraction from GPM filename."""

import datetime
from typing import Any

import pytest

from gpm.io.info import (
    FILE_KEYS,
    TIME_KEYS,
    check_groups,
    get_end_time_from_filepaths,
    get_granule_from_filepaths,
    get_info_from_filepath,
    get_product_from_filepaths,
    get_season,
    get_start_end_time_from_filepaths,
    get_start_time_from_filepaths,
    get_time_component,
    get_version_from_filepaths,
    group_filepaths,
)


def test_get_start_time_from_filepaths(
    remote_filepaths: dict[str, dict[str, Any]],
) -> None:
    """Test that the start time is correctly extracted from filepaths."""
    # Although can be done as a test. To ensure the list order is identical
    # do individually.
    for remote_filepath, info_dict in remote_filepaths.items():
        generated_start_time = get_start_time_from_filepaths(remote_filepath)
        assert [info_dict["start_time"]] == generated_start_time

    # Also test all the filepaths at once
    generated_start_time = get_start_time_from_filepaths(list(remote_filepaths.keys()))
    expected_start_time = [info_dict["start_time"] for info_dict in remote_filepaths.values()]
    assert expected_start_time == generated_start_time


def test_get_end_time_from_filepaths(
    remote_filepaths: dict[str, dict[str, Any]],
) -> None:
    """Test that the end time is correctly extracted from filepaths."""
    # Although can be done as a test. To ensure the list order is identical
    # do individually.
    for remote_filepath, info_dict in remote_filepaths.items():
        generated_end_time = get_end_time_from_filepaths(remote_filepath)
        assert [info_dict["end_time"]] == generated_end_time


def test_get_version_from_filepaths(
    remote_filepaths: dict[str, dict[str, Any]],
) -> None:
    """Test that the version is correctly extracted from filepaths."""
    # Although can be done as a test. To ensure the list order is identical
    # do individually.
    for remote_filepath, info_dict in remote_filepaths.items():
        generated_version = get_version_from_filepaths(remote_filepath)

        assert [info_dict["version"]] == generated_version


def test_get_granule_from_filepaths(
    remote_filepaths: dict[str, dict[str, Any]],
) -> None:
    """Test get_granule_from_filepaths."""
    for remote_filepath, info_dict in remote_filepaths.items():
        if info_dict["product_type"] == "NRT":
            continue

        generated_granule = get_granule_from_filepaths(remote_filepath)
        assert [info_dict["granule_id"]] == generated_granule


def test_get_start_end_time_from_filepaths(
    remote_filepaths: dict[str, dict[str, Any]],
) -> None:
    """Test get_start_end_time_from_filepaths."""
    for remote_filepath, info_dict in remote_filepaths.items():
        generated_start_time, generated_end_time = get_start_end_time_from_filepaths(
            remote_filepath,
        )
        assert [info_dict["start_time"]] == generated_start_time
        assert [info_dict["end_time"]] == generated_end_time


def test_get_product_from_filepaths(
    remote_filepaths: dict[str, dict[str, Any]],
) -> None:
    """Test get_product_from_filepaths."""
    for remote_filepath, info_dict in remote_filepaths.items():
        product = get_product_from_filepaths(remote_filepath)
        assert [info_dict["product"]] == product


def test_invalid_filepaths():
    with pytest.raises(ValueError):
        get_info_from_filepath("invalid_filepath")

    # Invalid JAXA product type
    with pytest.raises(ValueError):
        get_info_from_filepath(
            "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/1B/GPMCOR_KAR_2007050002_0135_036081_1B😵_DAB_07A.h5",
        )

    # Unknown product (not in products.yaml)
    with pytest.raises(ValueError):
        get_info_from_filepath(
            "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/😥.GPM.DPR.V9-20211125.20200705-S170044-E183317.036092.V07A.HDF5",
        )

    # Filepath not a string
    with pytest.raises(TypeError):
        get_info_from_filepath(123)


def test_check_groups():
    """Test check_groups function."""
    valid_groups = ["product_level", "satellite", "sensor", "year"]
    assert check_groups(valid_groups) == valid_groups

    invalid_groups = ["invalid1", "invalid2"]
    with pytest.raises(ValueError):
        check_groups(invalid_groups)

    with pytest.raises(TypeError):
        check_groups(1)


def test_get_time_component():
    """Test get_time_component function."""
    start_time = datetime.datetime(2020, 2, 1, 2, 3, 4)
    assert get_time_component(start_time, "year") == "2020"
    assert get_time_component(start_time, "month") == "2"
    assert get_time_component(start_time, "day") == "1"
    assert get_time_component(start_time, "doy") == "32"
    assert get_time_component(start_time, "dow") == "5"
    assert get_time_component(start_time, "hour") == "2"
    assert get_time_component(start_time, "minute") == "3"
    assert get_time_component(start_time, "second") == "4"
    assert get_time_component(start_time, "month_name") == "February"
    assert get_time_component(start_time, "quarter") == "1"
    assert get_time_component(start_time, "season") == "DJF"


def test_get_season():
    """Test get_season function."""
    assert get_season(datetime.datetime(2020, 1, 1)) == "DJF"
    assert get_season(datetime.datetime(2020, 4, 1)) == "MAM"
    assert get_season(datetime.datetime(2020, 7, 1)) == "JJA"
    assert get_season(datetime.datetime(2020, 10, 1)) == "SON"


def test_group_filepaths():
    """Test group_filepaths function."""
    filepaths = [
        # RS NASA
        "/home/ghiggi/data/GPM/RS/V05/PMW/1A-GMI/2015/08/01/1A.GPM.GMI.COUNT2016.20150801-S144642-E161915.008090.V05A.HDF5",
        # NRT NASA
        "/home/ghiggi/data/GPM/NRT/RADAR/2A-DPR/2022/08/18/2A.GPM.DPR.V920211125.20220818-S004128-E011127.V07A.RT-H5",
        # JAXA
        "/home/ghiggi/data/GPM/RS/V07/RADAR/1B-Ka/2020/10/28/GPMCOR_KAR_2010280754_0927_037875_1BS_DAB_07A.h5",
        # IMERG
        "/home/ghiggi/data/GPM/RS/V07/IMERG/IMERG-FR/2011/06/13/3B-HHR.MS.MRG.3IMERG.20110613-S110000-E112959.0660.V07B.HDF5",
    ]

    # Test groups = None
    assert group_filepaths(filepaths, None) == filepaths

    # Test single group
    assert group_filepaths([filepaths[0]], "year") == {"2015": [filepaths[0]]}

    # Test multiple groups
    assert group_filepaths([filepaths[0]], groups=["sensor", "year", "month"]) == {"GMI/2015/8": [filepaths[0]]}

    # Test all time keys pass
    assert group_filepaths(filepaths, TIME_KEYS)

    # Test all file keys pass
    assert group_filepaths(filepaths, FILE_KEYS)
