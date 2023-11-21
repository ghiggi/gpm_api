#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:41:14 2023

@author: ghiggi
"""

import pytest
import datetime
import numpy as np
import os
import platform
import ntpath as ntp
import posixpath as ptp
import pytz
import pandas as pd
from typing import List, Dict, Any
from gpm_api.io import checks
from gpm_api.io.products import available_products, available_scan_modes, available_versions


def test_check_base_dir() -> None:
    """Check path constructor for base_dir"""

    # Check text entry for Unix/Windows
    if platform.system() == "Windows":
        res = checks.check_base_dir("C:\\Users\\user\\gpm")
        assert res == ntp.join(
            "C:", os.path.sep, "Users", "user", "gpm"
        ), "Windows path is not returned"
    else:
        res = checks.check_base_dir("/home/user/gpm")
        assert res == ptp.join(ptp.sep, "home", "user", "gpm"), "Unix path is not returned"

    # Check final slash is removed
    res = checks.check_base_dir(f"{os.path.join(os.path.expanduser('~'), 'gpm')}{os.path.sep}")
    assert res == os.path.join(os.path.expanduser("~"), "gpm"), "Leading slash is not removed"

    # Check if GPM, it is removed
    res = checks.check_base_dir(os.path.join(os.path.join(os.path.expanduser("~"), "gpm", "GPM")))
    assert res == os.path.join(os.path.join(os.path.expanduser("~"), "gpm")), "GPM is not removed"


def test_check_filepaths() -> None:
    """Check path constructor for filepaths"""

    # Create list of unique filepaths (may not reflect real files)
    filepaths = [
        os.path.join(
            "home",
            "user",
            "gpm",
            "2A.GPM.DPR.V8-20180723.20141231-S003429-E020702.004384.V06A.HDF5",
        ),
        os.path.join(
            "home",
            "user",
            "gpm",
            "2A.GPM.DPR.V8-20180723.20180603-S003429-E020702.004384.V06A.HDF5",
        ),
    ]

    res = checks.check_filepaths(filepaths)
    assert res == filepaths, "List of filepaths is not returned"

    # Check if single string is converted to list
    res = checks.check_filepaths(filepaths[0])
    assert res == [filepaths[0]], "String is not converted to list"

    # Check if not list or string, TypeError is raised
    with pytest.raises(TypeError):
        checks.check_filepaths(123)


def test_check_variables() -> None:
    """Check variables"""

    var_list = ["precipitationCal", "precipitationUncal", "HQprecipitation"]

    # Check if None, None is returned
    res = checks.check_variables(None)
    assert res is None, "None is not returned"

    # Check if string, string is returned
    res = checks.check_variables(var_list[0])
    assert res == [var_list[0]], "String is not returned"

    # Check if list, list is returned
    res = checks.check_variables(var_list)
    assert isinstance(res, np.ndarray), "Array is not returned"

    for var in var_list:
        assert var in res, f"Variable '{var}' is not returned"

    # Check if numpy array, list is returned
    var_list_ndarray = np.array(var_list)
    res = checks.check_variables(var_list_ndarray)
    assert isinstance(res, np.ndarray), "numpy array is not returned"
    assert np.array_equal(res, var_list_ndarray), "Return not equal to input"

    # Check if not list or string, TypeError is raised
    with pytest.raises(TypeError):
        checks.check_variables(123)


def test_check_groups() -> None:
    """Test check_groups()

    Similar logic to check_variables
    """

    group_list = ["NS", "HS", "MS"]

    # Check if None, None is returned
    res = checks.check_groups(None)
    assert res is None, "None is not returned"

    # Check if string, string is returned
    res = checks.check_groups(group_list[0])
    assert res == [group_list[0]], "String is not returned"

    # Check if list, list is returned
    res = checks.check_groups(group_list)
    assert isinstance(res, np.ndarray), "Array is not returned"

    for group in group_list:
        assert group in res, f"Group '{group}' is not returned"

    # Check if numpy array, list is returned
    group_list_ndarray = np.array(group_list)
    res = checks.check_groups(group_list_ndarray)
    assert isinstance(res, np.ndarray), "numpy array is not returned"
    assert np.array_equal(res, group_list_ndarray), "Return not equal to input"

    # Check if not list or string, TypeError is raised
    with pytest.raises(TypeError):
        checks.check_groups(123)


def test_check_storage() -> None:
    """Test check_storage()"""

    # Check valid storage
    valid_storage = ["ges_disc", "pps", "local", "GES_DISC", "PPS", "LOCAL"]
    expected_return = ["ges_disc", "pps", "local", "ges_disc", "pps", "local"]

    for storage, expected in zip(valid_storage, expected_return):
        returned_storage = checks.check_storage(storage)
        assert (
            returned_storage == expected
        ), f"Function returned '{returned_storage}' for storage '{storage}', expected '{expected}'"

    # Check invalid storage
    with pytest.raises(ValueError):
        checks.check_storage("invalid_storage")

    with pytest.raises(TypeError):
        checks.check_storage(123)


def test_check_remote_storage() -> None:
    """Test check_remote_storage()"""

    # Check valid storage
    valid_storage = ["ges_disc", "pps", "GES_DISC", "PPS"]
    expected_return = ["ges_disc", "pps", "ges_disc", "pps"]

    for storage, expected in zip(valid_storage, expected_return):
        returned_storage = checks.check_remote_storage(storage)
        assert (
            returned_storage == expected
        ), f"Function returned '{returned_storage}' for storage '{storage}', expected '{expected}'"

    # Check invalid storage
    with pytest.raises(ValueError):
        checks.check_remote_storage("invalid_storage")

    with pytest.raises(TypeError):
        checks.check_remote_storage(123)


def test_check_version(
    versions: List[int],
) -> None:
    """Test check_version()

    Possible versions are integers of 4-7
    """

    # Check if None, None is returned
    with pytest.raises(ValueError):
        checks.check_version(None)

    # Check if string, exception is raised
    with pytest.raises(ValueError):
        checks.check_version("6A")

    # Check if outside range
    with pytest.raises(ValueError):
        checks.check_version(123)

    # Check available range should not raise exception
    for version in versions:
        checks.check_version(version)
        # Should run without raising Exception

    # Try versions outside of range
    for version in list(range(0, 3)) + list(range(8, 10)):
        with pytest.raises(ValueError):
            checks.check_version(version)


def test_check_product_version(
    check,  # For non-failing asserts
    product_info: Dict[str, Any],
    versions: List[int],
) -> None:
    """Test check_product_version()"""

    for product, info in product_info.items():
        # Check valid versions
        valid_versions = info.get("available_versions", [])

        for version in valid_versions:
            with check:
                assert checks.check_product_version(version, product) == version

        # Check last version return if None
        last_version = info.get("available_versions", [])[-1]
        with check:
            assert checks.check_product_version(None, product) == last_version

        # Check invalid versions
        invalid_versions = list(set(versions) - set(info.get("available_versions", [])))

        for version in invalid_versions:
            with check.raises(ValueError):
                checks.check_product_version(version, product)


def test_check_product(
    product_types: List[str],
) -> None:
    """Test check_product()

    Depends on available_products(), test ambiguous product names similar to
    those that exist
    """

    # Test a product that does exist
    for product_type in product_types:
        for product in available_products(product_type=product_type):
            checks.check_product(product, product_type=product_type)
            # Should run without raising Exception

    # Test a product that isn't a string
    for product_type in product_types:
        for product in [("IMERG"), 123, None]:
            with pytest.raises(ValueError):
                checks.check_product(product, product_type=product_type)


def test_check_product_type(
    product_types: List[str],
) -> None:
    """Test check_product_type()"""

    # Test a product_type that does exist
    for product_type in product_types:
        checks.check_product_type(product_type)
        # Should run without raising Exception

    # Test a product_type that doesn't exist
    for product_type in ["IMERG", 123, None]:
        with pytest.raises(ValueError):
            checks.check_product_type(product_type)


def test_check_product_category(
    product_categories: List[str],
) -> None:
    """Test check_product_category()"""

    # Test types that aren't strings
    for product_category in [123, None]:
        with pytest.raises(ValueError):
            checks.check_product_category(product_category)

    # Test a product_category that does exist
    for product_category in product_categories:
        checks.check_product_category(product_category)
        # Should run without raising Exception

    # Test a product_category that doesn't exist
    for product_category in ["NOT", "A", "CATEGORY"]:
        with pytest.raises(ValueError):
            checks.check_product_category(product_category)


def test_check_product_level(
    product_levels: List[str],
) -> None:
    """Test check_product_level()"""

    # Test types that aren't strings
    for product_level in [123, None]:
        with pytest.raises(ValueError):
            checks.check_product_level(product_level)

    # Test a product_level that does exist
    for product_level in product_levels:
        checks.check_product_level(product_level)
    # Should run without raising Exception

    # Test a product_level that doesn't exist
    for product_level in ["NOT", "A", "LEVEL"]:
        with pytest.raises(ValueError):
            checks.check_product_level(product_level)


def test_check_product_validity(
    product_types: List[str],
) -> None:
    """Test check_product_validity()"""

    # Test a product that does exist
    for product_type in product_types:
        for product in available_products(product_type=product_type):
            checks.check_product_validity(product, product_type=product_type)
            # Should run without raising Exception

    # Test a product that doesn't exist
    for product_type in product_types:
        for product in [("IMERG"), 123, None]:
            with pytest.raises(ValueError):
                checks.check_product_validity(product, product_type=product_type)
            # Test a None product type
            with pytest.raises(ValueError):
                checks.check_product_validity(product, product_type=None)


def test_check_time() -> None:
    """Test that time is returned a datetime object from varying inputs"""

    # Test a string
    res = checks.check_time("2014-12-31")
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31)

    # Test a string with hh/mm/ss
    res = checks.check_time("2014-12-31 12:30:30")
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31, 12, 30, 30)

    # Test a string with <date>T<time>
    res = checks.check_time("2014-12-31T12:30:30")
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31, 12, 30, 30)

    # Test a datetime object
    res = checks.check_time(datetime.datetime(2014, 12, 31))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31)

    # Test a datetime timestamp with h/m/s/ms
    res = checks.check_time(datetime.datetime(2014, 12, 31, 12, 30, 30, 300))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31, 12, 30, 30, 300)

    # Test a np.datetime64 object of "datetime64[s]"
    res = checks.check_time(np.datetime64("2014-12-31"))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31)

    # Test a object of datetime64[ns] casts to datetime64[ms]
    res = checks.check_time(np.datetime64("2014-12-31T12:30:30.934549845", "s"))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31, 12, 30, 30)

    # Test a datetime.date
    res = checks.check_time(datetime.date(2014, 12, 31))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31)

    # Test a datetime object inside a numpy array
    with pytest.raises(ValueError):
        res = checks.check_time(np.array([datetime.datetime(2014, 12, 31, 12, 30, 30)]))
        assert isinstance(res, datetime.datetime)
        assert res == datetime.datetime(2014, 12, 31, 12, 30, 30)

    # Test a pandas Timestamp object inside a numpy array
    with pytest.raises(ValueError):
        res = checks.check_time(np.array([pd.Timestamp("2014-12-31 12:30:30")]))
        assert isinstance(res, datetime.datetime)
        assert res == datetime.datetime(2014, 12, 31, 12, 30, 30)

    # Test a pandas Timestamp object
    res = checks.check_time(pd.Timestamp("2014-12-31 12:30:30"))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31, 12, 30, 30)

    # Test automatic casting to seconds accuracy
    res = checks.check_time(np.datetime64("2014-12-31T12:30:30.934549845", "ns"))
    assert res == datetime.datetime(2014, 12, 31, 12, 30, 30)

    # Test a non isoformat string
    with pytest.raises(ValueError):
        checks.check_time("2014/12/31")

    # Test a non datetime object
    with pytest.raises(TypeError):
        checks.check_time(123)

    # Check numpy single timestamp
    res = checks.check_time(np.array(["2014-12-31"], dtype="datetime64[s]"))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31)

    # Check numpy multiple timestamp
    with pytest.raises(ValueError):
        checks.check_time(np.array(["2014-12-31", "2015-01-01"], dtype="datetime64[s]"))

    # Test with numpy non datetime64 object
    with pytest.raises(ValueError):
        checks.check_time(np.array(["2014-12-31"]))

    # Check non-UTC timezone
    with pytest.raises(ValueError):
        checks.check_time(
            datetime.datetime(2014, 12, 31, 12, 30, 30, 300, tzinfo=pytz.timezone("Europe/Zurich"))
        )


def test_check_date() -> None:
    """Check date/datetime object is returned from varying inputs"""

    # Test a datetime object
    res = checks.check_date(datetime.datetime(2014, 12, 31))
    assert isinstance(res, datetime.date)
    assert res == datetime.date(2014, 12, 31)

    # Test a datetime timestamp with h/m/s/ms
    res = checks.check_date(datetime.datetime(2014, 12, 31, 12, 30, 30, 300))
    assert isinstance(res, datetime.date)
    assert res == datetime.date(2014, 12, 31)

    # Test a string is cast to date
    res = checks.check_date("2014-12-31")
    assert isinstance(res, datetime.date)

    # Test a np datetime object is cast to date
    res = checks.check_date(np.datetime64("2014-12-31"))
    assert isinstance(res, datetime.date)

    # Test None raises exception
    with pytest.raises(ValueError):
        checks.check_date(None)


def test_check_start_end_time() -> None:
    """Check start and end time are valid"""

    # Test a string
    res = checks.check_start_end_time(
        "2014-12-31",
        "2015-01-01",
    )
    assert isinstance(res, tuple)

    # Test the reverse for exception
    with pytest.raises(ValueError):
        checks.check_start_end_time(
            "2015-01-01",
            "2014-12-31",
        )

    # Test a datetime object
    res = checks.check_start_end_time(
        datetime.datetime(2014, 12, 31),
        datetime.datetime(2015, 1, 1),
    )
    assert isinstance(res, tuple)

    # Test the reverse datetime object for exception
    with pytest.raises(ValueError):
        checks.check_start_end_time(
            datetime.datetime(2015, 1, 1),
            datetime.datetime(2014, 12, 31),
        )

    # Test a datetime timestamp with h/m/s/ms
    res = checks.check_start_end_time(
        datetime.datetime(2014, 12, 31, 12, 30, 30, 300),
        datetime.datetime(2015, 1, 1, 12, 30, 30, 300),
    )

    # Test end time in the future
    with pytest.raises(ValueError):
        checks.check_start_end_time(
            datetime.datetime(2014, 12, 31, 12, 30, 30, 300),
            datetime.datetime(2125, 1, 1, 12, 30, 30, 300),
        )

    # Test start time in the future
    with pytest.raises(ValueError):
        checks.check_start_end_time(
            datetime.datetime(2125, 12, 31, 12, 30, 30, 300),
            datetime.datetime(2126, 1, 1, 12, 30, 30, 300),
        )

    # Check a time that is generated in another timezone but does not directly
    # carry timezone information. This should fail if the check is done on utcnow()
    with pytest.raises(ValueError):
        for timezone in ["Europe/Zurich", "Australia/Melbourne"]:
            # Remove timezone information
            checks.check_start_end_time(
                datetime.datetime(2014, 12, 31, 12, 30, 30, 300),
                datetime.datetime.now(tz=pytz.timezone(timezone)).replace(tzinfo=None),
            )
            # Keep timezone information, should throw exception
            checks.check_start_end_time(
                datetime.datetime(2014, 12, 31, 12, 30, 30, 300),
                datetime.datetime.now(tz=pytz.timezone(timezone)),
            )

    # This should pass as the time is in UTC
    checks.check_start_end_time(
        datetime.datetime(2014, 12, 31, 12, 30, 30, 300),
        datetime.datetime.now(tz=pytz.utc),
    )

    # Do the same but in a timezone that is behind UTC (this should pass)
    for timezone in ["America/New_York", "America/Santiago"]:
        checks.check_start_end_time(
            datetime.datetime(2014, 12, 31, 12, 30, 30, 300),
            datetime.datetime.now(tz=pytz.timezone(timezone)).replace(tzinfo=None),
        )

    # Test endtime in UTC. This should pass as UTC time generated in the test is slightly
    # behind the current time tested in the function
    checks.check_start_end_time(
        datetime.datetime(2014, 12, 31, 12, 30, 30, 300),
        datetime.datetime.utcnow(),
    )


def test_check_valid_time_request(
    check,  # For non-failing asserts
    product_info: Dict[str, Any],
) -> None:
    """Test check_valid_time_request()"""

    for product, info in product_info.items():
        valid_start_time = info["start_time"]
        valid_end_time = info["end_time"]

        if valid_start_time is not None:
            # Check valid times
            start_time = valid_start_time
            end_time = valid_start_time + datetime.timedelta(days=1)
            checks.check_valid_time_request(start_time, end_time, product)

            # Check invalid start time
            start_time = valid_start_time - datetime.timedelta(days=1)
            end_time = valid_start_time + datetime.timedelta(days=1)
            with check.raises(ValueError):
                checks.check_valid_time_request(start_time, end_time, product)

        # Check invalid end time
        if valid_end_time is not None:
            start_time = valid_end_time - datetime.timedelta(days=1)
            end_time = valid_end_time + datetime.timedelta(days=1)
            with check.raises(ValueError):
                checks.check_valid_time_request(start_time, end_time, product)


def test_check_scan_mode(
    products: List[str],
) -> None:
    """Check scan mode is valid"""

    for product in products:
        for version in available_versions(product):
            # Get available scan modes
            scan_modes = available_scan_modes(product, version)

            for scan_mode in scan_modes:
                res = checks.check_scan_mode(scan_mode, product, version)
                assert (
                    res == res
                ), f"Function returned {res} for scan_mode {scan_mode}, expected {scan_mode}"

            # Test a scan mode that doesn't exist
            for scan_mode in ["NOT", "A", "SCAN", "MODE"]:
                with pytest.raises(ValueError):
                    checks.check_scan_mode(scan_mode, product, version)

            # Try to have function infer scan mode
            res = checks.check_scan_mode(None, product, version)
            assert (
                res in scan_modes
            ), f"Function returned {res} for scan_mode {scan_mode}, expected {scan_mode}"

            # Test a scan mode that isn't a string
            with pytest.raises(ValueError):
                checks.check_scan_mode(123, product, version)
