#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:41:14 2023

@author: ghiggi
"""

import pytest
import datetime
import numpy as np
import platform
from gpm_api.io import checks
from gpm_api.io.products import available_products, available_scan_modes


def test_is_not_empty() -> None:
    """Test is_not_empty()"""

    # Test a non empty object

    res = checks.is_not_empty([1, 2, 3])
    assert res is True, "Function returned False, expected True"

    # Test an empty object
    for empty_object in [None, False, True, (), {}, []]:
        res = checks.is_not_empty(empty_object)
    assert res is False, "Function returned True, expected False"


def test_is_empty() -> None:
    """Test is_empty()"""

    # Test a non empty object
    res = checks.is_empty([1, 2, 3])
    assert res is False, "Function returned True, expected False"

    # Test an empty object
    for empty_object in [None, False, True, (), {}, []]:
        res = checks.is_empty(empty_object)
        assert res is True, "Function returned False, expected True"


def test_check_base_dir() -> None:
    """Check path constructor for base_dir"""

    # Check leading slash is removed
    res = checks.check_base_dir("/home/user/gpm/")
    assert res == "/home/user/gpm", "Leading slash is not removed"

    # Check leading slash is removed
    res = checks.check_base_dir("/home/user/gpm")
    assert res == "/home/user/gpm", "Leading slash is not removed"

    # Check if GPM, it is removed
    res = checks.check_base_dir("/home/user/gpm/GPM")
    assert res == "/home/user/gpm", "GPM is not removed"

    # # Check windows path
    # res = checks.check_base_dir("C:\\home\\user\\gpm\\GPM")
    # assert res == "C:\\home\\user\\gpm", "GPM is not removed"


def test_check_filepaths() -> None:
    """Check path constructor for filepaths"""

    # Create list of unique filepaths (may not reflect real files)
    filepaths = [
        "/home/user/gpm/2A.GPM.DPR.V8-20180723.20141231-S003429-E020702.004384.V06A.HDF5",
        "/home/user/gpm/2A.GPM.DPR.V8-20180723.20180603-S003429-E020702.004384.V06A.HDF5",
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


def test_check_version(
    versions: list[int],
) -> None:
    """Test check_version()

    Possible versions are integers of 4-7
    """

    # Check if None, None is returned
    with pytest.raises(ValueError):
        res = checks.check_version(None)

    # Check if string, exception is raised
    with pytest.raises(ValueError):
        checks.check_version("6A")

    # Check if outside range
    with pytest.raises(ValueError):
        checks.check_version(123)

    # Check available range should not raise exception
    for version in versions:
        res = checks.check_version(version)
        assert res is None, f"Function returned {res} for version {version}, expected None"

    # Try versions outside of range
    for version in list(range(0, 3)) + list(range(8, 10)):
        with pytest.raises(ValueError):
            checks.check_version(version)


def test_check_product(
    product_types: list[str],
) -> None:
    """Test check_product()

    Depends on available_products(), test ambiguous product names similar to
    those that exist
    """

    # Test a product that does exist
    for product_type in product_types:
        for product in available_products(product_type=product_type):
            res = checks.check_product(product, product_type=product_type)
            assert res is None, f"Function returned {res} for product {product} expected None"

    # Test a product that isn't a string
    for product_type in product_types:
        for product in [("IMERG"), 123, None]:
            with pytest.raises(ValueError):
                checks.check_product(product, product_type=product_type)


def test_check_product_type(
    product_types: list[str],
) -> None:
    """Test check_product_type()"""

    # Test a product_type that does exist
    for product_type in product_types:
        res = checks.check_product_type(product_type)
        assert res is None, (
            f"Function returned {res} for product_type {product_type}, " f"expected None"
        )

    # Test a product_type that doesn't exist
    for product_type in ["IMERG", 123, None]:
        with pytest.raises(ValueError):
            checks.check_product_type(product_type)


def test_check_product_category(
    product_categories: list[str],
) -> None:
    """Test check_product_category()"""

    # Test types that aren't strings
    for product_category in [123, None]:
        with pytest.raises(ValueError):
            checks.check_product_category(product_category)

    # Test a product_category that does exist
    for product_category in product_categories:
        res = checks.check_product_category(product_category)
        assert res is None, (
            f"Function returned {res} for product_category {product_category}," f" expected None"
        )

    # Test a product_category that doesn't exist
    for product_category in ["NOT", "A", "CATEGORY"]:
        with pytest.raises(ValueError):
            checks.check_product_category(product_category)


def test_check_product_level(
    product_levels: list[str],
) -> None:
    """Test check_product_level()"""

    # Test types that aren't strings
    for product_level in [123, None]:
        with pytest.raises(ValueError):
            checks.check_product_level(product_level)

    # Test a product_level that does exist
    for product_level in product_levels:
        res = checks.check_product_level(product_level)
        assert (
            res is None
        ), f"Function returned {res} for product_level {product_level}, expected None"

    # Test a product_level that doesn't exist
    for product_level in ["NOT", "A", "LEVEL"]:
        with pytest.raises(ValueError):
            checks.check_product_level(product_level)


def test_check_product_validity(
    product_types: list[str],
) -> None:
    """Test check_product_validity()"""

    # Test a product that does exist
    for product_type in product_types:
        for product in available_products(product_type=product_type):
            res = checks.check_product_validity(product, product_type=product_type)
            assert res is None, f"Function returned {res} for product {product}, expected None"

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
    res = checks.check_time(np.datetime64("2014-12-31T12:30:30.934549845", "ns"))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31, 12, 30, 30, 934550)

    # Test a datetime.date
    res = checks.check_time(datetime.date(2014, 12, 31))
    assert isinstance(res, datetime.datetime)
    assert res == datetime.datetime(2014, 12, 31)

    # Test a non isoformat string
    with pytest.raises(ValueError):
        checks.check_time("2014/12/31")

    # Test a non datetime object
    with pytest.raises(TypeError):
        checks.check_time(123)

    # Check numpy multiple timestamp
    with pytest.raises(ValueError):
        checks.check_time(np.array(["2014-12-31", "2015-01-01"], dtype="datetime64[s]"))

    # Test with numpy non datetime64 object
    with pytest.raises(ValueError):
        checks.check_time(np.array(["2014-12-31"]))


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
    res = checks.check_start_end_time("2014-12-31", "2015-01-01")
    assert isinstance(res, tuple)

    # Test the reverse for exception
    with pytest.raises(ValueError):
        checks.check_start_end_time("2015-01-01", "2014-12-31")

    # Test a datetime object
    res = checks.check_start_end_time(
        datetime.datetime(2014, 12, 31), datetime.datetime(2015, 1, 1)
    )
    assert isinstance(res, tuple)

    # Test the reverse datetime object for exception
    with pytest.raises(ValueError):
        checks.check_start_end_time(datetime.datetime(2015, 1, 1), datetime.datetime(2014, 12, 31))

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


def test_check_scan_mode(
    versions: list[int],
    products: list[str],
) -> None:
    """Check scan mode is valid"""

    for product in products:
        for version in versions:
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
