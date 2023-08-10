from gpm_api.io import filter
from typing import Dict, Any, List
import datetime
import pytest


def test_granule_within_time() -> None:
    """Test is_granule_within_time()"""

    # Set a file time 01.01.14 01:00 to 04:00 (start, end)
    file_time = ("2014-01-01T01:00:00Z", "2014-01-01T04:00:00Z")

    # Test `True` assumptions
    true_assumptions = [
        ("2014-01-01T00:00:00Z", "2014-01-01T05:00:00Z"),  # Crosses start
        ("2014-01-01T02:00:00Z", "2014-01-01T03:00:00Z"),  # Within
        ("2014-01-01T03:00:00Z", "2014-01-01T05:00:00Z"),  # Crosses end
    ]

    for start_time, end_time in true_assumptions:
        assert (
            filter.is_granule_within_time(
                start_time=start_time,
                end_time=end_time,
                file_start_time=file_time[0],
                file_end_time=file_time[1],
            )
            is True
        )

    # Test `False` assumptions
    false_assumptions = [
        ("2014-01-01T00:00:00Z", "2014-01-01T00:01:01Z"),  # Ends at start
        ("2013-01-01T00:00:00Z", "2013-01-01T05:00:00Z"),  # Before start
        ("2014-01-01T00:00:00Z", "2014-01-01T00:59:59Z"),  # Before start
        ("2014-01-01T05:00:00Z", "2014-01-01T06:00:00Z"),  # After end
        ("2014-01-01T04:00:00Z", "2014-01-01T05:00:00Z"),  # Starts at end
    ]

    for start_time, end_time in false_assumptions:
        assert (
            filter.is_granule_within_time(
                start_time=start_time,
                end_time=end_time,
                file_start_time=file_time[0],
                file_end_time=file_time[1],
            )
            is False
        )


def test_filter_filepaths(
    server_paths: Dict[str, Dict[str, Any]],
    products: Dict[str, Dict[str, Any]],
) -> None:
    """Test filter filepaths"""

    # Test year filtering
    # Count and assert 2019 paths
    count_2019 = 0
    for server_path, props in server_paths.items():
        if props["year"] == 2019 and props["product"] == "2A-DPR":
            count_2019 += 1

    res = filter.filter_filepaths(
        filepaths=list(server_paths.keys()),
        product="2A-DPR",
        start_time=datetime.datetime(2019, 1, 1),
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert len(res) == count_2019


def test_filter_by_time(
    server_paths: Dict[str, Dict[str, Any]],
) -> None:
    """Test filter filepaths"""

    # Test year filtering
    # Count and assert 2019 paths
    count_2019 = 0
    for server_path, props in server_paths.items():
        if props["year"] == 2019:
            count_2019 += 1

    res = filter.filter_by_time(
        filepaths=list(server_paths.keys()),
        start_time=datetime.datetime(2019, 1, 1),
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert len(res) == count_2019

    # Test None filepaths
    res = filter.filter_by_time(
        filepaths=None,
        start_time=datetime.datetime(2019, 1, 1),
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert res == []

    # Test empty filepath list
    res = filter.filter_by_time(
        filepaths=[],
        start_time=datetime.datetime(2019, 1, 1),
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert res == []

    # Test empty start time
    count_until_2019 = 0
    for server_path, props in server_paths.items():
        if props["year"] == 2019:
            count_until_2019 += 1
    res = filter.filter_by_time(
        filepaths=list(server_paths.keys()),
        start_time=None,
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert len(res) == count_until_2019

    # Test empty end time (Error as time given (datetime.datetime.now())
    # requires date to be less than now() in supportive
    # function checks.check_start_end_time)
    count_from_2019 = 0
    for server_path, props in server_paths.items():
        if props["year"] >= 2019:
            count_from_2019 += 1

    with pytest.raises(ValueError):
        res = filter.filter_by_time(
            filepaths=list(server_paths.keys()),
            start_time=datetime.datetime(2019, 1, 1),
            end_time=None,
        )


def test_filter_by_product(
    server_paths: Dict[str, Dict[str, Any]],
    products: List[str],
) -> None:
    products_2A_DPR = 0
    for server_path, props in server_paths.items():
        # Check 2A-DPR
        if props["product"] == "2A-DPR":
            products_2A_DPR += 1

    filter.filter_by_product(
        filepaths=list(server_paths.keys()),
        product="2A-DPR",
    )

    assert len(server_paths) == products_2A_DPR


# for product in products:
#     if product != "2A-DPR" and product not in [
#         props["product"] for x, props in server_paths.items()
#     ]:
#         res = filter.filter_by_product(
#             filepaths=list(server_paths.keys()),
#             product=product,
#         )

#         assert res == []
