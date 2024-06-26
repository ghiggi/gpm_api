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
"""This module test the file filtering routines."""

import datetime
from typing import Any

from gpm.io.filter import (
    filter_by_product,
    filter_by_time,
    filter_by_version,
    filter_filepaths,
    is_granule_within_time,
)


def test_granule_within_time() -> None:
    """Test is_granule_within_time()."""
    # Set a file time 01.01.14 01:00 to 04:00 (start, end)
    file_time = ("2014-01-01T01:00:00Z", "2014-01-01T04:00:00Z")

    # Test True assumptions
    true_assumptions = [
        ("2014-01-01T00:00:00Z", "2014-01-01T05:00:00Z"),  # Crosses start
        ("2014-01-01T02:00:00Z", "2014-01-01T03:00:00Z"),  # Within
        ("2014-01-01T03:00:00Z", "2014-01-01T05:00:00Z"),  # Crosses end
    ]

    for start_time, end_time in true_assumptions:
        assert (
            is_granule_within_time(
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
            is_granule_within_time(
                start_time=start_time,
                end_time=end_time,
                file_start_time=file_time[0],
                file_end_time=file_time[1],
            )
            is False
        )


class TestFilterFilepaths:
    """Test filter filepaths."""

    product = "2A-DPR"

    def test_year_filtering(
        self,
        remote_filepaths: dict[str, dict[str, Any]],
    ) -> None:
        # Count and assert 2019 paths
        count_2019 = 0
        for info_dict in remote_filepaths.values():
            if info_dict["year"] == 2019 and info_dict["product"] == self.product and info_dict["version"] == 7:
                count_2019 += 1

        res = filter_filepaths(
            filepaths=list(remote_filepaths.keys()),
            product=self.product,
            start_time=datetime.datetime(2019, 1, 1),
            end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
            version=7,
        )

        assert len(res) == count_2019

    def test_none_filepath(self) -> None:
        res = filter_filepaths(
            filepaths=None,
            product=self.product,
            start_time=datetime.datetime(2019, 1, 1),
            end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
            version=7,
        )
        assert res == []

    def test_empty_filepath_list(self) -> None:
        res = filter_filepaths(
            filepaths=[],
            product=self.product,
            start_time=datetime.datetime(2019, 1, 1),
            end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
            version=7,
        )
        assert res == []

    def test_empty_start_time(
        self,
        remote_filepaths: dict[str, dict[str, Any]],
    ) -> None:
        count_until_2019 = 0
        for info_dict in remote_filepaths.values():
            if info_dict["year"] == 2019 and info_dict["product"] == self.product:
                count_until_2019 += 1
        res = filter_filepaths(
            filepaths=list(remote_filepaths.keys()),
            product=self.product,
            start_time=None,
            end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
            version=7,
        )

        assert len(res) == count_until_2019

    def test_empty_end_time(self, remote_filepaths: dict[str, dict[str, Any]]) -> None:
        """Test empty end_time.

        Error as time given (datetime.datetime.now()) requires date to be less than now() in supportive
        function checks.check_start_end_time).
        """
        count_from_2019 = 0
        for info_dict in remote_filepaths.values():
            if info_dict["year"] >= 2019 and info_dict["product"] == self.product:
                count_from_2019 += 1

        res = filter_filepaths(
            filepaths=list(remote_filepaths.keys()),
            product=self.product,
            start_time=datetime.datetime(2019, 1, 1),
            end_time=None,
            version=7,
        )
        assert len(res) == count_from_2019

    def test_unmatched_version(
        self,
        remote_filepaths: dict[str, dict[str, Any]],
    ) -> None:
        res = filter_filepaths(
            filepaths=list(remote_filepaths.keys()),
            product=self.product,
            start_time=datetime.datetime(2019, 1, 1),
            end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
            version=0,
        )
        assert res == []

    def test_unmatched_product(
        self,
        remote_filepaths: dict[str, dict[str, Any]],
    ) -> None:
        res = filter_filepaths(
            filepaths=list(remote_filepaths.keys()),
            product="1A-GMI",
            start_time=datetime.datetime(2019, 1, 1),
            end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
            version=7,
        )
        assert res == []

    def test_bad_filepath(
        self,
    ) -> None:
        # Test return []
        res = filter_filepaths(filepaths=["bad_filepath"], product="1A-GMI")
        assert res == []


def test_filter_by_time(remote_filepaths: dict[str, dict[str, Any]]) -> None:
    """Test filter filepaths."""
    # Test year filtering
    # Count and assert 2019 paths
    count_2019 = 0
    for info_dict in remote_filepaths.values():
        if info_dict["year"] == 2019:
            count_2019 += 1

    res = filter_by_time(
        filepaths=list(remote_filepaths.keys()),
        start_time=datetime.datetime(2019, 1, 1),
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert len(res) == count_2019

    # Test None filepaths
    res = filter_by_time(
        filepaths=None,
        start_time=datetime.datetime(2019, 1, 1),
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert res == []

    # Test empty filepath list
    res = filter_by_time(
        filepaths=[],
        start_time=datetime.datetime(2019, 1, 1),
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert res == []

    # Test empty start time
    count_until_2019 = 0
    for info_dict in remote_filepaths.values():
        if info_dict["year"] == 2019:
            count_until_2019 += 1
    res = filter_by_time(
        filepaths=list(remote_filepaths.keys()),
        start_time=None,
        end_time=datetime.datetime(2019, 12, 31, 23, 59, 59),
    )

    assert len(res) == count_until_2019

    # Test empty end time (should default to utcnow which will technically be
    # in the past by the time it gets to the function)
    count_from_2019 = 0
    for info_dict in remote_filepaths.values():
        if info_dict["year"] >= 2019:
            count_from_2019 += 1

    res = filter_by_time(
        filepaths=list(remote_filepaths.keys()),
        start_time=datetime.datetime(2019, 1, 1),
        end_time=None,
    )

    # Test granule starting on previous day
    count_previous_day = 0
    for info_dict in remote_filepaths.values():
        if info_dict["start_time"].day != info_dict["end_time"].day:
            count_previous_day += 1

    res = filter_by_time(
        filepaths=list(remote_filepaths.keys()),
        start_time=datetime.datetime(2020, 7, 6, 0, 0, 20),
        end_time=datetime.datetime(2020, 7, 6, 0, 0, 30),
    )

    assert len(res) == count_previous_day


def test_filter_by_product(
    remote_filepaths: dict[str, dict[str, Any]],
) -> None:
    """Test filter by product.

    Use predefined remote_filepaths list to validate filter
    """
    # Check 2A-DPR
    products_2A_DPR = 0
    for info_dict in remote_filepaths.values():
        # Ensure exists in remote_filepath list
        if info_dict["product"] == "2A-DPR":
            products_2A_DPR += 1

    assert products_2A_DPR > 0, "The test remote_filepaths fixture does not contain expected value"

    filtered_filepaths = filter_by_product(
        filepaths=list(remote_filepaths.keys()),
        product="2A-DPR",
    )
    assert len(filtered_filepaths) == products_2A_DPR

    # Test None filepath
    assert filter_by_product(filepaths=None, product="2A-DPR") == []

    # Test empty filepaths
    assert filter_by_product(filepaths=[], product="2A-DPR") == []


def test_filter_by_version(
    remote_filepaths: dict[str, dict[str, Any]],
    versions: list[int],
) -> None:
    """Test filtering by version."""
    # Test each version
    for version in versions:
        paths_with_matching_version = 0
        for info_dict in remote_filepaths.values():
            if info_dict["version"] == version:
                paths_with_matching_version += 1

        # Only test if there are matching versions in remote_filepaths
        if paths_with_matching_version > 0:
            res = filter_by_version(list(remote_filepaths.keys()), version)

            assert len(res) == paths_with_matching_version

        # Test None filepaths
        res = filter_by_version(None, version)
        assert res == []

        # Test empty filepaths list
        res = filter_by_version([], version)
        assert res == []
