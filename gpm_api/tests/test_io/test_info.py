import pytest
from gpm_api.io import info
from typing import Any, Dict


def test_get_start_time_from_filepaths(
    server_paths: Dict[str, Dict[str, Any]],
) -> None:
    """Test that the start time is correctly extracted from filepaths"""

    # Although can be done as a test. To ensure the list order is identical
    # do individually.
    for server_path, info_dict in server_paths.items():
        generated_start_time = info.get_start_time_from_filepaths(server_path)
        assert [info_dict["start_time"]] == generated_start_time


def test_get_end_time_from_filepaths(
    server_paths: Dict[str, Dict[str, Any]],
) -> None:
    """Test that the end time is correctly extracted from filepaths"""

    # Although can be done as a test. To ensure the list order is identical
    # do individually.
    for server_path, info_dict in server_paths.items():
        generated_end_time = info.get_end_time_from_filepaths(server_path)
        assert [info_dict["end_time"]] == generated_end_time


def test_get_version_from_filepaths(
    server_paths: Dict[str, Dict[str, Any]],
) -> None:
    """Test that the version is correctly extracted from filepaths"""

    # Although can be done as a test. To ensure the list order is identical
    # do individually.
    for server_path, info_dict in server_paths.items():
        generated_version = info.get_version_from_filepaths(server_path)

        assert [info_dict["version"]] == generated_version