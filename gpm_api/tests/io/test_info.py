import pytest
from gpm_api.io import info
from typing import Any


def test_get_start_time_from_filepaths(
    server_paths: dict[str, dict[str, Any]],
) -> None:
    """Test that the start time is correctly extracted from filepaths"""

    # Although can be done as a test. To ensure the list order is identical
    # do individually.
    for server_path, props in server_paths.items():
        generated_start_time = info.get_start_time_from_filepaths(server_path)
        assert [props["start_time"]] == generated_start_time


def test_get_end_time_from_filepaths(
    server_paths: dict[str, dict[str, Any]],
) -> None:
    """Test that the end time is correctly extracted from filepaths"""

    # Although can be done as a test. To ensure the list order is identical
    # do individually.
    for server_path, props in server_paths.items():
        generated_end_time = info.get_end_time_from_filepaths(server_path)
        assert [props["end_time"]] == generated_end_time


def test_get_version_from_filepaths(
    server_paths: dict[str, dict[str, Any]],
) -> None:
    """Test that the version is correctly extracted from filepaths"""

    # Although can be done as a test. To ensure the list order is identical
    # do individually.
    for server_path, props in server_paths.items():
        generated_version = info.get_version_from_filepaths(server_path)

        assert [props["version"]] == generated_version
