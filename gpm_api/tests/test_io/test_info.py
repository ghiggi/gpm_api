import pytest
from gpm_api.io import info
from typing import Any, Dict


def test_get_start_time_from_filepaths(
    remote_filepaths: Dict[str, Dict[str, Any]],
) -> None:
    """Test that the start time is correctly extracted from filepaths"""

    # Although can be done as a test. To ensure the list order is identical
    # do individually.
    for remote_filepath, info_dict in remote_filepaths.items():
        generated_start_time = info.get_start_time_from_filepaths(remote_filepath)
        assert [info_dict["start_time"]] == generated_start_time

    # Also test all the filepaths at once
    generated_start_time = info.get_start_time_from_filepaths(list(remote_filepaths.keys()))
    expected_start_time = [info_dict["start_time"] for info_dict in remote_filepaths.values()]
    assert expected_start_time == generated_start_time


def test_get_end_time_from_filepaths(
    remote_filepaths: Dict[str, Dict[str, Any]],
) -> None:
    """Test that the end time is correctly extracted from filepaths"""

    # Although can be done as a test. To ensure the list order is identical
    # do individually.
    for remote_filepath, info_dict in remote_filepaths.items():
        generated_end_time = info.get_end_time_from_filepaths(remote_filepath)
        assert [info_dict["end_time"]] == generated_end_time


def test_get_version_from_filepaths(
    remote_filepaths: Dict[str, Dict[str, Any]],
) -> None:
    """Test that the version is correctly extracted from filepaths"""

    # Although can be done as a test. To ensure the list order is identical
    # do individually.
    for remote_filepath, info_dict in remote_filepaths.items():
        generated_version = info.get_version_from_filepaths(remote_filepath)

        assert [info_dict["version"]] == generated_version


def test_get_granule_from_filepaths(
    remote_filepaths: Dict[str, Dict[str, Any]],
) -> None:
    """Test get_granule_from_filepaths"""

    for remote_filepath, info_dict in remote_filepaths.items():
        if info_dict["product_type"] == "NRT":
            continue

        generated_granule = info.get_granule_from_filepaths(remote_filepath)
        assert [info_dict["granule_id"]] == generated_granule


def test_get_start_end_time_from_filepaths(
    remote_filepaths: Dict[str, Dict[str, Any]],
) -> None:
    """Test get_start_end_time_from_filepaths"""

    for remote_filepath, info_dict in remote_filepaths.items():
        generated_start_time, generated_end_time = info.get_start_end_time_from_filepaths(
            remote_filepath
        )
        assert [info_dict["start_time"]] == generated_start_time
        assert [info_dict["end_time"]] == generated_end_time


def test_invalid_filepaths():
    with pytest.raises(ValueError):
        info.get_info_from_filepath("invalid_filepath")

    # Invalid JAXA product type
    with pytest.raises(ValueError):
        info.get_info_from_filepath(
            "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/1B/GPMCOR_KAR_2007050002_0135_036081_1B😵_DAB_07A.h5"
        )

    # Unknown product (not in product_def.yml)
    with pytest.raises(ValueError):
        info.get_info_from_filepath(
            "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/😥.GPM.DPR.V9-20211125.20200705-S170044-E183317.036092.V07A.HDF5"
        )

    # Filepath not a string
    with pytest.raises(TypeError):
        info.get_info_from_filepath(123)
