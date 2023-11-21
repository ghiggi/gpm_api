from gpm_api.io import data_integrity as di
from typing import List, Tuple
import os
import posixpath as pxp
import ntpath as ntp
import xarray as xr


def test_get_corrupted_filepaths(
    local_filepaths_unix: List[str],
    local_filepaths_windows: List[str],
) -> None:
    """Test get_corrupted_filepaths function"""

    # Test that all paths are "corrupted" (in this case there is no data)
    for abs_paths in [local_filepaths_unix, local_filepaths_windows]:
        res = di.get_corrupted_filepaths(abs_paths)

        assert len(abs_paths) == len(
            res
        ), "Corrupted paths array should be the same length as input paths"
        assert abs_paths == res, "Corrupted paths array should be the same as input paths"


def test_get_corrupted_filepaths_real_files(
    tmpdir: str,
) -> None:
    filepath = os.path.join(tmpdir, "test.h5")

    # Create hdf5 file
    array = xr.DataArray([])
    array.to_netcdf(filepath)

    # Test that no paths are "corrupted"
    res = di.get_corrupted_filepaths([filepath])
    assert len(res) == 0, "Corrupted paths array should be empty"

    # Corrupt file by truncating it
    with open(filepath, "r+") as f:
        file_size = os.path.getsize(filepath)
        f.truncate(round(file_size / 2))

    # Test that all paths are "corrupted"
    res = di.get_corrupted_filepaths([filepath])
    assert len(res) == 1, "Corrupted paths array should have one path"


def test_remove_corrupted_filepaths(
    local_filepaths: List[Tuple[str, ...]],
    tmpdir: str,
) -> None:
    """Test remove_corrupted_filepaths function

    Create a fake file, delete it then validate
    """

    abs_paths = [os.path.join(tmpdir, *filepath) for filepath in local_filepaths]

    # Create a fake file
    for filepath in abs_paths:
        # Create folder
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write("Hello World")

        assert os.path.exists(filepath), "Temporary file was not created"

    di.remove_corrupted_filepaths(abs_paths)

    for filepath in abs_paths:
        assert not os.path.exists(filepath), "Temporary file was not removed"


def test_check_filepaths_integrity(
    local_filepaths: List[Tuple[str, ...]],
    tmpdir: str,
) -> None:
    """Test remove_corrupted_filepaths function

    Create a fake file, delete it then validate
    """

    abs_paths = [os.path.join(tmpdir, *filepath) for filepath in local_filepaths]

    # Create a fake file
    for filepath in abs_paths:
        # Create folder
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write("Hello World")

        assert os.path.exists(filepath), "Temporary file was not created"

    # Test function without removing files
    di.check_filepaths_integrity(abs_paths, remove_corrupted=False)

    for filepath in abs_paths:
        assert os.path.exists(filepath), "Temporary file was removed when it should not have been"

    # Test function with removing files
    di.check_filepaths_integrity(abs_paths, remove_corrupted=True)

    for filepath in abs_paths:
        assert not os.path.exists(
            filepath
        ), "Temporary file was not removed when it should not have been"
