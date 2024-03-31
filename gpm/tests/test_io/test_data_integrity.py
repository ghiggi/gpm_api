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
"""This module test the data integrity checks."""
import datetime
import os

import pytest
import xarray as xr
from pytest_mock.plugin import MockerFixture

from gpm.io import data_integrity as di


def test_get_corrupted_filepaths(
    local_filepaths_unix: list[str],
    local_filepaths_windows: list[str],
) -> None:
    """Test get_corrupted_filepaths function."""
    # Test that all paths are "corrupted" (in this case there is no data)
    for abs_paths in [local_filepaths_unix, local_filepaths_windows]:
        res = di.get_corrupted_filepaths(abs_paths)

        assert len(abs_paths) == len(
            res,
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


def _write_dummy_file(filepath):
    """Write a dummy file to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("Hello World")


def test_remove_corrupted_filepaths(
    local_filepaths: list[tuple[str, ...]],
    tmpdir: str,
) -> None:
    """Test remove_corrupted_filepaths function.

    Create a fake file, delete it then validate
    """
    abs_paths = [os.path.join(tmpdir, *filepath) for filepath in local_filepaths]

    # Create a fake file
    for filepath in abs_paths:
        _write_dummy_file(filepath)
        assert os.path.exists(filepath), "Temporary file was not created"

    di.remove_corrupted_filepaths(abs_paths)

    for filepath in abs_paths:
        assert not os.path.exists(filepath), "Temporary file was not removed"


def test_check_filepaths_integrity(
    local_filepaths: list[tuple[str, ...]],
    tmpdir: str,
) -> None:
    """Test remove_corrupted_filepaths function.

    Create a fake file, delete it then validate
    """
    abs_paths = [os.path.join(tmpdir, *filepath) for filepath in local_filepaths]

    # Create a fake file
    for filepath in abs_paths:
        _write_dummy_file(filepath)
        assert os.path.exists(filepath), "Temporary file was not created"

    # Test function without removing files
    di.check_filepaths_integrity(abs_paths, remove_corrupted=False)
    for filepath in abs_paths:
        assert os.path.exists(filepath), "Temporary file was removed when it should not have been"

    # Test function with removing files
    di.check_filepaths_integrity(abs_paths, remove_corrupted=True)
    for filepath in abs_paths:
        assert not os.path.exists(
            filepath,
        ), "Temporary file was not removed when it should not have been"


@pytest.mark.parametrize("remove_corrupted", [True, False])
@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("filepaths", [[], ["some_corrupted_filepaths"]])
def test_check_archive_integrity(
    tmpdir: str,
    mocker: MockerFixture,
    filepaths,
    remove_corrupted,
    verbose,
) -> None:
    """Test check_archive_integrity function."""
    product = "2A-DPR"
    product_type = "RS"
    version = 7
    start_time = datetime.datetime(2022, 9, 7, 12, 0, 0)
    end_time = datetime.datetime(2022, 9, 7, 13, 0, 0)

    # Create fake filepaths
    filepaths = [os.path.join(tmpdir, filepath) for filepath in filepaths]

    # Create fake files
    for filepath in filepaths:
        _write_dummy_file(filepath)
        assert os.path.exists(filepath), "Temporary file was not created"

    # Mock find_filepaths
    mocker.patch.object(di, "find_filepaths", autospec=True, return_value=filepaths)

    # If no filepaths, test raise value error
    if len(filepaths) == 0:
        with pytest.raises(ValueError):
            di.check_archive_integrity(
                version=version,
                product=product,
                product_type=product_type,
                start_time=start_time,
                end_time=end_time,
                remove_corrupted=remove_corrupted,
                verbose=verbose,
            )
    # Else test return list of corrupted filepaths
    else:
        l_corrupted = di.check_archive_integrity(
            version=version,
            product=product,
            product_type=product_type,
            start_time=start_time,
            end_time=end_time,
            remove_corrupted=remove_corrupted,
            verbose=verbose,
        )
        assert l_corrupted == filepaths
