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
"""This module test that GPM-API Dataset structure, attributes and values does not mute across software versions."""
import glob
import os

import pytest
import xarray as xr

import gpm
from gpm import _root_path
from gpm.dataset.granule import open_granule_dataset

PRODUCT_TYPES = ["RS"]

GRANULES_DIR_PATH = os.path.join(_root_path, "gpm", "tests", "data", "granules")

ORBIT_EXAMPLE_FILEPATH = glob.glob(os.path.join(GRANULES_DIR_PATH, "cut", "RS", "V7", "2A-DPR", "*"))[0]
GRID_EXAMPLE_FILEPATH = glob.glob(os.path.join(GRANULES_DIR_PATH, "cut", "RS", "V7", "IMERG-FR", "*"))[0]

gpm.config.set(
    {
        "warn_non_contiguous_scans": False,
        "warn_non_regular_timesteps": False,
        "warn_invalid_geolocation": False,
    },
)


def remove_unfixed_attributes(ds):
    # Remove history attribute
    ds.attrs.pop("history")

    # Remove attributes conflicting between python versions
    if "crsWGS84" in ds.coords:
        ds.coords["crsWGS84"].attrs.pop("crs_wkt")
        ds.coords["crsWGS84"].attrs.pop("horizontal_datum_name")
        ds.coords["crsWGS84"].attrs.pop("spatial_ref")
    return ds


def check_dataset_equality(cut_filepath):
    processed_dir = os.path.dirname(cut_filepath.replace("cut", "processed"))
    processed_filenames = os.listdir(processed_dir)
    processed_filepaths = [os.path.join(processed_dir, filename) for filename in processed_filenames]
    scan_modes = [os.path.splitext(filename)[0] for filename in processed_filenames]
    for scan_mode, processed_filepath in zip(scan_modes, processed_filepaths, strict=False):
        # Open files
        ds = open_granule_dataset(cut_filepath, scan_mode=scan_mode).compute()
        ds_expected = xr.open_dataset(processed_filepath).compute()

        # Close connection to file
        ds.close()
        ds_expected.close()

        # Remove attributes that are allowed to change
        ds_expected = remove_unfixed_attributes(ds_expected)
        ds = remove_unfixed_attributes(ds)

        # Check equality
        # assert ds.equals(ds_expected)
        xr.testing.assert_allclose(ds, ds_expected)


def test_open_granule_on_real_files():
    """Test open_granule_dataset on real files.

    Load cut granules and check that the new file is identical to the saved reference.

    Run `python gpm/tests/data/scripts/generate_test_granule_data.py` or
    "update_processed_test_data" to regenerate or update the test granules.
    The expected granules directory structure is:

    tests/data/granules
    ├── cut
    │   ├── V7/RS/1A-GMI
    │   │   └── 1A.GPM.GMI.COUNT2021.20140304-S223658-E000925.000082.V07A.HDF5
    │   └── ...
    └── processed
        ├── V7/RS/1A-GMI
        │    ├── S1.nc
        │    ├── S2.nc
        │    ├── S4.nc
        │    └── S5.nc
        └── ...
    """
    if not os.path.exists(GRANULES_DIR_PATH):
        pytest.skip(
            "Test granules not found. Please run `git submodule update --init` to clone "
            "existing test data, or `python generate_test_granule_data.py` to generate new test data.",
        )

    cut_dir_path = os.path.join(GRANULES_DIR_PATH, "cut")

    for product_type in PRODUCT_TYPES:
        if product_type == "RS":
            cut_filepaths = glob.glob(os.path.join(cut_dir_path, "RS", "*", "*", "*"))
        else:
            cut_filepaths = glob.glob(os.path.join(cut_dir_path, "NRT", "*", "*"))

        if len(cut_filepaths) == 0:
            raise ValueError("No test data found.")
        # Execute checks over all datasets
        list_failed_checks = []
        for cut_filepath in cut_filepaths:
            try:
                check_dataset_equality(cut_filepath)
            except Exception as e:
                version, product = cut_filepath.split(os.sep)[-3:-1]
                list_failed_checks.append((f"{version} of {product}", str(e)))

        # Report which tests failed
        if len(list_failed_checks) > 0:
            failed_products = [product_id for (product_id, err) in list_failed_checks]
            msg = f"Failed dataset comparison for {failed_products}. Errors are: {list_failed_checks}"
            raise ValueError(msg)


class TestOpenGranuleMethods:

    @pytest.mark.parametrize("filepath", [ORBIT_EXAMPLE_FILEPATH, GRID_EXAMPLE_FILEPATH])
    def test_open_granule_dataset(self, filepath):
        """Test open granule with open_granule_dataset."""
        ds = gpm.open_granule_dataset(filepath, cache=False, lock=False, decode_cf=True)
        assert isinstance(ds, xr.Dataset)
        ds.close()

    @pytest.mark.parametrize("filepath", [ORBIT_EXAMPLE_FILEPATH, GRID_EXAMPLE_FILEPATH])
    def test_open_raw_datatree(self, filepath):
        """Test open granule with open_raw_datatree."""
        dt = gpm.open_raw_datatree(filepath, cache=False, lock=False, decode_cf=True)
        assert isinstance(dt, xr.DataTree)
        dt.close()

    @pytest.mark.parametrize("filepath", [ORBIT_EXAMPLE_FILEPATH])
    def test_open_granule_datatree(self, filepath):
        """Test open granule with open_granule_datatree."""
        dt = gpm.open_granule_datatree(filepath, cache=False, lock=False, decode_cf=True)
        assert isinstance(dt, xr.DataTree)
        dt.close()
