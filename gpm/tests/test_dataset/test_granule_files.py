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
import os
import xarray as xr
import pytest
from gpm.dataset.granule import open_granule
from gpm import _root_path
import glob
import gpm


PRODUCT_TYPES = ["RS"]


gpm.config.set(
    {
        "warn_non_contiguous_scans": False,
        "warn_non_regular_timesteps": False,
        "warn_invalid_geolocation": False,
    }
)


def test_open_granule_on_real_files():
    """Test open_granule on real files.

    Load cut granules and check that the new file is identical to the saved reference.

    Run `python gpm/tests/test_dataset/generate_test_granule_data.py` to generate the test granules.
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

    granules_dir_path = os.path.join(_root_path, "gpm", "tests", "data", "granules")

    if not os.path.exists(granules_dir_path):
        pytest.skip(
            "Test granules not found. Please run `git submodule update --init` to clone existing test data, or `python generate_test_granule_data.py` to generate new test data."
        )

    cut_dir_path = os.path.join(granules_dir_path, "cut")

    for product_type in PRODUCT_TYPES:
        if product_type == "RS":
            cut_filepaths = glob.glob(os.path.join(cut_dir_path, "RS", "*", "*", "*"))
        else:
            cut_filepaths = glob.glob(os.path.join(cut_dir_path, "NRT", "*", "*"))

        if len(cut_filepaths) == 0:
            raise ValueError("No test data found.")

        for cut_filepath in cut_filepaths:
            print(cut_filepath)
            processed_dir = os.path.dirname(cut_filepath.replace("cut", "processed"))
            processed_filenames = os.listdir(processed_dir)
            processed_filepaths = [
                os.path.join(processed_dir, filename) for filename in processed_filenames
            ]
            scan_modes = [os.path.splitext(filename)[0] for filename in processed_filenames]
            for scan_mode, processed_filepath in zip(scan_modes, processed_filepaths):
                ds = open_granule(cut_filepath, scan_mode=scan_mode).compute()
                ds_expected = xr.open_dataset(processed_filepath).compute()

                for _ds in [ds, ds_expected]:
                    # Remove history attribute
                    _ds.attrs.pop("history")

                    # Remove attributes conflicting between python versions
                    if "crsWGS84" in _ds.coords:
                        _ds.coords["crsWGS84"].attrs.pop("crs_wkt")
                        _ds.coords["crsWGS84"].attrs.pop("horizontal_datum_name")
                        _ds.coords["crsWGS84"].attrs.pop("spatial_ref")

                # Check equality
                xr.testing.assert_identical(ds, ds_expected)
