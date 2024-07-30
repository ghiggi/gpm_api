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
"""This module tests the dataframe utilities functions."""
import pyproj

from gpm.dataset.crs import set_dataset_crs
from gpm.tests.utils.fake_datasets import get_orbit_dataarray
from gpm.utils.dataframe import to_dask_dataframe, to_pandas_dataframe


def test_to_pandas_dataframe():
    # Create dataset
    da = get_orbit_dataarray(
        start_lon=0,
        start_lat=0,
        end_lon=10,
        end_lat=20,
        width=1e6,
        n_along_track=10,
        n_cross_track=5,
        n_range=2,
    )

    ds = da.to_dataset(name="dummy_var")
    # Add crsWGS84 coordinate (to be discarded by to_pandas_dataframe)
    crs = pyproj.CRS(proj="longlat", ellps="WGS84")
    ds = set_dataset_crs(ds, crs=crs, grid_mapping_name="crsWGS84", inplace=False)
    # Convert to dask
    ds = ds.chunk("auto")
    # Write to pandas
    df = to_pandas_dataframe(ds)

    # Check results
    # - cross-track, along_track, crsWGS84 are not in the columns !
    expected_columns = [
        "range",
        "lat",
        "lon",
        "gpm_granule_id",
        "gpm_cross_track_id",
        "gpm_along_track_id",
        "gpm_id",
        "time",
        "height",
        "dummy_var",
    ]
    assert list(df.columns) == expected_columns
    assert df.shape == (100, 10)
    assert df["gpm_id"].dtype.name == "string"


def test_to_dask_dataframe():
    # Create dataset
    da = get_orbit_dataarray(
        start_lon=0,
        start_lat=0,
        end_lon=10,
        end_lat=20,
        width=1e6,
        n_along_track=10,
        n_cross_track=5,
        n_range=2,
    )
    ds = da.to_dataset(name="dummy_var")
    # Add crsWGS84 coordinate (to be discarded by to_dask_dataframe)
    crs = pyproj.CRS(proj="longlat", ellps="WGS84")
    ds = set_dataset_crs(ds, crs=crs, grid_mapping_name="crsWGS84", inplace=False)

    # Create a dataset with nonuniform chunking
    # - And test that unify chunks before conversion to dask dataframe
    ds = ds.chunk("auto")
    ds["dummy_var"] = ds["dummy_var"].chunk({"range": 1, "cross_track": 2, "along_track": 2})

    # Convert to dask dataframe
    df = to_dask_dataframe(ds)

    # Check results
    # - cross-track, along_track, crsWGS84 are not in the columns !
    expected_columns = [
        "range",
        "lat",
        "lon",
        "gpm_granule_id",
        "gpm_cross_track_id",
        "gpm_along_track_id",
        "gpm_id",
        "time",
        "height",
        "dummy_var",
    ]
    assert list(df.columns) == expected_columns
    assert df["gpm_id"].dtype.name == "string"

    assert df.compute().shape == (100, 10)
