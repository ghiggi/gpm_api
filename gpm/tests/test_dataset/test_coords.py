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
"""This module test the GPM-API Dataset coordinates."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from deepdiff import DeepDiff

from gpm.dataset import coords

MAX_TIMESTAMP = 2**31 - 1


# Utils functions ##############################################################


def get_random_datetime_array_and_dataset(n_values):
    """Return random datetimes as numpy array and xarrray dataset."""
    rng = np.random.default_rng()
    timestamps = rng.integers(0, MAX_TIMESTAMP, size=n_values)
    datetimes = pd.to_datetime(timestamps, unit="s")
    ds = xr.Dataset(
        {
            "Year": ("along_track", datetimes.year),
            "Month": ("along_track", datetimes.month),
            "DayOfMonth": ("along_track", datetimes.day),
            "Hour": ("along_track", datetimes.hour),
            "Minute": ("along_track", datetimes.minute),
            "Second": ("along_track", datetimes.second),
        },
    )
    return datetimes.to_numpy(), ds


# Tests for public functions ###################################################


def test_get_orbit_coords():
    """Test get_orbit_coords."""
    scan_mode = "S1"
    shape = (10, 3)
    rng = np.random.default_rng()
    granule_id = rng.integers(0, 100000)

    # Create random datatree
    rng = np.random.default_rng()
    lon = xr.DataArray(rng.random(shape), dims=["along_track", "cross_track"])
    lat = xr.DataArray(rng.random(shape), dims=["along_track", "cross_track"])
    time_array, time_ds = get_random_datetime_array_and_dataset(shape[0])

    dt = xr.DataTree.from_dict({scan_mode: xr.DataTree.from_dict({"ScanTime": time_ds})})
    dt[scan_mode]["Longitude"] = lon
    dt[scan_mode]["Latitude"] = lat
    dt.attrs["FileHeader"] = f"GranuleNumber={granule_id}"

    # Test get_orbit_coords arrays values
    expected_coords = {
        "lon": (["along_track", "cross_track"], lon.data),
        "lat": (["along_track", "cross_track"], lat.data),
        "time": (["along_track"], time_array),
        "gpm_id": (["along_track"], np.array([f"{granule_id}-{i}" for i in range(shape[0])])),
        "gpm_granule_id": (["along_track"], np.repeat(granule_id, shape[0])),
        "gpm_cross_track_id": (["cross_track"], np.arange(shape[1])),
        "gpm_along_track_id": (["along_track"], np.arange(shape[0])),
    }
    returned_coords = coords.get_orbit_coords(dt, scan_mode)
    returned_coords = {k: (list(da.dims), da.data) for k, da in returned_coords.items()}

    # Ensure same integer for gpm_granule_id
    expected_coords["gpm_granule_id"] = (
        expected_coords["gpm_granule_id"][0],
        expected_coords["gpm_granule_id"][1].astype(int),
    )
    returned_coords["gpm_granule_id"] = (
        returned_coords["gpm_granule_id"][0],
        returned_coords["gpm_granule_id"][1].astype(int),
    )
    # Compare same type
    diff = DeepDiff(expected_coords, returned_coords)
    assert diff == {}, f"Dictionaries are not equal: {diff}"


def test_get_grid_coords():
    """Test get_grid_coords."""
    scan_mode = "Grid"
    n_values = 10

    # Create random datatree
    rng = np.random.default_rng()
    lon = rng.random(n_values)
    lat = rng.random(n_values)
    timestamp = rng.integers(0, MAX_TIMESTAMP)
    time_formated = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    ds = xr.Dataset()
    ds.coords["lon"] = ("lon", lon)
    ds.coords["lat"] = ("lat", lat)

    dt = xr.DataTree.from_dict({scan_mode: ds})
    dt.attrs["FileHeader"] = f"StartGranuleDateTime={time_formated};\nTimeInterval=HALF_HOUR;"

    # Test get_grid_coords
    start_time = datetime.fromtimestamp(timestamp)
    end_time = datetime.fromtimestamp(timestamp) + timedelta(minutes=30)
    time_bnds = [[start_time, end_time]]
    expected_coords = {
        "time": np.array([end_time]).astype("M8[ns]"),
        "lon": lon,
        "lat": lat,
        "time_bnds": np.array(time_bnds).astype("M8[ns]"),
    }
    returned_coords = coords.get_grid_coords(dt, scan_mode)
    returned_coords = {k: da.data for k, da in returned_coords.items()}
    diff = DeepDiff(expected_coords, returned_coords)
    assert diff == {}, f"Dictionaries are not equal: {diff}"


def test_get_coords(monkeypatch):
    """Test get_coords."""
    # Mock get_orbit_coords and get_grid_coords
    monkeypatch.setattr(coords, "get_orbit_coords", lambda *args: "return from get_orbit_coords")
    monkeypatch.setattr(coords, "get_grid_coords", lambda *args: "return from get_grid_coords")

    # Test get_coords
    scan_mode = "S1"
    dt = xr.DataTree()
    returned_coords = coords.get_coords(dt, scan_mode)
    assert returned_coords == "return from get_orbit_coords"

    scan_mode = "Grid"
    dt = xr.DataTree()
    returned_coords = coords.get_coords(dt, scan_mode)
    assert returned_coords == "return from get_grid_coords"


def test_get_coords_attrs_dict() -> None:
    """Test get_coords_attrs_dict."""
    # Test dataset with no matching attributes
    ds = xr.Dataset()
    returned_dict = coords.get_coords_attrs_dict(ds)
    assert returned_dict == {}

    # Test with one matching attribute in coords
    ds = xr.Dataset()
    ds.coords["lat"] = []
    returned_dict = coords.get_coords_attrs_dict(ds)
    assert "lat" in returned_dict
    assert returned_dict["lat"] != {}

    # Test with one matching attribute in data_vars
    ds = xr.Dataset()
    ds["lat"] = xr.DataArray([])
    returned_dict = coords.get_coords_attrs_dict(ds)
    assert "lat" in returned_dict
    assert returned_dict["lat"] != {}

    # Test with unrelevant attributes
    ds = xr.Dataset()
    ds.coords["unrelevant_1"] = []
    ds["unrelevant_2"] = xr.DataArray([])
    returned_dict = coords.get_coords_attrs_dict(ds)
    assert returned_dict == {}

    # Test all attributes
    relevant_attributes = [
        "lat",
        "lon",
        "gpm_granule_id",
        "time",
        "gpm_cross_track_id",
        "gpm_along_track_id",
        "gpm_id",
    ]
    ds = xr.Dataset()
    for attribute in relevant_attributes:
        ds.coords[attribute] = []

    returned_dict = coords.get_coords_attrs_dict(ds)
    for attribute in relevant_attributes:
        assert attribute in returned_dict
        assert returned_dict[attribute] != {}


def test_set_coords_attrs() -> None:
    """Test set_coords_attrs."""
    # Test dataset with one relevant attribute and one unrelevant attribute
    ds = xr.Dataset()
    ds["lat"] = xr.DataArray([])
    ds["unrelevant"] = xr.DataArray([])

    returned_ds = coords.set_coords_attrs(ds)
    assert returned_ds["lat"].attrs != {}
    assert returned_ds["unrelevant"].attrs == {}
