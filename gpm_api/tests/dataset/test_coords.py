from datetime import datetime, timedelta
from datatree import DataTree
import numpy as np
import pandas as pd
import pytest
from pytest import SaneEqualityArray
import xarray as xr
from gpm_api.dataset import coords


MAX_TIMESTAMP = 2**31 - 1


# Utils functions ##############################################################


def get_random_datetime_array_and_dataset(n_values):
    """Return random datetimes as numpy array and xarrray dataset."""

    timestamps = np.random.randint(0, MAX_TIMESTAMP, size=n_values)
    datetimes = pd.to_datetime(timestamps, unit="s")
    ds = xr.Dataset(
        {
            "Year": ("along_track", datetimes.year),
            "Month": ("along_track", datetimes.month),
            "DayOfMonth": ("along_track", datetimes.day),
            "Hour": ("along_track", datetimes.hour),
            "Minute": ("along_track", datetimes.minute),
            "Second": ("along_track", datetimes.second),
        }
    )
    return datetimes.to_numpy(), ds


# Tests for public functions ###################################################


def test_get_orbit_coords():
    """Test get_orbit_coords"""

    scan_mode = "S1"
    granule_id = np.random.randint(0, 100000)
    shape = (10, 3)

    # Create random datatree
    lon = xr.DataArray(np.random.rand(*shape), dims=["along_track", "cross_track"])
    lat = xr.DataArray(np.random.rand(*shape), dims=["along_track", "cross_track"])
    time_array, time_ds = get_random_datetime_array_and_dataset(shape[0])

    dt = DataTree.from_dict({scan_mode: DataTree.from_dict({"ScanTime": time_ds})})
    dt[scan_mode]["Longitude"] = lon
    dt[scan_mode]["Latitude"] = lat
    dt.attrs["FileHeader"] = f"GranuleNumber={granule_id}"

    # Test get_orbit_coords
    expected_coords = {
        "lon": (["along_track", "cross_track"], lon),
        "lat": (["along_track", "cross_track"], lat),
        "time": (["along_track"], time_array),
        "gpm_id": (["along_track"], [f"{granule_id}-{i}" for i in range(shape[0])]),
        "gpm_granule_id": (["along_track"], np.repeat(granule_id, shape[0])),
        "gpm_cross_track_id": (["cross_track"], np.arange(shape[1])),
        "gpm_along_track_id": (["along_track"], np.arange(shape[0])),
    }
    expected_coords = {k: (v[0], SaneEqualityArray(v[1])) for k, v in expected_coords.items()}
    returned_coords = coords.get_orbit_coords(dt, scan_mode)
    assert returned_coords == expected_coords


def test_get_grid_coords():
    """Test get_grid_coords"""

    scan_mode = "Grid"
    n_values = 10

    # Create random datatree
    # time = np.random.randint(0, MAX_TIMESTAMP)
    lon = np.random.rand(n_values)
    lat = np.random.rand(n_values)
    timestamp = np.random.randint(0, MAX_TIMESTAMP)
    time_formated = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    ds = xr.Dataset()
    # ds.coords["time"] = ("time", [time])
    ds.coords["lon"] = ("lon", lon)
    ds.coords["lat"] = ("lat", lat)

    dt = DataTree.from_dict({scan_mode: ds})
    dt.attrs["FileHeader"] = f"StartGranuleDateTime={time_formated}"

    # Test get_grid_coords
    corrected_datetime = datetime.fromtimestamp(timestamp) + timedelta(minutes=30)
    expected_coords = {
        "time": np.array([corrected_datetime]),
        "lon": lon,
        "lat": lat,
    }
    expected_coords = {k: SaneEqualityArray(v) for k, v in expected_coords.items()}
    returned_coords = coords.get_grid_coords(dt, scan_mode)
    assert returned_coords == expected_coords


def test_get_coords(monkeypatch):
    """Test get_coords"""

    # Mock get_orbit_coords and get_grid_coords
    monkeypatch.setattr(coords, "get_orbit_coords", lambda *args: "return from get_orbit_coords")
    monkeypatch.setattr(coords, "get_grid_coords", lambda *args: "return from get_grid_coords")

    # Test get_coords
    scan_mode = "S1"
    dt = DataTree()
    returned_coords = coords.get_coords(dt, scan_mode)
    assert returned_coords == "return from get_orbit_coords"

    scan_mode = "Grid"
    dt = DataTree()
    returned_coords = coords.get_coords(dt, scan_mode)
    assert returned_coords == "return from get_grid_coords"


def test_get_coords_attrs_dict() -> None:
    """Test get_coords_attrs_dict"""

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
    """Test set_coords_attrs"""

    # Test dataset with one relevant attribute and one unrelevant attribute
    ds = xr.Dataset()
    ds["lat"] = xr.DataArray([])
    ds["unrelevant"] = xr.DataArray([])

    returned_ds = coords.set_coords_attrs(ds)
    assert returned_ds["lat"].attrs != {}
    assert returned_ds["unrelevant"].attrs == {}
