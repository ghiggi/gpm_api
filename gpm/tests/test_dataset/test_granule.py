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
"""This module test the creation of a GPM-API Dataset."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gpm.dataset import conventions, datatree, granule
from gpm.dataset.conventions import finalize_dataset
from gpm.utils.time import ensure_time_validity

# Tests for public functions ###################################################


def test_get_variables():
    """Test get_variables."""
    da = xr.DataArray()
    dataset = xr.Dataset(data_vars={"var_1": da, "var_2": da})
    expected_variables = ["var_1", "var_2"]
    returned_variables = granule.get_variables(dataset)
    assert returned_variables == expected_variables


def test_get_variables_dims():
    """Test get_variables_dims."""
    array_1 = np.zeros(shape=(3, 3))
    array_2 = np.zeros(shape=(3, 3))
    dataarray_1 = xr.DataArray(array_1, dims=["dim_1", "dim_2"])
    dataarray_2 = xr.DataArray(array_2, dims=["dim_2", "dim_3"])
    dataset = xr.Dataset(data_vars={"var_1": dataarray_1, "var_2": dataarray_2})

    expected_variables_dims = ["dim_1", "dim_2", "dim_3"]
    returned_variables_dims = granule.get_variables_dims(dataset)
    assert np.array_equal(returned_variables_dims, expected_variables_dims)

    # Check dataset with no variables
    dataset = xr.Dataset()
    expected_variables_dims = []
    returned_variables_dims = granule.get_variables_dims(dataset)
    assert returned_variables_dims == expected_variables_dims


def test_unused_var_dims_and_remove():
    """Test unused_var_dims and remove_unused_var_dims."""
    array = np.zeros(shape=(3,))
    dataarray = xr.DataArray(array, dims=["used_dim"])
    dataset = xr.Dataset(data_vars={"var": dataarray})
    dataset.coords["coord"] = xr.DataArray(array, dims=["unused_dim"])
    # Note: using dataset.expand_dims does not work, as it adds the dimension to all variables.

    # Check list of unused dimensions
    expected_unused_dims = ["unused_dim"]
    returned_unused_dims = granule.unused_var_dims(dataset)
    assert returned_unused_dims == expected_unused_dims

    # Remove unused dimensions
    returned_dataset = granule.remove_unused_var_dims(dataset)
    expected_dims = ["used_dim"]
    assert list(returned_dataset.dims) == expected_dims


def test_open_granule_dataset(monkeypatch):
    """Test open_granule_dataset."""
    filepath = "RS/V07/RADAR/2A-DPR/2022/07/06/2A.GPM.DPR.V9-20211125.20220706-S043937-E061210.047456.V07A.HDF5"
    scan_mode = "FS"

    ds = xr.Dataset()
    dt = xr.DataTree.from_dict({scan_mode: ds})

    # Mock units tested elsewhere
    monkeypatch.setattr(
        granule,
        "get_granule_attrs",
        lambda *args, **kwargs: {"attribute": "attribute_value"},
    )
    monkeypatch.setattr(granule, "get_coords", lambda *args, **kwargs: {"coord": "coord_value"})
    monkeypatch.setattr(
        granule,
        "_get_relevant_groups_variables",
        lambda *args, **kwargs: ([""], []),
    )

    def patch_finalize_dataset(ds, *args, **kwargs):
        ds.attrs["finalized"] = True
        return ds

    monkeypatch.setattr(granule, "finalize_dataset", patch_finalize_dataset)

    # Mock datatree opening from filepath
    monkeypatch.setattr(datatree, "open_raw_datatree", lambda *args, **kwargs: dt)

    returned_dataset = granule.open_granule_dataset(filepath)
    expected_attribute_keys = ["attribute", "ScanMode", "finalized"]
    expected_coordinate_keys = ["coord"]
    assert isinstance(returned_dataset, xr.Dataset)
    assert list(returned_dataset.attrs) == expected_attribute_keys
    assert list(returned_dataset.coords) == expected_coordinate_keys


# Tests for internal functions #################################################


def test_prefix_dataset_group_variables():
    """Test _prefix_dataset_group_variables."""
    da = xr.DataArray()
    dataset = xr.Dataset(data_vars={"var_1": da, "var_2": da})
    group = "group_1"

    expected_data_vars = ["group_1/var_1", "group_1/var_2"]
    returned_dataset = granule._prefix_dataset_group_variables(dataset, group)
    assert isinstance(returned_dataset, xr.Dataset)
    assert list(returned_dataset.data_vars) == expected_data_vars


def test_remove_dummy_variables():
    """Test _remove_dummy_variables."""
    da = xr.DataArray()
    dataset = xr.Dataset(
        data_vars={
            # Dummy variables
            "Latitude": da,
            "Longitude": da,
            "time_bnds": da,
            "lat_bnds": da,
            "lon_bnds": da,
            # Real variables
            "real_var_1": da,
        },
    )

    expected_data_vars = ["lat_bnds", "lon_bnds", "real_var_1"]
    returned_dataset = granule._remove_dummy_variables(dataset)
    assert list(returned_dataset.data_vars) == expected_data_vars


def test_subset_dataset_variables():
    """Test _subset_dataset_variables."""
    da = xr.DataArray()
    dataset = xr.Dataset(data_vars={"var_1": da, "var_2": da})

    # Subset variables
    variables = ["var_1"]
    returned_dataset = granule._subset_dataset_variables(dataset, variables)
    assert isinstance(returned_dataset, xr.Dataset)
    assert list(returned_dataset.data_vars) == variables

    # Variables not in dataset
    variables = ["var_1", "var_3"]
    expected_data_vars = ["var_1"]
    returned_dataset = granule._subset_dataset_variables(dataset, variables)
    assert list(returned_dataset.data_vars) == expected_data_vars

    # No variables
    variables = []
    returned_dataset = granule._subset_dataset_variables(dataset, variables)
    assert list(returned_dataset.data_vars) == variables

    # With variables None: return all variables
    expected_data_vars = ["var_1", "var_2"]
    returned_dataset = granule._subset_dataset_variables(dataset, None)
    assert list(returned_dataset.data_vars) == expected_data_vars


def test_process_group_dataset():
    """Test _process_group_dataset."""
    da = xr.DataArray()
    dataset = xr.Dataset(
        data_vars={
            # Dummy variable
            "Latitude": da,
            # Kept variable
            "var_1": da,
            # Removed variable
            "var_2": da,
        },
    )
    variables = ["Latitude", "var_1"]
    group = "group_1"

    expected_data_vars = ["var_1"]
    returned_dataset = granule._process_group_dataset(dataset, group, variables)
    assert isinstance(returned_dataset, xr.Dataset)
    assert list(returned_dataset.data_vars) == expected_data_vars

    # Prefix group variables
    expected_data_vars = ["group_1/var_1"]
    returned_dataset = granule._process_group_dataset(dataset, group, variables, prefix_group=True)
    assert list(returned_dataset.data_vars) == expected_data_vars


def test_get_flattened_scan_mode_dataset():
    """Test _get_flattened_scan_mode_dataset."""
    da = xr.DataArray()

    # Build source datatree
    scan_mode = "scan_mode"
    dt = xr.DataTree.from_dict(
        {
            scan_mode: xr.DataTree.from_dict(
                {
                    "group_1": xr.DataTree(),
                    "group_2": xr.DataTree(),
                },
            ),
        },
    )
    dt[scan_mode]["group_1"]["var_1"] = da
    dt[scan_mode]["group_2"]["var_2"] = da
    dt[scan_mode]["var_3"] = da

    # Without specifying group: empty dataset
    group = []
    expected_data_vars = []
    returned_dataset = granule._get_flattened_scan_mode_dataset(dt, scan_mode, group)
    assert isinstance(returned_dataset, xr.Dataset)
    assert list(returned_dataset.data_vars) == expected_data_vars

    # Group same as scan_mode: return only top-level variables of scan_mode
    group = [scan_mode]
    expected_data_vars = ["var_3"]
    returned_dataset = granule._get_flattened_scan_mode_dataset(dt, scan_mode, group)
    assert list(returned_dataset.data_vars) == expected_data_vars

    # Specifying sub-groups: return variables of sub-groups
    group = ["group_1", "group_2"]
    expected_data_vars = ["var_1", "var_2"]
    returned_dataset = granule._get_flattened_scan_mode_dataset(dt, scan_mode, group)
    assert list(returned_dataset.data_vars) == expected_data_vars

    # Check variables filtering and group prefixing
    group = ["group_1", "group_2"]
    variables = ["var_1"]
    expected_data_vars = ["group_1/var_1"]
    returned_dataset = granule._get_flattened_scan_mode_dataset(
        dt,
        scan_mode,
        group,
        variables=variables,
        prefix_group=True,
    )
    assert list(returned_dataset.data_vars) == expected_data_vars


def test_ensure_time_validity():
    """Test ensure_time_validity."""

    def construct_dataset_and_check_validation(input_datetimes, expected_datetimes):
        da = xr.DataArray(input_datetimes)
        ds = xr.Dataset({"time": da})
        returned_dataset = ensure_time_validity(ds)
        assert np.array_equal(
            returned_dataset["time"].to_numpy(),
            np.array(expected_datetimes, dtype="datetime64[ns]"),
        )

    # Check with a single timestep
    timestamps = [0]
    datetimes = [datetime.fromtimestamp(t) for t in timestamps]
    construct_dataset_and_check_validation(datetimes, datetimes)

    # Check multiple timesteps without NaT (Not a Time)
    timestamps = [0, 1, 2]
    datetimes = [datetime.fromtimestamp(t) for t in timestamps]
    construct_dataset_and_check_validation(datetimes, datetimes)

    # Check multiple timesteps with NaT
    timestamps = np.linspace(0, 20, 21)
    datetimes = [datetime.fromtimestamp(t) for t in timestamps]
    missing_slice = slice(5, 10)
    datetimes_incomplete = datetimes.copy()
    datetimes_incomplete[missing_slice] = [pd.NaT] * len(datetimes[missing_slice])
    construct_dataset_and_check_validation(datetimes_incomplete, datetimes)

    # Check with multiple holes
    datetimes_incomplete = datetimes.copy()
    for missing_slice in [slice(2, 7), slice(10, 15)]:
        datetimes_incomplete[missing_slice] = [pd.NaT] * len(datetimes[missing_slice])
    construct_dataset_and_check_validation(datetimes_incomplete, datetimes)

    # Check with more than 10 consecutive missing timesteps
    datetimes_incomplete = datetimes.copy()
    missing_slice = slice(5, 16)
    datetimes_incomplete[missing_slice] = [pd.NaT] * len(datetimes[missing_slice])
    with pytest.raises(ValueError):
        construct_dataset_and_check_validation(datetimes_incomplete, datetimes)


def get_sample_orbit_dataset():
    # Return a default dataset
    rng = np.random.default_rng()
    da = xr.DataArray(rng.random((1, 2, 2)), dims=("other", "along_track", "cross_track"))
    lon = xr.DataArray(rng.random((2, 2)), dims=("cross_track", "along_track"))
    lat = xr.DataArray(rng.random((2, 2)), dims=("cross_track", "along_track"))
    start_time = datetime(2018, 1, 1, 12, 30, 0)
    end_time = datetime(2018, 1, 1, 12, 32, 0)
    time = [start_time, end_time]
    ds = xr.Dataset({"var": da, "time": time, "lon": lon, "lat": lat})
    return ds.set_coords(["lon", "lat"])


def get_sample_grid_dataset():
    # Return a default dataset
    rng = np.random.default_rng()
    da = xr.DataArray(rng.random((1, 1, 1, 1)), dims=("lat", "lon", "time", "other"))
    lon = xr.DataArray([1], dims=("lon"))
    lat = xr.DataArray([2], dims=("lat"))
    time = [0]
    ds = xr.Dataset({"var": da, "time": time, "lon": lon, "lat": lat})
    return ds.set_coords(["lon", "lat"])


def test_finalize_dataset_crs(monkeypatch):
    """Test finalize_dataset."""
    product = "product"
    scan_mode = "scan_mode"
    ds = get_sample_orbit_dataset()

    # Patch
    def mock_set_coordinates(ds, *args, **kwargs):
        return ds.assign_coords({"decoding_coordinates": True})

    monkeypatch.setattr(conventions, "set_coordinates", mock_set_coordinates)

    # Check decoding coordinates
    ds = finalize_dataset(ds, product=product, scan_mode=scan_mode, decode_cf=False)
    assert ds.coords["decoding_coordinates"].to_numpy()

    # Check CF decoding
    original_decode_cf = xr.decode_cf

    def mock_decode_cf(ds, *args, **kwargs):
        ds.attrs["decoded"] = True
        return original_decode_cf(ds, *args, **kwargs)

    monkeypatch.setattr(xr, "decode_cf", mock_decode_cf)

    ds = get_sample_orbit_dataset()
    ds = finalize_dataset(ds, product=product, scan_mode=scan_mode, decode_cf=True)
    assert ds.attrs["decoded"]

    # Check CRS information
    def mock_set_dataset_crs(ds, *args, **kwargs):
        ds.attrs["crs"] = True
        return ds

    monkeypatch.setattr(conventions, "set_dataset_crs", mock_set_dataset_crs)

    ds = get_sample_orbit_dataset()
    ds = finalize_dataset(ds, product=product, scan_mode=scan_mode, decode_cf=False)
    assert ds.attrs["crs"]


def test_finalize_dataset_reshaping(monkeypatch):
    """Test reshaping in finalize_dataset."""
    product = "product"
    scan_mode = "scan_mode"

    # Test grid case
    ds = get_sample_grid_dataset()  # (lat, lon, time, other)
    ds = finalize_dataset(ds, product=product, scan_mode=scan_mode, decode_cf=False)
    expected_dims = ("time", "other", "lat", "lon")
    assert ds["var"].dims == expected_dims

    # Test orbit case
    ds = get_sample_orbit_dataset()  # (other, along_track, cross_track)
    ds = finalize_dataset(ds, product=product, scan_mode=scan_mode, decode_cf=False)
    expected_dims = ("cross_track", "along_track", "other")
    assert ds["var"].dims == expected_dims

    # Test orbit case only along_track
    ds = get_sample_orbit_dataset()
    ds = ds.isel(cross_track=0)  # (other, along_track)
    ds = finalize_dataset(ds, product=product, scan_mode=scan_mode, decode_cf=False)
    expected_dims = ("along_track", "other")
    assert ds["var"].dims == expected_dims


def test_finalize_dataset_time_subsetting(monkeypatch):
    """Test time subsetting in finalize_dataset."""
    product = "product"
    scan_mode = "scan_mode"
    ds = get_sample_orbit_dataset()

    def mock_subset_by_time(ds, start_time, end_time):
        ds.attrs["start_time"] = start_time
        ds.attrs["end_time"] = end_time
        return ds

    monkeypatch.setattr(conventions, "subset_by_time", mock_subset_by_time)

    start_time = datetime(2018, 1, 1, 12, 30, 0)
    end_time = datetime(2018, 1, 1, 12, 32, 0)
    ds = finalize_dataset(
        ds,
        product=product,
        scan_mode=scan_mode,
        decode_cf=False,
        start_time=start_time,
        end_time=end_time,
    )
    assert ds.attrs["start_time"] == start_time
    assert ds.attrs["end_time"] == end_time


def test_finalize_dataset_time_encoding():
    """Test time encoding int finalize_dataset."""
    product = "product"
    scan_mode = "scan_mode"
    ds = get_sample_orbit_dataset()
    ds = finalize_dataset(ds, product=product, scan_mode=scan_mode, decode_cf=False)
    expected_time_encoding = {
        "units": "seconds since 1970-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
    }
    assert ds["time"].encoding == expected_time_encoding


def test_finalize_dataset_attrs(monkeypatch):
    """Test addition of attributes in finalize_dataset."""
    product = "product"
    scan_mode = "scan_mode"
    ds = get_sample_orbit_dataset()

    def mock_set_coords_attrs(ds, *args, **kwargs):
        ds.attrs["coords_attrs"] = True
        return ds

    monkeypatch.setattr(conventions, "set_coords_attrs", mock_set_coords_attrs)

    def mock_add_history(ds, *args, **kwargs):
        ds.attrs["history"] = True
        return ds

    monkeypatch.setattr(conventions, "add_history", mock_add_history)

    ds = finalize_dataset(ds, product=product, scan_mode=scan_mode, decode_cf=False)
    assert ds.attrs["coords_attrs"]
    assert ds.attrs["history"]
    assert ds.attrs["gpm_api_product"] == product
