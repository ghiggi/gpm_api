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
"""This module test the manipulation utilities."""

import numpy as np
import pytest
import xarray as xr

from gpm.dataset.dimensions import VERTICAL_DIMS
from gpm.utils.manipulations import (
    _get_vertical_dim,
    extract_dataset_above_bin,
    extract_dataset_below_bin,
    extract_l2_dataset,
    get_bin_dataarray,
    get_bright_band_mask,
    get_height_at_bin,
    get_height_at_temperature,
    get_height_dataarray,
    get_liquid_phase_mask,
    get_range_axis,
    get_solid_phase_mask,
    integrate_profile_concentration,
    select_bin_variables,
    select_frequency_variables,
    select_spatial_2d_variables,
    select_spatial_3d_variables,
    select_transect_variables,
    slice_range_at_bin,
    slice_range_at_height,
    slice_range_at_max_value,
    slice_range_at_min_value,
    slice_range_at_temperature,
    slice_range_at_value,
    subset_range_where_values,
    subset_range_with_valid_data,
)

# Fixtures imported from gpm.tests.conftest:
# - orbit_dataarray
# - orbit_dataset_collection
# - grid_dataset_collection


# Fixtures #####################################################################


THICKNESS = 8


def create_bins_dataarray(value=1, cross_track_size=1, along_track_size=1):
    bins = np.ones((cross_track_size, along_track_size)) * value
    da_bins = xr.DataArray(bins, dims=["cross_track", "along_track"]).squeeze()
    return da_bins


def create_3d_dataarray(cross_track_size=5, along_track_size=6, range_size=8, thickness=THICKNESS):
    # Create data
    height_1d = np.arange(range_size) * thickness
    data = np.arange(range_size * cross_track_size * along_track_size)
    # Create radar data
    # - Set range as first dimension to have values from 1 to range_size in [:, 0, 0]
    range_bin_id = np.arange(1, range_size + 1)
    data_3d = data.reshape(range_size, cross_track_size, along_track_size).astype(float)
    height_3d = height_1d[:, np.newaxis, np.newaxis] * np.ones((range_size, cross_track_size, along_track_size))
    # Create DataArray
    da = xr.DataArray(data_3d, dims=["range", "cross_track", "along_track"])
    da_height = xr.DataArray(height_3d, dims=["range", "cross_track", "along_track"])
    da = da.assign_coords(
        {"height": da_height, "range": range_bin_id},
    )
    # Reorder to classical order
    return da.transpose("cross_track", "along_track", "range")


@pytest.fixture()
def dataarray_3d() -> xr.DataArray:
    da = create_3d_dataarray()
    # Reorder to classical order
    da = da.transpose("cross_track", "along_track", "range")
    # dataarray_3d = da
    return da


# Public functions #############################################################


def test_integrate_profile_concentration(
    dataarray_3d: xr.DataArray,
) -> None:
    """Test integrate_profile_concentration function."""
    returned_da = integrate_profile_concentration(dataarray_3d, name="integrated")

    expected_data = np.sum(dataarray_3d.data * THICKNESS, axis=-1)
    np.testing.assert_allclose(returned_da.to_numpy(), expected_data)

    # With scaling factor
    scale_factor = 2
    expected_data /= scale_factor
    returned_da = integrate_profile_concentration(
        dataarray_3d,
        name="integrated",
        scale_factor=scale_factor,
        units="units",
    )
    np.testing.assert_allclose(returned_da.to_numpy(), expected_data)

    # Missing units
    with pytest.raises(ValueError):
        integrate_profile_concentration(
            dataarray_3d,
            name="integrated",
            scale_factor=-1,
        )


def test_slice_range_at_bin() -> None:
    """Test slice_range_at_bin function."""
    # Create data
    dataarray_3d = create_3d_dataarray(cross_track_size=2, along_track_size=2, range_size=3)
    da_bins = create_bins_dataarray(value=1, cross_track_size=2, along_track_size=2)
    expected_sliced_data = dataarray_3d.data[:, :, 0]  # bin value 1 correspond to index 0 in python

    # Test with a data array
    returned_da = slice_range_at_bin(dataarray_3d, bins=da_bins)
    np.testing.assert_allclose(returned_da.to_numpy(), expected_sliced_data)

    # Test with a dataset
    variable = "variable"
    ds = xr.Dataset({variable: dataarray_3d})
    returned_da = slice_range_at_bin(ds, bins=da_bins)[variable]
    np.testing.assert_allclose(returned_da.to_numpy(), expected_sliced_data)

    # Test with bins in dataset
    bins_name = "bins"
    ds[bins_name] = da_bins
    returned_da = slice_range_at_bin(ds, bins=bins_name)[variable]
    np.testing.assert_allclose(returned_da.to_numpy(), expected_sliced_data)

    # Test raise error if providing list of range bins
    with pytest.raises(TypeError):  # list --> suggest to use sel(range...)
        slice_range_at_bin(ds, bins=[2, 3, 5])

    with pytest.raises(TypeError):  # int --> suggest to use sel(range...)
        slice_range_at_bin(ds, bins=2)

    with pytest.raises(TypeError):  # float
        slice_range_at_bin(ds, bins=2.1)


def test_extract_dataset_below_bin() -> None:
    """Test extract_dataset_below_bin function."""
    # Create data
    range_size = 5
    bin_value = 2
    cross_track_size = 4
    along_track_size = 1
    dataarray_3d = create_3d_dataarray(
        cross_track_size=cross_track_size,
        along_track_size=along_track_size,
        range_size=range_size,
    )
    da_bins = create_bins_dataarray(
        value=bin_value,
        cross_track_size=cross_track_size,
        along_track_size=along_track_size,
    )
    da_bins_above = create_bins_dataarray(
        value=bin_value - 1,
        cross_track_size=cross_track_size,
        along_track_size=along_track_size,
    )
    da_bins_below = create_bins_dataarray(
        value=bin_value + 1,
        cross_track_size=cross_track_size,
        along_track_size=along_track_size,
    )

    var = "variable"
    bins = "bins"
    ds = xr.Dataset()
    ds[var] = dataarray_3d
    ds[bins] = da_bins
    ds["bins_above"] = da_bins_above
    ds["bins_below"] = da_bins_below

    # DEBUG
    # new_range_size = None
    # strict = False
    # reverse = False
    # ds[var]

    # ---------------------------
    # Test with default arguments
    ds_out = extract_dataset_below_bin(ds, bins=bins)  # new_range_size=None, strict=False, reverse=False
    # - Test bins values
    assert np.all(ds_out[bins].data == 1)
    assert np.all(np.isnan(ds_out["bins_above"].data))
    expected_da_bins_below = da_bins_below - da_bins + 1
    np.testing.assert_allclose(ds_out["bins_below"].data, expected_da_bins_below.data)
    # - Test variable values (first cross-track index)
    expected_data = np.array([4, 8, 12, 16, np.nan])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)

    # ---------------------------
    # Test with DataArray with additional dimension
    ds["var"] = ds[var].expand_dims({"radar_frequency": 2})
    ds_out = extract_dataset_below_bin(ds, bins=bins, strict=False)
    expected_data = np.array([4, 8, 12, 16, np.nan])
    np.testing.assert_allclose(ds_out["var"].isel(radar_frequency=0, cross_track=0).data.squeeze(), expected_data)
    np.testing.assert_allclose(ds_out["var"].isel(radar_frequency=1, cross_track=0).data.squeeze(), expected_data)

    # ---------------------------
    # Test with a dataset with subsetted range dimension
    ds_subset = ds.isel(range=slice(1, 4))
    ds_out = extract_dataset_below_bin(ds_subset, bins=bins, strict=False)
    # - Test bins values
    assert np.all(ds_out[bins].data == 1)
    assert np.all(np.isnan(ds_out["bins_above"].data))
    expected_da_bins_below = da_bins_below - da_bins + 1
    np.testing.assert_allclose(ds_out["bins_below"].data, expected_da_bins_below.data)
    # - Test variable values (first cross-track index)
    expected_data = np.array([4, 8, 12])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)

    # ---------------------------
    # Test with strict=True
    ds_out = extract_dataset_below_bin(ds, bins=bins, strict=True)  # new_range_size=None, reverse=False
    # - Test bins values
    assert np.all(ds_out[bins].data == 1)
    assert np.all(np.isnan(ds_out["bins_above"].data))
    expected_da_bins_below = da_bins_below - da_bins + 1
    np.testing.assert_allclose(ds_out["bins_below"].data, expected_da_bins_below.data)
    # - Test variable values (first cross-track index)
    expected_data = np.array([8, 12, 16, np.nan, np.nan])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)

    # ---------------------------
    # Test with reverse=True
    ds_out = extract_dataset_below_bin(ds, bins=bins, reverse=True)  # new_range_size=None, strict=False
    # - Test bins values
    assert np.all(ds_out[bins].data == range_size)
    assert np.all(np.isnan(ds_out["bins_above"].data))
    expected_da_bins_below = range_size - (da_bins_below - da_bins + 1) + 1
    np.testing.assert_allclose(ds_out["bins_below"].data, expected_da_bins_below.data)
    # - Test variable values (first cross-track index)
    expected_data = np.array([np.nan, 16, 12, 8, 4])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)

    # ---------------------------
    # Test with reverse=True + new_range_size = 4
    new_range_size = 4
    ds_out = extract_dataset_below_bin(ds, bins=bins, reverse=True, new_range_size=new_range_size)  # strict=False
    # - Test bins values
    assert np.all(ds_out[bins].data == new_range_size)
    assert np.all(np.isnan(ds_out["bins_above"].data))
    expected_da_bins_below = new_range_size - (da_bins_below - da_bins + 1) + 1
    np.testing.assert_allclose(ds_out["bins_below"].data, expected_da_bins_below.data)
    # - Test variable values (first cross-track index)
    expected_data = np.array([16, 12, 8, 4])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)

    # ---------------------------
    # Test with bins = range_size
    da_bins_max = create_bins_dataarray(
        value=range_size,
        cross_track_size=cross_track_size,
        along_track_size=along_track_size,
    )
    ds[bins] = da_bins_max
    # - With strict=False --> It works
    ds_out = extract_dataset_below_bin(ds, bins=bins, strict=False)
    expected_data = np.array([16, np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)
    ds_out = extract_dataset_below_bin(ds, bins=bins, strict=False, reverse=True)
    expected_data = np.array([np.nan, np.nan, np.nan, np.nan, 16])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)
    # - With strict=True, it fails if all bins are the last range gate
    with pytest.raises(ValueError):
        extract_dataset_below_bin(ds, bins=bins, strict=True)
    # - With strict=True and some valid bins (at cross_track_idx=0), it works
    ds[bins].data[0] = bin_value
    ds_out = extract_dataset_below_bin(ds, bins=bins, strict=True)
    assert np.all(np.isnan(ds_out[var].isel(cross_track=slice(1, None)).data))
    expected_data = np.array([8, 12, 16, np.nan, np.nan])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)

    # ---------------------------
    # Test with bins = 1
    ds[bins] = create_bins_dataarray(value=1, cross_track_size=cross_track_size, along_track_size=along_track_size)
    ds_out = extract_dataset_below_bin(ds, bins=bins, strict=False)
    np.testing.assert_allclose(ds_out[var].data.squeeze(), ds[var].data.squeeze())


def test_extract_dataset_above_bin() -> None:
    """Test extract_dataset_above_bin function."""
    # Create data
    range_size = 5
    bin_value = 3
    cross_track_size = 4
    along_track_size = 1
    dataarray_3d = create_3d_dataarray(
        cross_track_size=cross_track_size,
        along_track_size=along_track_size,
        range_size=range_size,
    )
    da_bins = create_bins_dataarray(
        value=bin_value,
        cross_track_size=cross_track_size,
        along_track_size=along_track_size,
    )
    da_bins.data[0] = da_bins.data[0] + 1
    da_bins_above = create_bins_dataarray(
        value=bin_value - 1,
        cross_track_size=cross_track_size,
        along_track_size=along_track_size,
    )
    da_bins_above.data[0] = da_bins_above.data[0] + 1
    da_bins_below = create_bins_dataarray(
        value=bin_value + 1,
        cross_track_size=cross_track_size,
        along_track_size=along_track_size,
    )
    da_bins_below.data[0] = da_bins_below.data[0] + 1

    var = "variable"
    bins = "bins"
    ds = xr.Dataset()
    ds[var] = dataarray_3d
    ds[bins] = da_bins
    ds["bins_above"] = da_bins_above
    ds["bins_below"] = da_bins_below

    # DEBUG
    # new_range_size = None
    # strict = False
    # reverse = False
    # ds[var]

    ds = ds.transpose("range", ...)

    # ---------------------------
    # Test with default arguments
    ds_out = extract_dataset_above_bin(ds, bins=bins)  # new_range_size=None, strict=False, reverse=False
    # - Test bins values
    assert np.all(ds_out[bins].data == range_size)
    assert np.all(np.isnan(ds_out["bins_below"].data))
    new_range_size = range_size
    expected_da_bins_below = new_range_size - (da_bins - da_bins_above)
    np.testing.assert_allclose(ds_out["bins_above"].data, expected_da_bins_below.data)
    # - Test variable values (first cross-track index)
    expected_data = np.array([np.nan, 0, 4, 8, 12])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)

    # ---------------------------
    # Test with range not in last position
    # --> NOTE: I had to transpose range to last position in the function to pass ! Mysterious bug
    ds_out = extract_dataset_above_bin(ds.transpose("range", ...), bins=bins)
    # - Test bins values
    assert np.all(ds_out[bins].data == range_size)
    assert np.all(np.isnan(ds_out["bins_below"].data))
    new_range_size = range_size
    expected_da_bins_below = new_range_size - (da_bins - da_bins_above)
    np.testing.assert_allclose(ds_out["bins_above"].data, expected_da_bins_below.data)
    # - Test variable values (first cross-track index)
    expected_data = np.array([np.nan, 0, 4, 8, 12])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)

    # ---------------------------
    # Test with DataArray with additional dimension
    ds["var"] = ds[var].expand_dims({"radar_frequency": 2})
    ds_out = extract_dataset_above_bin(ds, bins=bins, strict=False)
    expected_data = np.array([np.nan, 0, 4, 8, 12])
    np.testing.assert_allclose(ds_out["var"].isel(radar_frequency=0, cross_track=0).data.squeeze(), expected_data)
    np.testing.assert_allclose(ds_out["var"].isel(radar_frequency=1, cross_track=0).data.squeeze(), expected_data)

    # ---------------------------
    # Test with a dataset with subsetted range dimension
    ds_subset = ds.isel(range=slice(1, 4))
    ds_out = extract_dataset_above_bin(ds_subset, bins=bins, strict=False)
    # - Test bins values
    assert np.all(ds_out[bins].data == len(ds_subset["range"]))
    assert np.all(np.isnan(ds_out["bins_below"].data))
    new_range_size = len(ds_subset["range"])
    expected_da_bins_below = new_range_size - (da_bins - da_bins_above)
    np.testing.assert_allclose(ds_out["bins_above"].data, expected_da_bins_below.data)
    # - Test variable values (first cross-track index)
    expected_data = np.array([4, 8, 12])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)

    # ---------------------------
    # Test with strict=True
    ds_out = extract_dataset_above_bin(ds, bins=bins, strict=True)  # new_range_size=None, reverse=False
    # - Test bins values
    assert np.all(ds_out[bins].data == range_size)
    assert np.all(np.isnan(ds_out["bins_below"].data))
    new_range_size = range_size
    expected_da_bins_below = new_range_size - (da_bins - da_bins_above)
    np.testing.assert_allclose(ds_out["bins_above"].data, expected_da_bins_below.data)
    # - Test variable values (first cross-track index)
    expected_data = np.array([np.nan, np.nan, 0, 4, 8])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)

    # ---------------------------
    # Test with reverse=True
    ds_out = extract_dataset_above_bin(ds, bins=bins, reverse=True)  # new_range_size=None, strict=False
    # - Test bins values
    assert np.all(ds_out[bins].data == 1)
    assert np.all(np.isnan(ds_out["bins_below"].data))
    new_range_size = range_size
    expected_da_bins_below = da_bins - da_bins_above + 1
    np.testing.assert_allclose(ds_out["bins_above"].data, expected_da_bins_below.data)
    # - Test variable values (first cross-track index)
    expected_data = np.array([12, 8, 4, 0, np.nan])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)

    # ---------------------------
    # Test with reverse=True + new_range_size = 3
    new_range_size = 3
    ds_out = extract_dataset_above_bin(ds, bins=bins, new_range_size=new_range_size)  # strict=False
    # - Test bins values
    assert np.all(ds_out[bins].data == new_range_size)
    assert np.all(np.isnan(ds_out["bins_below"].data))
    expected_da_bins_below = da_bins - da_bins_above + 1
    np.testing.assert_allclose(ds_out["bins_above"].data, expected_da_bins_below.data)
    # - Test variable values (first cross-track index)
    expected_data = np.array([4, 8, 12])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)

    # ---------------------------
    # Test with bins = 1
    da_bins_1 = create_bins_dataarray(value=1, cross_track_size=cross_track_size, along_track_size=along_track_size)
    ds[bins] = da_bins_1
    # - With strict=False --> It works
    ds_out = extract_dataset_above_bin(ds, bins=bins, strict=False)
    expected_data = np.array([np.nan, np.nan, np.nan, np.nan, 0])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)
    ds_out = extract_dataset_above_bin(ds, bins=bins, strict=False, reverse=True)
    expected_data = np.array([0, np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)
    # - With strict=True, it fails if all bins are the first range gate
    with pytest.raises(ValueError):
        extract_dataset_above_bin(ds, bins=bins, strict=True)
    # - With strict=True and some valid bins (at cross_track_idx=0), it works
    ds[bins].data[0] = bin_value
    ds_out = extract_dataset_above_bin(ds, bins=bins, strict=True)
    assert np.all(np.isnan(ds_out[var].isel(cross_track=slice(1, None)).data))
    expected_data = np.array([np.nan, np.nan, np.nan, 0, 4])
    np.testing.assert_allclose(ds_out[var].isel(cross_track=0).data.squeeze(), expected_data)

    # ---------------------------
    # Test with bins = range_size
    ds[bins] = create_bins_dataarray(
        value=range_size,
        cross_track_size=cross_track_size,
        along_track_size=along_track_size,
    )
    ds_out = extract_dataset_above_bin(ds, bins=bins, strict=False)
    np.testing.assert_allclose(ds_out[var].data.squeeze(), ds[var].data.squeeze())


@pytest.mark.parametrize(("scan_mode", "range_size"), [("FS", 176), ("NS", 176), ("HS", 88), ("MS", 176)])
def test_extract_l2_dataset(scan_mode, range_size) -> None:
    """Test extract_l2_dataset function."""
    # Create data
    cross_track_size = 4
    along_track_size = 1
    dataarray_3d = create_3d_dataarray(
        cross_track_size=cross_track_size,
        along_track_size=along_track_size,
        range_size=250,
    )
    da_bins = create_bins_dataarray(
        value=197,
        cross_track_size=cross_track_size,
        along_track_size=along_track_size,
    )
    var = "variable"
    bins = "binEllipsoid"
    ds = xr.Dataset()
    ds[var] = dataarray_3d
    ds[bins] = da_bins
    ds.attrs["ScanMode"] = scan_mode

    ds_l2 = extract_l2_dataset(
        ds=ds,
        bin_ellipsoid=bins,
        shortened_range=True,
        new_range_size=None,
    )
    assert len(ds_l2["range"]) == range_size


def test_get_bin_dataarray() -> None:
    """Test extract_dataset_above_bin function."""
    # Create dataset
    range_size = 5
    cross_track_size = 4
    along_track_size = 1
    dataarray_3d = create_3d_dataarray(
        cross_track_size=cross_track_size,
        along_track_size=along_track_size,
        range_size=range_size,
    )
    var_3d = "3d_variable"
    ds = xr.Dataset()
    ds[var_3d] = dataarray_3d

    # Test bins variable with range dimension
    with pytest.raises(ValueError) as excinfo:
        get_bin_dataarray(ds, bins="3d_variable")
    assert "The bin DataArray must not have the 'range' dimension." in str(excinfo.value)

    # Test bins variable with frequency dimension
    ds["var_multifrequency"] = ds[var_3d].expand_dims({"radar_frequency": 2}).isel(range=0)
    with pytest.raises(ValueError) as excinfo:
        get_bin_dataarray(ds, bins="var_multifrequency")
    assert "The bin DataArray must not have the 'radar_frequency' dimension." in str(excinfo.value)

    # Test bins variable with other dimension
    ds["var_multi_dim"] = ds[var_3d].expand_dims({"other_dim": 2}).isel(range=0)
    with pytest.raises(ValueError) as excinfo:
        get_bin_dataarray(ds, bins="var_multi_dim")
    assert "The bin DataArray is allowed to only have spatial dimensions." in str(excinfo.value)

    # Test out-of-range bins values
    ds["invalid_bins"] = create_bins_dataarray(
        value=range_size + 1,
        cross_track_size=cross_track_size,
        along_track_size=along_track_size,
    )
    with pytest.raises(ValueError) as excinfo:
        get_bin_dataarray(ds, bins="invalid_bins")
    assert "All range bin indices are outside of the available range gates" in str(excinfo.value)

    # Test nans bins values
    ds["nan_bins"] = create_bins_dataarray(
        value=np.nan,
        cross_track_size=cross_track_size,
        along_track_size=along_track_size,
    )
    with pytest.raises(ValueError) as excinfo:
        get_bin_dataarray(ds, bins="nan_bins")
    assert "All range bin indices are NaN" in str(excinfo.value)

    # Test mixture of invalid values
    ds["nan_bins"].data[0] = range_size + 1
    with pytest.raises(ValueError) as excinfo:
        get_bin_dataarray(ds, bins="nan_bins")
    assert "All range bin indices are invalid" in str(excinfo.value)


def test_get_height_dataarray():
    """Test height DataArray extraction from xarray object."""
    # Create xarray objects
    dataarray_3d = create_3d_dataarray(
        cross_track_size=3,
        along_track_size=3,
        range_size=3,
    )
    dataarray_3d = dataarray_3d.drop_vars("height")
    ds = xr.Dataset()
    ds["var_3d"] = dataarray_3d
    ds["height"] = dataarray_3d.copy()

    # Test height extraction from Dataset variable
    xr.testing.assert_identical(ds["height"], get_height_dataarray(ds))

    # Test height extraction from Dataset coordinate
    ds = ds.set_coords("height")
    xr.testing.assert_identical(ds["height"], get_height_dataarray(ds))

    # Test height extraction from DataArray coordinate
    xr.testing.assert_identical(ds["height"], get_height_dataarray(ds["var_3d"]))

    # Test raise error if DataArray without "height" coordinate
    da = ds["var_3d"].drop_vars("height")
    with pytest.raises(ValueError) as excinfo:
        get_height_dataarray(da)
    assert "Expecting a xarray.DataArray with the 'height' coordinate" in str(excinfo.value)


def test_get_height_at_bin() -> None:
    """Test get_height_at_bin function."""
    # Create data
    dataarray_3d = create_3d_dataarray(cross_track_size=2, along_track_size=2, range_size=3)
    da_bins = create_bins_dataarray(value=1, cross_track_size=2, along_track_size=2)
    expected_sliced_data = dataarray_3d["height"].data[:, :, 0]  # bin value 1 correspond to index 0 in python

    # Test with a data array
    returned_da = get_height_at_bin(dataarray_3d, bins=da_bins)
    np.testing.assert_allclose(returned_da.to_numpy(), expected_sliced_data)

    # Test with a dataset
    variable = "variable"
    ds = xr.Dataset({variable: dataarray_3d})
    returned_da = get_height_at_bin(ds, bins=da_bins)
    np.testing.assert_allclose(returned_da.to_numpy(), expected_sliced_data)

    # Test with bins in dataset
    bins_name = "bins"
    ds[bins_name] = da_bins
    returned_da = get_height_at_bin(ds, bins=bins_name)
    np.testing.assert_allclose(returned_da.to_numpy(), expected_sliced_data)


def test_subset_range_with_valid_data(
    dataarray_3d: xr.DataArray,
) -> None:
    """Test subset_range_with_valid_data function."""
    dataarray_3d.data[:, :, 0] = np.nan  # Fully removes this height level
    dataarray_3d.data[:2, :3, 1] = np.nan  # These are kept
    returned_da = subset_range_with_valid_data(dataarray_3d)
    expected_data = dataarray_3d.data[:, :, 1:]
    np.testing.assert_allclose(returned_da.to_numpy(), expected_data)

    # Test fully nan
    dataarray_3d.data[:, :, :] = np.nan
    with pytest.raises(ValueError):
        subset_range_with_valid_data(dataarray_3d)


def test_subset_range_where_values(
    dataarray_3d: xr.DataArray,
) -> None:
    """Test subset_range_where_values function."""
    # Test with no values within range
    dataarray_3d.data[:, :, :] = 0
    vmin = 10
    vmax = 20
    with pytest.raises(ValueError):
        returned_da = subset_range_where_values(
            dataarray_3d,
            vmin=vmin,
            vmax=vmax,
        )

    # Test with valid values
    dataarray_3d.data[:, :, 1] = 11  # keep this layer
    dataarray_3d.data[2, 2, 2] = 12  # layer kept even if single value is valid
    dataarray_3d.data[2, 2, 3] = 21  # not valid
    returned_da = subset_range_where_values(
        dataarray_3d,
        vmin=vmin,
        vmax=vmax,
    )
    expected_data = dataarray_3d.data[:, :, 1:3]
    np.testing.assert_allclose(returned_da.data, expected_data)


def test_slice_range_at_value(
    dataarray_3d: xr.DataArray,
) -> None:
    """Test slice_range_at_value function."""
    value = 100
    returned_slice = slice_range_at_value(dataarray_3d, value)
    vertical_indices = np.abs(dataarray_3d - value).argmin(dim="range")
    expected_slice = dataarray_3d.isel({"range": vertical_indices}).data
    np.testing.assert_allclose(returned_slice.data, expected_slice)


def test_slice_range_at_max_value(
    dataarray_3d: xr.DataArray,
) -> None:
    """Test slice_range_at_max_value function."""
    returned_slice = slice_range_at_max_value(dataarray_3d)
    expected_slice = dataarray_3d.isel({"range": -1}).data
    np.testing.assert_allclose(returned_slice.data, expected_slice)


def test_slice_range_at_min_value(
    dataarray_3d: xr.DataArray,
) -> None:
    """Test slice_range_at_min_value function."""
    returned_slice = slice_range_at_min_value(dataarray_3d)
    expected_slice = dataarray_3d.isel({"range": 0}).data
    np.testing.assert_allclose(returned_slice.data, expected_slice)


@pytest.mark.parametrize("variable", ["airTemperature", "height"])
def test_slice_range_at(
    variable: str,
    dataarray_3d: xr.DataArray,
) -> None:
    """Test slice_range_at_temperature and slice_range_at_height functions."""
    manipulations_functions = {
        "airTemperature": slice_range_at_temperature,
        "height": slice_range_at_height,
    }
    manipulations_function = manipulations_functions[variable]

    ds = xr.Dataset({variable: dataarray_3d})
    value = 105
    returned_slice = manipulations_function(ds, value)
    expected_slice = dataarray_3d.isel({"range": 3}).data
    np.testing.assert_allclose(returned_slice[variable].data, expected_slice)


def test_get_height_at_temperature(
    dataarray_3d: xr.DataArray,
) -> None:
    """Test get_height_at_temperature function."""
    da_temperature = dataarray_3d.copy()
    da_height = dataarray_3d.copy()
    da_height.data[:] = da_height.data[:] + 500

    temperature = 105
    returned_da = get_height_at_temperature(da_height, da_temperature, temperature)
    expected_data = da_height.data[:, :, 3]
    np.testing.assert_allclose(returned_da.data, expected_data)


def test_get_range_axis(
    dataarray_3d: xr.DataArray,
) -> None:
    """Test get_range_axis function."""
    returned_index = get_range_axis(dataarray_3d)
    assert returned_index == 2


def test_get_bright_band_mask() -> None:
    """Test get_bright_band_mask function."""
    da = create_3d_dataarray(cross_track_size=2, along_track_size=2, range_size=4)
    ds = xr.Dataset({"variable": da})
    ds["binBBTop"] = create_bins_dataarray(value=2, cross_track_size=2, along_track_size=2)
    ds["binBBBottom"] = create_bins_dataarray(value=3, cross_track_size=2, along_track_size=2)

    da_mask = get_bright_band_mask(ds)
    assert np.all(~da_mask.isel(range=0).data)
    assert np.all(da_mask.isel(range=1).data)
    assert np.all(da_mask.isel(range=2).data)
    assert np.all(~da_mask.isel(range=3).data)


class TestGetPhaseMask:
    """Test get_liquid_phase_mask and get_solid_phase_mask functions."""

    rng = np.random.default_rng()
    height_zero_deg = rng.integers(3, 6, size=(5, 6)) * 8
    da_height_zero_deg = xr.DataArray(height_zero_deg, dims=["cross_track", "along_track"])

    @pytest.fixture()
    def phase_dataarray(
        self,
        dataarray_3d: xr.DataArray,
    ) -> xr.DataArray:
        ds = xr.Dataset(
            {
                "variable": dataarray_3d,
                "heightZeroDeg": self.da_height_zero_deg,
            },
        )
        return ds

    def test_get_liquid_phase_mask(
        self,
        phase_dataarray: xr.Dataset,
    ) -> None:
        """Test get_liquid_phase_mask function."""
        returned_mask = get_liquid_phase_mask(phase_dataarray)
        expected_mask = phase_dataarray["height"] < self.da_height_zero_deg
        np.testing.assert_allclose(returned_mask.data, expected_mask.data)

    def test_get_solid_phase_mask(
        self,
        phase_dataarray: xr.Dataset,
    ) -> None:
        """Test get_solid_phase_mask function."""
        returned_mask = get_solid_phase_mask(phase_dataarray)
        expected_mask = phase_dataarray["height"] >= self.da_height_zero_deg
        np.testing.assert_allclose(returned_mask.data, expected_mask.data)


class TestSelectVariables:

    def test_spatial_3d(
        self,
        orbit_dataset_collection: xr.Dataset,
        grid_dataset_collection: xr.Dataset,
    ) -> None:
        """Test select_spatial_3d_variables function."""
        # Orbit
        returned_ds = select_spatial_3d_variables(orbit_dataset_collection)
        expected_ds = orbit_dataset_collection[["variable_3d"]]
        xr.testing.assert_identical(returned_ds, expected_ds)

        # Grid
        returned_ds = select_spatial_3d_variables(grid_dataset_collection)
        expected_ds = grid_dataset_collection[["variable_3d"]]
        xr.testing.assert_identical(returned_ds, expected_ds)

    def test_spatial_2d(
        self,
        orbit_dataset_collection: xr.Dataset,
        grid_dataset_collection: xr.Dataset,
    ) -> None:
        """Test select_spatial_2d_variables function."""
        # Orbit
        returned_ds = select_spatial_2d_variables(orbit_dataset_collection)
        expected_ds = orbit_dataset_collection[["bin_variable", "variableBin", "variable_2d"]]
        xr.testing.assert_identical(returned_ds, expected_ds)

        # Grid
        returned_ds = select_spatial_2d_variables(grid_dataset_collection)
        expected_ds = grid_dataset_collection[["variable_2d"]]
        xr.testing.assert_identical(returned_ds, expected_ds)

    def test_transect(
        self,
        orbit_dataset_collection: xr.Dataset,
        grid_dataset_collection: xr.Dataset,
    ) -> None:
        """Test select_transect_variables function."""
        # Orbit
        orbit_dataset = orbit_dataset_collection.isel(along_track=0)
        returned_ds = select_transect_variables(orbit_dataset)
        expected_ds = orbit_dataset[["variable_3d"]]
        xr.testing.assert_identical(returned_ds, expected_ds)

        # Grid
        grid_dataset = grid_dataset_collection.isel(lon=0)
        returned_ds = select_transect_variables(grid_dataset)
        expected_ds = grid_dataset[["variable_3d"]]
        xr.testing.assert_identical(returned_ds, expected_ds)

    def test_bin_variables(
        self,
        orbit_dataset_collection: xr.Dataset,
    ) -> None:
        """Test select_bin_variables function."""
        # Orbit
        returned_ds = select_bin_variables(orbit_dataset_collection)
        expected_ds = orbit_dataset_collection[["bin_variable", "variableBin"]]
        xr.testing.assert_identical(returned_ds, expected_ds)

    def test_frequency_variables(
        self,
        orbit_dataset_collection,
    ) -> None:
        """Test select_frequency_variables function."""
        returned_ds = select_frequency_variables(orbit_dataset_collection)
        expected_ds = orbit_dataset_collection[["variable_frequency"]]
        xr.testing.assert_identical(returned_ds, expected_ds)


# Private functions ############################################################


def test__get_vertical_dim() -> None:
    """Test _get_vertical_dim function."""
    for vertical_dim in VERTICAL_DIMS:
        n_dims = 2
        da = xr.DataArray(np.zeros((0,) * n_dims), dims=["other", vertical_dim])
        assert _get_vertical_dim(da) == vertical_dim

    # Test no vertical dimension
    da = xr.DataArray(np.zeros((0,)), dims=["other"])
    with pytest.raises(ValueError):
        _get_vertical_dim(da)

    # Test multiple vertical dimensions
    da = xr.DataArray(np.zeros((0,) * 3), dims=["other", *VERTICAL_DIMS[:2]])
    with pytest.raises(ValueError):
        _get_vertical_dim(da)
