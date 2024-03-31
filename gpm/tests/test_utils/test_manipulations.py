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
import numpy as np
import pytest
import xarray as xr

from gpm.dataset.dimensions import VERTICAL_DIMS
from gpm.utils import manipulations

# Fixtures imported from gpm.tests.conftest:
# - orbit_dataarray
# - dataset_collection


# Fixtures #####################################################################


THICKNESS = 8


@pytest.fixture
def a_3d_dataarray() -> xr.DataArray:
    n_x = 5
    n_y = 6
    n_height = 8
    x = np.arange(n_x)
    y = np.arange(n_y)
    height = np.arange(n_height) * THICKNESS

    data = np.arange(n_height * n_x * n_y).reshape(n_height, n_x, n_y)
    data = data.astype(float)
    da = xr.DataArray(data, coords={"height": height, "x": x, "y": y})
    return da.transpose("x", "y", "height")


@pytest.fixture
def binnable_orbit_dataarray(
    orbit_dataarray: xr.DataArray,
) -> xr.DataArray:
    n_range = 8
    da = orbit_dataarray.expand_dims(dim={"range": n_range})
    return da.assign_coords({"gpm_range_id": ("range", np.arange(n_range))})


# Public functions #############################################################


def test_integrate_profile_concentration(
    a_3d_dataarray: xr.DataArray,
) -> None:
    """Test integrate_profile_concentration function"""

    returned_da = manipulations.integrate_profile_concentration(a_3d_dataarray, name="integrated")

    expected_data = np.sum(a_3d_dataarray.data * THICKNESS, axis=-1)
    np.testing.assert_allclose(returned_da.values, expected_data)

    # With scaling factor
    scale_factor = 2
    expected_data /= scale_factor
    returned_da = manipulations.integrate_profile_concentration(
        a_3d_dataarray,
        name="integrated",
        scale_factor=scale_factor,
        units="units",
    )
    np.testing.assert_allclose(returned_da.values, expected_data)

    # Missing units
    with pytest.raises(ValueError):
        manipulations.integrate_profile_concentration(
            a_3d_dataarray,
            name="integrated",
            scale_factor=-1,
        )


def test_check_variable_availabilty() -> None:
    """Test check_variable_availabilty function"""

    variable = "variable"
    ds = xr.Dataset({"variable": xr.DataArray()})

    # Should not raise any error
    manipulations.check_variable_availabilty(ds, variable, "arg_name")

    # Should raise an error
    with pytest.raises(ValueError):
        manipulations.check_variable_availabilty(ds, "other_variable", "arg_name")


def test_get_variable_dataarray() -> None:
    """Test get_variable_dataarray function"""

    variable = "variable"
    ds = xr.Dataset({"variable": xr.DataArray([1, 2, 3])})
    da = ds["variable"]

    xr.testing.assert_identical(manipulations.get_variable_dataarray(ds, variable), da)
    xr.testing.assert_identical(manipulations.get_variable_dataarray(da, variable), da)

    with pytest.raises(TypeError):
        manipulations.get_variable_dataarray([], variable)

    with pytest.raises(ValueError):
        manipulations.get_variable_dataarray(ds, "other_variable")


def test_get_variable_at_bin(
    a_3d_dataarray: xr.DataArray,
) -> None:
    """Test get_variable_at_bin function"""

    bins = xr.DataArray([2, 3, 5])
    expected_binned_data = a_3d_dataarray.data[:, :, [1, 2, 4]]

    # Test with a data array
    returned_da = manipulations.get_variable_at_bin(a_3d_dataarray, bins)
    np.testing.assert_allclose(returned_da.values, expected_binned_data)

    # Test with a dataset
    variable = "variable"
    ds = xr.Dataset({variable: a_3d_dataarray})
    returned_da = manipulations.get_variable_at_bin(ds, bins, variable)
    np.testing.assert_allclose(returned_da.values, expected_binned_data)

    # Test with bins in dataset
    bins_name = "bins"
    ds[bins_name] = bins
    returned_da = manipulations.get_variable_at_bin(ds, bins_name, variable)
    np.testing.assert_allclose(returned_da.values, expected_binned_data)

    with pytest.raises(TypeError):
        manipulations.get_variable_at_bin(ds, [2, 3, 5], variable)

    # Bins with more dimensions
    bins = xr.DataArray(np.arange(8).reshape(2, 2, 2) + 1)
    returned_da = manipulations.get_variable_at_bin(a_3d_dataarray, bins)

    bins = xr.DataArray(np.arange(16).reshape(2, 2, 2, 2) + 1)
    with pytest.raises(NotImplementedError):
        manipulations.get_variable_at_bin(a_3d_dataarray, bins)


def test_get_height_at_bin(
    a_3d_dataarray: xr.DataArray,
) -> None:
    """Test get_height_at_bin function"""

    ds = xr.Dataset({"height": a_3d_dataarray})
    bins = xr.DataArray([2, 3, 5])
    expected_binned_data = ds["height"].data[:, :, [1, 2, 4]]
    returned_da = manipulations.get_height_at_bin(ds, bins)
    np.testing.assert_allclose(returned_da.values, expected_binned_data)


def test_slice_range_with_valid_data(
    a_3d_dataarray: xr.DataArray,
) -> None:
    """Test slice_range_with_valid_data function"""

    a_3d_dataarray.data[:, :, 0] = np.nan  # Fully removes this height level
    a_3d_dataarray.data[:2, :3, 1] = np.nan  # These are kept
    returned_da = manipulations.slice_range_with_valid_data(a_3d_dataarray)
    expected_data = a_3d_dataarray.data[:, :, 1:]
    np.testing.assert_allclose(returned_da.values, expected_data)

    # Test fully nan
    a_3d_dataarray.data[:, :, :] = np.nan
    with pytest.raises(ValueError):
        manipulations.slice_range_with_valid_data(a_3d_dataarray)


def test_slice_range_where_values(
    a_3d_dataarray: xr.DataArray,
) -> None:
    """Test slice_range_where_values function"""

    # Test with no values within range
    a_3d_dataarray.data[:, :, :] = 0
    vmin = 10
    vmax = 20
    with pytest.raises(ValueError):
        returned_da = manipulations.slice_range_where_values(
            a_3d_dataarray,
            vmin=vmin,
            vmax=vmax,
        )

    # Test with valid values
    a_3d_dataarray.data[:, :, 1] = 11  # keep this layer
    a_3d_dataarray.data[2, 2, 2] = 12  # layer kept even if single value is valid
    a_3d_dataarray.data[2, 2, 3] = 21  # not valid
    returned_da = manipulations.slice_range_where_values(
        a_3d_dataarray,
        vmin=vmin,
        vmax=vmax,
    )
    expected_data = a_3d_dataarray.data[:, :, 1:3]
    np.testing.assert_allclose(returned_da.data, expected_data)


def test_slice_range_at_value(
    a_3d_dataarray: xr.DataArray,
) -> None:
    """Test slice_range_at_value function"""

    value = 100
    returned_slice = manipulations.slice_range_at_value(a_3d_dataarray, value)
    vertical_indices = np.abs(a_3d_dataarray - value).argmin(dim="height")
    expected_slice = a_3d_dataarray.isel(height=vertical_indices).data
    np.testing.assert_allclose(returned_slice.data, expected_slice)


def test_slice_range_at_max_value(
    a_3d_dataarray: xr.DataArray,
) -> None:
    """Test slice_range_at_max_value function"""

    returned_slice = manipulations.slice_range_at_max_value(a_3d_dataarray)
    expected_slice = a_3d_dataarray.isel(height=-1).data
    np.testing.assert_allclose(returned_slice.data, expected_slice)


def test_slice_range_at_min_value(
    a_3d_dataarray: xr.DataArray,
) -> None:
    """Test slice_range_at_min_value function"""

    returned_slice = manipulations.slice_range_at_min_value(a_3d_dataarray)
    expected_slice = a_3d_dataarray.isel(height=0).data
    np.testing.assert_allclose(returned_slice.data, expected_slice)


@pytest.mark.parametrize("variable", ["airTemperature", "height"])
def test_slice_range_at(
    variable: str,
    a_3d_dataarray: xr.DataArray,
) -> None:
    """Test slice_range_at_temperature and slice_range_at_height functions"""

    manipulations_functions = {
        "airTemperature": manipulations.slice_range_at_temperature,
        "height": manipulations.slice_range_at_height,
    }
    manipulations_function = manipulations_functions[variable]

    ds = xr.Dataset({variable: a_3d_dataarray})
    value = 105
    returned_slice = manipulations_function(ds, value)
    expected_slice = a_3d_dataarray.isel(height=3).data
    np.testing.assert_allclose(returned_slice[variable].data, expected_slice)


def test_get_height_at_temperature(
    a_3d_dataarray: xr.DataArray,
) -> None:
    """Test get_height_at_temperature function"""

    da_temperature = a_3d_dataarray.copy()
    da_height = a_3d_dataarray.copy()
    da_height.data[:] = da_height.data[:] + 500

    temperature = 105
    returned_da = manipulations.get_height_at_temperature(da_height, da_temperature, temperature)
    expected_data = da_height.data[:, :, 3]
    np.testing.assert_allclose(returned_da.data, expected_data)


def test_get_range_axis(
    a_3d_dataarray: xr.DataArray,
) -> None:
    """Test get_range_axis function"""

    returned_index = manipulations.get_range_axis(a_3d_dataarray)
    assert returned_index == 2


def test_get_dims_without(
    a_3d_dataarray: xr.DataArray,
) -> None:
    """Test get_dims_without function"""

    removeds_dims = ["x", "height"]
    returned_dims = manipulations.get_dims_without(a_3d_dataarray, removeds_dims)
    expected_dims = ["y"]
    assert returned_dims == expected_dims


def test_get_xr_shape(
    a_3d_dataarray: xr.DataArray,
) -> None:
    """Test get_xr_shape function"""

    # Test with data array
    dimensions = ["x", "height"]
    returned_shape = manipulations.get_xr_shape(a_3d_dataarray, dimensions)
    expected_shape = [5, 8]
    assert returned_shape == expected_shape

    # Test with dataset
    ds = xr.Dataset({"variable": a_3d_dataarray})
    returned_shape = manipulations.get_xr_shape(ds, dimensions)
    assert returned_shape == expected_shape


def test_create_bin_idx_data_array(
    binnable_orbit_dataarray: xr.DataArray,
) -> None:
    """Test create_bin_idx_data_array function"""

    dims = {"cross_track": 5, "along_track": 20, "range": 8}

    # Test with a data array
    returned_da = manipulations.create_bin_idx_data_array(binnable_orbit_dataarray)
    expected_shape = (dims["cross_track"], dims["along_track"], dims["range"])
    assert returned_da.shape == expected_shape

    computed_bins = returned_da.data[0, 0, :]
    expected_bins = binnable_orbit_dataarray["gpm_range_id"].data + 1
    np.testing.assert_allclose(computed_bins, expected_bins)

    # Test with a dataset
    ds = xr.Dataset({"variable": binnable_orbit_dataarray})
    returned_da = manipulations.create_bin_idx_data_array(ds)
    assert returned_da.shape == expected_shape
    computed_bins = returned_da.data[0, 0, :]
    np.testing.assert_allclose(computed_bins, expected_bins)


def test_get_bright_band_mask(
    binnable_orbit_dataarray: xr.DataArray,
) -> None:
    """Test get_bright_band_mask function"""

    ds = xr.Dataset({"variable": binnable_orbit_dataarray})
    ds["binBBTop"] = 4
    ds["binBBBottom"] = 6

    returned_band = manipulations.get_bright_band_mask(ds)
    computed_band_subset = returned_band.data[0, 0, :]
    expected_band_subset = np.array([0, 0, 0, 1, 1, 1, 0, 0], dtype=bool)
    np.testing.assert_allclose(computed_band_subset, expected_band_subset)


class TestGetPhaseMask:
    """Test get_liquid_phase_mask and get_solid_phase_mask functions"""

    height_zero_deg = np.random.randint(3, 6, size=(5, 6)) * 8
    da_height_zero_deg = xr.DataArray(height_zero_deg, dims=["x", "y"])

    @pytest.fixture
    def phase_dataarray(
        self,
        a_3d_dataarray: xr.DataArray,
    ) -> xr.DataArray:
        return xr.Dataset(
            {
                "variable": a_3d_dataarray,
                "heightZeroDeg": self.da_height_zero_deg,
            }
        )

    def test_get_liquid_phase_mask(
        self,
        phase_dataarray: xr.Dataset,
    ) -> None:
        """Test get_liquid_phase_mask function"""

        returned_mask = manipulations.get_liquid_phase_mask(phase_dataarray)
        expected_mask = (
            phase_dataarray["height"].data[:, np.newaxis, np.newaxis] < self.height_zero_deg
        )
        np.testing.assert_allclose(returned_mask.data, expected_mask)

    def test_get_solid_phase_mask(
        self,
        phase_dataarray: xr.Dataset,
    ) -> None:
        """Test get_solid_phase_mask function"""

        returned_mask = manipulations.get_solid_phase_mask(phase_dataarray)
        expected_mask = (
            phase_dataarray["height"].data[:, np.newaxis, np.newaxis] >= self.height_zero_deg
        )
        np.testing.assert_allclose(returned_mask.data, expected_mask)


class TestSelectVariables:
    def test_spatial_3d(
        self,
        dataset_collection: xr.Dataset,
    ) -> None:
        """Test select_spatial_3d_variables function"""

        returned_ds = manipulations.select_spatial_3d_variables(dataset_collection)
        expected_ds = dataset_collection[["variable_2", "variable_3"]]
        xr.testing.assert_identical(returned_ds, expected_ds)

    def test_spatial_2d(
        self,
        dataset_collection: xr.Dataset,
    ) -> None:
        """Test select_spatial_2d_variables function"""

        returned_ds = manipulations.select_spatial_2d_variables(dataset_collection)
        expected_ds = dataset_collection[["variable_0", "variable_1"]]
        xr.testing.assert_identical(returned_ds, expected_ds)

    def test_transect(
        self,
        dataset_collection: xr.Dataset,
    ) -> None:
        """Test select_transect_variables function"""

        returned_ds = manipulations.select_transect_variables(dataset_collection)
        expected_ds = dataset_collection[["variable_4", "variable_5"]]
        xr.testing.assert_identical(returned_ds, expected_ds)


# Private functions ############################################################


def test__get_vertical_dim() -> None:
    """Test _get_vertical_dim function"""

    for vertical_dim in VERTICAL_DIMS:
        n_dims = 2
        da = xr.DataArray(np.zeros((0,) * n_dims), dims=["other", vertical_dim])
        assert manipulations._get_vertical_dim(da) == vertical_dim

    # Test no vertical dimension
    da = xr.DataArray(np.zeros((0,)), dims=["other"])
    with pytest.raises(ValueError):
        manipulations._get_vertical_dim(da)

    # Test multiple vertical dimensions
    da = xr.DataArray(np.zeros((0,) * 3), dims=["other", *VERTICAL_DIMS[:2]])
    with pytest.raises(ValueError):
        manipulations._get_vertical_dim(da)
