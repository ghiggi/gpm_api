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
import pytest
import numpy as np
import xarray as xr

from gpm.utils import manipulations
from gpm.dataset.dimensions import VERTICAL_DIMS


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

    data = np.arange(n_x * n_y * n_height).reshape(n_x, n_y, n_height)
    data = data.astype(float)
    return xr.DataArray(data, coords={"x": x, "y": y, "height": height})


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


# def test_slice_range_where_values(
#     a_3d_dataarray: xr.DataArray,
# ) -> None:
#     """Test slice_range_where_values function"""

#     a_3d_dataarray.data[:, :, :] = 0
#     a_3d_dataarray.data[:, :, 1] = 107
#     vmin = 105
#     vmax = 110
#     returned_da = manipulations.slice_range_where_values(
#         a_3d_dataarray,
#         vmin=vmin,
#         vmax=vmax,
#     )
#     print(returned_da)


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
