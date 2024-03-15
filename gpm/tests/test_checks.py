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

from gpm import checks

# Fixtures imported from gpm.tests.conftest:
# - orbit_dataarray
# - grid_dataarray


# Utils functions ##############################################################


def make_dataset(dataarrays: list[xr.DataArray]) -> xr.Dataset:
    """Return a dataset with the given data arrays"""

    return xr.Dataset({f"variable_{i}": da for i, da in enumerate(dataarrays)})


# Fixtures #####################################################################


@pytest.fixture
def orbit_3d_dataarray(orbit_dataarray: xr.DataArray) -> xr.DataArray:
    """Return a 3D orbit data array"""

    # Add a vertical dimension with shape larger than 1 to prevent squeezing
    return orbit_dataarray.expand_dims(dim={"height": 2})


@pytest.fixture
def grid_3d_dataarray(grid_dataarray: xr.DataArray) -> xr.DataArray:
    """Return a 3D grid data array"""

    # Add a vertical dimension with shape larger than 1 to prevent squeezing
    return grid_dataarray.expand_dims(dim={"height": 2})


@pytest.fixture
def orbit_transect_dataarray(orbit_dataarray: xr.DataArray) -> xr.DataArray:
    """Return a transect orbit data array"""

    orbit_dataarray = orbit_dataarray.expand_dims(dim={"height": 2})
    return orbit_dataarray.isel(along_track=0)


@pytest.fixture
def grid_transect_dataarray(grid_dataarray: xr.DataArray) -> xr.DataArray:
    """Return a transect grid data array"""

    grid_dataarray = grid_dataarray.expand_dims(dim={"height": 2})
    return grid_dataarray.isel(lat=0)


@pytest.fixture
def orbit_dataset(orbit_dataarray: xr.DataArray) -> xr.Dataset:
    """Return an orbit dataset"""

    return make_dataset([orbit_dataarray, orbit_dataarray])


@pytest.fixture
def grid_dataset(grid_dataarray: xr.DataArray) -> xr.Dataset:
    """Return a grid dataset"""

    return make_dataset([grid_dataarray, grid_dataarray])


@pytest.fixture
def invalid_dataset(orbit_dataarray: xr.DataArray) -> xr.Dataset:
    """Return an invalid dataset"""

    return make_dataset([orbit_dataarray, xr.DataArray()])


@pytest.fixture
def orbit_3d_dataset(orbit_3d_dataarray: xr.DataArray) -> xr.Dataset:
    """Return a 3D orbit dataset"""

    return make_dataset([orbit_3d_dataarray, orbit_3d_dataarray])


@pytest.fixture
def grid_3d_dataset(grid_3d_dataarray: xr.DataArray) -> xr.Dataset:
    """Return a 3D grid dataset"""

    return make_dataset([grid_3d_dataarray, grid_3d_dataarray])


@pytest.fixture
def invalid_3d_dataset(orbit_dataarray: xr.DataArray) -> xr.Dataset:
    """Return a 3D invalid dataset"""

    return make_dataset([orbit_dataarray, xr.DataArray()])


@pytest.fixture
def orbit_transect_dataset(orbit_transect_dataarray: xr.DataArray) -> xr.Dataset:
    """Return a transect orbit dataset"""

    return make_dataset([orbit_transect_dataarray, orbit_transect_dataarray])


@pytest.fixture
def grid_transect_dataset(grid_transect_dataarray: xr.DataArray) -> xr.Dataset:
    """Return a transect grid dataset"""

    return make_dataset([grid_transect_dataarray, grid_transect_dataarray])


@pytest.fixture
def invalid_transect_dataset(orbit_dataarray: xr.DataArray) -> xr.Dataset:
    """Return a transect invalid dataset"""

    return make_dataset([orbit_dataarray, xr.DataArray()])


# Public functions #############################################################


def test_check_is_xarray() -> None:
    """Test check_is_xarray function"""

    # Should not raise exception
    checks.check_is_xarray(xr.DataArray())
    checks.check_is_xarray(xr.Dataset())

    for invalid in [None, 0, "string", [], {}]:
        with pytest.raises(TypeError):
            checks.check_is_xarray(invalid)


def test_check_is_xarray_dataarray() -> None:
    """Test check_is_xarray_dataarray function"""

    # Should not raise exception
    checks.check_is_xarray_dataarray(xr.DataArray())

    for invalid in [None, 0, "string", [], {}, xr.Dataset()]:
        with pytest.raises(TypeError):
            checks.check_is_xarray_dataarray(invalid)


def test_check_is_xarray_dataset() -> None:
    """Test check_is_xarray_dataset function"""

    # Should not raise exception
    checks.check_is_xarray_dataset(xr.Dataset())

    for invalid in [None, 0, "string", [], {}, xr.DataArray()]:
        with pytest.raises(TypeError):
            checks.check_is_xarray_dataset(invalid)


def test_get_dataset_variables() -> None:
    """Test get_dataset_variables function"""

    variables = ["variable2", "variable1"]
    variables_sorted = ["variable1", "variable2"]
    ds = xr.Dataset({v: xr.DataArray() for v in variables})

    assert checks.get_dataset_variables(ds) == variables
    assert checks.get_dataset_variables(ds, sort=True) == variables_sorted


def test_is_orbit(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test is_orbit function"""

    assert checks.is_orbit(orbit_dataarray)
    assert not checks.is_orbit(grid_dataarray)
    assert not checks.is_orbit(xr.DataArray())

    # With one dimensional longitude
    n_x = 10
    n_y = 20
    x = np.arange(n_x)
    y = np.arange(n_y)
    data = np.random.rand(n_x, n_y)
    invalid_da = xr.DataArray(data, coords={"x": x, "y": y})
    assert not checks.is_orbit(invalid_da)


def test_is_grid(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test is_grid function"""

    assert not checks.is_grid(orbit_dataarray)
    assert checks.is_grid(grid_dataarray)
    assert not checks.is_grid(xr.DataArray())


def test_check_is_orbit(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test check_is_orbit function"""

    # Should not raise exception
    checks.check_is_orbit(orbit_dataarray)

    for invalid in [grid_dataarray, xr.DataArray()]:
        with pytest.raises(ValueError):
            checks.check_is_orbit(invalid)


def test_check_is_grid(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test check_is_grid function"""

    # Should not raise exception
    checks.check_is_grid(grid_dataarray)

    for invalid in [orbit_dataarray, xr.DataArray()]:
        with pytest.raises(ValueError):
            checks.check_is_grid(invalid)


def test_check_is_gpm_object(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test check_is_gpm_object function"""

    # Should not raise exception
    checks.check_is_grid(grid_dataarray)
    checks.check_is_orbit(orbit_dataarray)

    with pytest.raises(ValueError):
        checks.check_is_gpm_object(xr.DataArray())


def test_is_spatial_2d(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
    orbit_dataset: xr.Dataset,
    grid_dataset: xr.Dataset,
    invalid_dataset: xr.Dataset,
) -> None:
    """Test is_spatial_2d function"""

    # Data arrays
    assert checks.is_spatial_2d(grid_dataarray)
    assert checks.is_spatial_2d(orbit_dataarray)
    assert not checks.is_spatial_2d(xr.DataArray())

    # Datasets
    assert checks.is_spatial_2d(grid_dataset)
    assert checks.is_spatial_2d(orbit_dataset)

    assert not checks.is_spatial_2d(invalid_dataset)


def test_is_spatial_3d(
    orbit_3d_dataarray: xr.DataArray,
    grid_3d_dataarray: xr.DataArray,
    orbit_3d_dataset: xr.Dataset,
    grid_3d_dataset: xr.Dataset,
    invalid_3d_dataset: xr.Dataset,
) -> None:
    """Test is_spatial_3d function"""

    # Data arrays
    assert checks.is_spatial_3d(grid_3d_dataarray)
    assert checks.is_spatial_3d(orbit_3d_dataarray)
    assert not checks.is_spatial_3d(xr.DataArray())

    # Datasets
    assert checks.is_spatial_3d(grid_3d_dataset)
    assert checks.is_spatial_3d(orbit_3d_dataset)
    assert not checks.is_spatial_3d(invalid_3d_dataset)


def test_is_transect(
    orbit_transect_dataarray: xr.DataArray,
    grid_transect_dataarray: xr.DataArray,
    orbit_transect_dataset: xr.Dataset,
    grid_transect_dataset: xr.Dataset,
    invalid_transect_dataset: xr.Dataset,
) -> None:
    """Test is_transect function"""

    # Data arrays
    assert checks.is_transect(grid_transect_dataarray)
    assert checks.is_transect(orbit_transect_dataarray)
    assert not checks.is_transect(xr.DataArray())

    # Datasets
    assert checks.is_transect(grid_transect_dataset)
    assert checks.is_transect(orbit_transect_dataset)
    assert not checks.is_transect(invalid_transect_dataset)


def test_check_is_spatial_2d(
    orbit_dataarray: xr.DataArray,
    orbit_dataset: xr.Dataset,
) -> None:
    """Test check_is_spatial_2d function"""

    # Should not raise exception
    checks.check_is_spatial_2d(orbit_dataarray)
    checks.check_is_spatial_2d(orbit_dataset)

    with pytest.raises(ValueError):
        checks.check_is_spatial_2d(xr.DataArray())


def test_check_is_spatial_3d(
    orbit_dataarray: xr.DataArray,
    orbit_3d_dataarray: xr.DataArray,
    orbit_3d_dataset: xr.Dataset,
) -> None:
    """Test check_is_spatial_3d function"""

    with pytest.raises(ValueError):
        checks.check_is_spatial_3d(orbit_dataarray)

    # Should not raise exception
    checks.check_is_spatial_3d(orbit_3d_dataarray)
    checks.check_is_spatial_3d(orbit_3d_dataset)


def test_check_is_transect(
    orbit_dataarray: xr.DataArray,
    orbit_transect_dataarray: xr.DataArray,
    orbit_transect_dataset: xr.Dataset,
) -> None:
    """Test check_is_transect function"""

    with pytest.raises(ValueError):
        checks.check_is_transect(orbit_dataarray)

    # Should not raise exception
    checks.check_is_transect(orbit_transect_dataarray)
    checks.check_is_transect(orbit_transect_dataset)


class TestGetVariables:
    @pytest.fixture
    def dataset_collection(
        self,
        orbit_dataarray: xr.DataArray,
        grid_dataarray: xr.DataArray,
        orbit_3d_dataarray: xr.DataArray,
        grid_3d_dataarray: xr.DataArray,
        orbit_transect_dataarray: xr.DataArray,
        grid_transect_dataarray: xr.DataArray,
    ) -> xr.Dataset:
        """Return a dataset with a variety of data arrays"""

        da_frequency = xr.DataArray(np.zeros((0, 0)), dims=["other", "radar_frequency"])

        return make_dataset(
            [
                orbit_dataarray,
                grid_dataarray,
                orbit_3d_dataarray,
                grid_3d_dataarray,
                orbit_transect_dataarray,
                grid_transect_dataarray,
                da_frequency,
                xr.DataArray(),
            ]
        )

    def test_spatial_2d(self, dataset_collection: xr.Dataset) -> None:
        """Test get_spatial_2d_variables function"""

        assert checks.get_spatial_2d_variables(dataset_collection) == ["variable_0", "variable_1"]

    def test_spatial_3d(self, dataset_collection: xr.Dataset) -> None:
        """Test get_spatial_3d_variables function"""

        assert checks.get_spatial_3d_variables(dataset_collection) == ["variable_2", "variable_3"]

    def test_transect(self, dataset_collection: xr.Dataset) -> None:
        """Test get_transect_variables function"""

        assert checks.get_transect_variables(dataset_collection) == ["variable_4", "variable_5"]

    def test_frequency(self, dataset_collection: xr.Dataset) -> None:
        """Test get_frequency_variables function"""

        assert checks.get_frequency_variables(dataset_collection) == ["variable_6"]


def test_get_vertical_dimension() -> None:
    """Test get_vertical_dimension function"""

    added_dims_list = [
        [],
        ["range"],
        ["nBnEnv"],
        ["height"],
    ]

    for added_dims in added_dims_list:
        n_dims = 1 + len(added_dims)
        da = xr.DataArray(np.zeros((0,) * n_dims), dims=["other", *added_dims])
        assert checks.get_vertical_dimension(da) == added_dims


def test_get_spatial_dimensions() -> None:
    """Test get_spatial_dimensions function"""

    added_dims_list = [
        [],
        ["along_track", "cross_track"],
        ["lat", "lon"],
        ["latitude", "longitude"],
        ["x", "y"],
    ]

    for added_dims in added_dims_list:
        n_dims = 1 + len(added_dims)
        da = xr.DataArray(np.zeros((0,) * n_dims), dims=["other", *added_dims])
        assert checks.get_spatial_dimensions(da) == added_dims


# Private functions ############################################################


def test__get_available_frequency_dims() -> None:
    """Test _get_available_frequency_dims function"""

    added_dims_list = [(), ("radar_frequency",), ("pmw_frequency",)]

    for added_dims in added_dims_list:
        n_dims = 1 + len(added_dims)
        da = xr.DataArray(np.zeros((0,) * n_dims), dims=["other", *added_dims])
        assert checks._get_available_frequency_dims(da) == added_dims


def test__is_expected_spatial_dims() -> None:
    """Test _is_expected_spatial_dims function"""

    assert checks._is_expected_spatial_dims(["y", "x"])

    # Orbit
    assert checks._is_expected_spatial_dims(["cross_track", "along_track"])

    # Grid
    assert checks._is_expected_spatial_dims(["lon", "lat"])
    assert checks._is_expected_spatial_dims(["latitude", "longitude"])

    # Invalid
    assert not checks._is_expected_spatial_dims(["other", "other"])


def test__is_spatial_2d_datarray(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test _is_spatial_2d_datarray function"""

    assert checks._is_spatial_2d_datarray(grid_dataarray, strict=True)
    assert checks._is_spatial_2d_datarray(orbit_dataarray, strict=True)

    # With extra dimension
    da = orbit_dataarray.expand_dims("extra")
    assert checks._is_spatial_2d_datarray(da, strict=False)
    assert not checks._is_spatial_2d_datarray(da, strict=True)

    # With extra vertical dimension (therefore 3D)
    da = orbit_dataarray.expand_dims("height")
    assert not checks._is_spatial_2d_datarray(da, strict=False)
    assert not checks._is_spatial_2d_datarray(da, strict=True)

    # Invalid
    assert not checks._is_spatial_2d_datarray(xr.DataArray(), strict=True)


def test__is_spatial_3d_datarray(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
    orbit_3d_dataarray: xr.DataArray,
    grid_3d_dataarray: xr.DataArray,
) -> None:
    """Test _is_spatial_3d_datarray function"""

    assert not checks._is_spatial_3d_datarray(grid_dataarray, strict=True)
    assert not checks._is_spatial_3d_datarray(orbit_dataarray, strict=True)

    # With random extra dimension
    da = orbit_dataarray.expand_dims("extra")
    assert not checks._is_spatial_3d_datarray(da, strict=False)
    assert not checks._is_spatial_3d_datarray(da, strict=True)

    # With extra vertical dimension (therefore 3D)
    assert checks._is_spatial_3d_datarray(grid_3d_dataarray, strict=False)
    assert checks._is_spatial_3d_datarray(grid_3d_dataarray, strict=True)
    assert checks._is_spatial_3d_datarray(orbit_3d_dataarray, strict=False)
    assert checks._is_spatial_3d_datarray(orbit_3d_dataarray, strict=True)

    # With vertical and random extra dimensions
    da = orbit_3d_dataarray.expand_dims("extra")
    assert checks._is_spatial_3d_datarray(da, strict=False)
    assert not checks._is_spatial_3d_datarray(da, strict=True)

    # Invalid
    assert not checks._is_spatial_3d_datarray(xr.DataArray(), strict=True)


def test__is_transect_datarray(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
    orbit_3d_dataarray: xr.DataArray,
    orbit_transect_dataarray: xr.DataArray,
    grid_transect_dataarray: xr.DataArray,
) -> None:
    """Test _is_transect_datarray function"""

    assert not checks._is_transect_datarray(grid_dataarray, strict=True)
    assert not checks._is_transect_datarray(orbit_dataarray, strict=True)

    # With only one dimension
    da = orbit_dataarray.isel(along_track=0)
    assert not checks._is_transect_datarray(da, strict=False)
    assert not checks._is_transect_datarray(da, strict=True)

    # With extra vertical dimension (therefore 3D)
    assert not checks._is_transect_datarray(orbit_3d_dataarray, strict=False)
    assert not checks._is_transect_datarray(orbit_3d_dataarray, strict=True)

    # Transect
    assert checks._is_transect_datarray(grid_transect_dataarray, strict=False)
    assert checks._is_transect_datarray(grid_transect_dataarray, strict=True)
    assert checks._is_transect_datarray(orbit_transect_dataarray, strict=False)
    assert checks._is_transect_datarray(orbit_transect_dataarray, strict=True)

    # With extra dimension
    da = orbit_transect_dataarray.expand_dims("extra")
    assert checks._is_transect_datarray(da, strict=False)
    assert not checks._is_transect_datarray(da, strict=True)

    # Invalid
    assert not checks._is_transect_datarray(xr.DataArray(), strict=True)


def test__is_spatial_2d_dataset(
    orbit_dataset: xr.Dataset,
    grid_dataset: xr.Dataset,
    invalid_dataset: xr.Dataset,
) -> None:
    """Test _is_spatial_2d_dataset function"""

    # Valid datasets
    assert checks._is_spatial_2d_dataset(grid_dataset, strict=True)
    assert checks._is_spatial_2d_dataset(orbit_dataset, strict=True)

    # Invalid dataset
    assert not checks._is_spatial_2d_dataset(invalid_dataset, strict=True)


def test__is_spatial_3d_dataset(
    grid_3d_dataset: xr.Dataset,
    orbit_3d_dataset: xr.Dataset,
    invalid_3d_dataset: xr.Dataset,
) -> None:
    """Test _is_spatial_3d_dataset function"""

    # Valid datasets
    assert checks._is_spatial_3d_dataset(grid_3d_dataset, strict=True)
    assert checks._is_spatial_3d_dataset(orbit_3d_dataset, strict=True)

    # Invalid dataset
    assert not checks._is_spatial_3d_dataset(invalid_3d_dataset, strict=True)


def test__is_transect_dataset(
    grid_transect_dataset: xr.Dataset,
    orbit_transect_dataset: xr.Dataset,
    invalid_transect_dataset: xr.Dataset,
) -> None:
    """Test _is_transect_dataset function"""

    # Valid datasets
    assert checks._is_transect_dataset(grid_transect_dataset, strict=True)
    assert checks._is_transect_dataset(orbit_transect_dataset, strict=True)

    # Invalid dataset
    assert not checks._is_transect_dataset(invalid_transect_dataset, strict=True)
