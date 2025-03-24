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

from gpm.checks import (
    _is_expected_spatial_dims,
    check_has_along_track_dim,
    check_has_cross_track_dim,
    check_has_frequency_dim,
    check_has_spatial_dim,
    check_has_vertical_dim,
    check_is_cross_section,
    check_is_gpm_object,
    check_is_grid,
    check_is_orbit,
    check_is_spatial_2d,
    check_is_spatial_3d,
    get_bin_variables,
    get_cross_section_variables,
    get_frequency_dimension,
    get_frequency_variables,
    get_spatial_2d_variables,
    get_spatial_3d_variables,
    get_spatial_dimensions,
    get_vertical_dimension,
    get_vertical_variables,
    has_frequency_dim,
    has_spatial_dim,
    has_vertical_dim,
    is_cross_section,
    is_grid,
    is_orbit,
    is_spatial_2d,
    is_spatial_3d,
)
from gpm.dataset.dimensions import FREQUENCY_DIMS, SPATIAL_DIMS, VERTICAL_DIMS

# Fixtures imported from gpm.tests.conftest:
# - orbit_dataarray
# - orbit_spatial_3d_dataarray
# - orbit_cross_section_dataarray
# - grid_dataarray
# - grid_spatial_3d_dataarray
# - grid_cross_section_dataarray
# - orbit_dataset_collection
# - grid_dataset_collection


# Utils functions ##############################################################


def make_dataset(dataarrays: list[xr.DataArray]) -> xr.Dataset:
    """Return a dataset with the given data arrays."""
    return xr.Dataset({f"variable_{i}": da for i, da in enumerate(dataarrays)})


# Fixtures #####################################################################


@pytest.fixture
def orbit_dataset(orbit_dataarray: xr.DataArray) -> xr.Dataset:
    """Return an orbit dataset."""
    return make_dataset([orbit_dataarray, orbit_dataarray])


@pytest.fixture
def grid_dataset(grid_dataarray: xr.DataArray) -> xr.Dataset:
    """Return a grid dataset."""
    return make_dataset([grid_dataarray, grid_dataarray])


@pytest.fixture
def orbit_spatial_3d_dataset(orbit_spatial_3d_dataarray: xr.DataArray) -> xr.Dataset:
    """Return a 3D orbit dataset."""
    return make_dataset([orbit_spatial_3d_dataarray, orbit_spatial_3d_dataarray])


@pytest.fixture
def grid_spatial_3d_dataset(grid_spatial_3d_dataarray: xr.DataArray) -> xr.Dataset:
    """Return a 3D grid dataset."""
    return make_dataset([grid_spatial_3d_dataarray, grid_spatial_3d_dataarray])


@pytest.fixture
def orbit_cross_section_dataset(orbit_cross_section_dataarray: xr.DataArray) -> xr.Dataset:
    """Return a cross-section orbit dataset."""
    return make_dataset([orbit_cross_section_dataarray, orbit_cross_section_dataarray])


@pytest.fixture
def grid_cross_section_dataset(grid_cross_section_dataarray: xr.DataArray) -> xr.Dataset:
    """Return a cross-section grid dataset."""
    return make_dataset([grid_cross_section_dataarray, grid_cross_section_dataarray])


####-----------------------------------------------------------------------------------------------------------------.
####################
#### Dimensions ####
####################


def test_get_vertical_dimension() -> None:
    """Test get_vertical_dimension function."""
    possible_dims_list = [
        [],
        *[[dim] for dim in VERTICAL_DIMS],
    ]

    for dims in possible_dims_list:
        n_dims = 1 + len(dims)
        da = xr.DataArray(np.zeros((0,) * n_dims), dims=["other", *dims])
        assert get_vertical_dimension(da) == dims


def test_get_spatial_dimensions() -> None:
    """Test get_spatial_dimensions function."""
    possible_dims_list = [
        [],
        *SPATIAL_DIMS,
    ]

    for dims in possible_dims_list:
        n_dims = 1 + len(dims)
        da = xr.DataArray(np.zeros((0,) * n_dims), dims=["other", *dims])
        assert get_spatial_dimensions(da) == dims


def test_get_frequency_dimension() -> None:
    """Test get_frequency_dimension function."""
    possible_dims_list = [
        [],
        *[[dim] for dim in FREQUENCY_DIMS],
    ]

    for dims in possible_dims_list:
        n_dims = 1 + len(dims)
        da = xr.DataArray(np.zeros((0,) * n_dims), dims=["other", *dims])
        assert get_frequency_dimension(da) == dims


def test_has_spatial_dims(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
    orbit_cross_section_dataarray: xr.DataArray,
    grid_cross_section_dataarray: xr.DataArray,
    orbit_spatial_3d_dataarray: xr.DataArray,
    grid_spatial_3d_dataarray: xr.DataArray,
) -> None:
    """Test the has_spatial_dims function."""
    #### Check with strict=False (the default)
    assert has_spatial_dim(grid_dataarray)
    assert has_spatial_dim(orbit_dataarray)

    assert has_spatial_dim(orbit_spatial_3d_dataarray)
    assert has_spatial_dim(grid_spatial_3d_dataarray)

    assert has_spatial_dim(grid_cross_section_dataarray)
    assert has_spatial_dim(orbit_cross_section_dataarray)
    assert not has_spatial_dim(xr.DataArray())

    #### Check with strict=True
    assert has_spatial_dim(grid_dataarray, strict=True)
    assert has_spatial_dim(orbit_dataarray, strict=True)

    assert not has_spatial_dim(grid_cross_section_dataarray, strict=True)
    assert not has_spatial_dim(orbit_cross_section_dataarray, strict=True)

    assert not has_spatial_dim(orbit_spatial_3d_dataarray, strict=True)
    assert not has_spatial_dim(grid_spatial_3d_dataarray, strict=True)

    # Check no spatial dimensions case
    da_profile = orbit_spatial_3d_dataarray.isel(cross_track=0, along_track=0)
    assert not has_spatial_dim(da_profile, strict=True)
    assert not has_spatial_dim(da_profile, strict=False)

    # Check dataset "any" condition
    ds = xr.Dataset()
    ds["var"] = orbit_dataarray
    ds["empty"] = xr.DataArray()
    assert has_spatial_dim(ds)

    # Check no spatial dimension in dataset
    assert not has_spatial_dim(xr.Dataset())


def test_has_vertical_dim(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
    orbit_cross_section_dataarray: xr.DataArray,
    grid_cross_section_dataarray: xr.DataArray,
    orbit_spatial_3d_dataarray: xr.DataArray,
    grid_spatial_3d_dataarray: xr.DataArray,
) -> None:
    """Test the has_vertical_dim function."""
    #### Check with strict=False (the default)
    assert has_vertical_dim(grid_cross_section_dataarray)
    assert has_vertical_dim(orbit_cross_section_dataarray)

    assert has_vertical_dim(orbit_spatial_3d_dataarray)
    assert has_vertical_dim(grid_spatial_3d_dataarray)

    assert not has_vertical_dim(grid_dataarray)
    assert not has_vertical_dim(orbit_dataarray)

    assert not has_vertical_dim(xr.DataArray())

    #### Check with strict=True
    assert not has_vertical_dim(grid_dataarray, strict=True)
    assert not has_vertical_dim(orbit_dataarray, strict=True)

    assert not has_vertical_dim(orbit_spatial_3d_dataarray, strict=True)
    assert not has_vertical_dim(grid_spatial_3d_dataarray, strict=True)

    assert not has_vertical_dim(grid_cross_section_dataarray, strict=True)
    assert not has_vertical_dim(orbit_cross_section_dataarray, strict=True)

    # Check only vertical dimension
    da_profile = orbit_spatial_3d_dataarray.isel(cross_track=0, along_track=0)
    assert has_vertical_dim(da_profile, strict=True)
    assert has_vertical_dim(da_profile, strict=False)

    # Check dataset "any" condition
    ds = xr.Dataset()
    ds["var"] = orbit_cross_section_dataarray
    ds["empty"] = xr.DataArray()
    assert has_vertical_dim(ds)

    # Check no vertical dimension in dataset
    assert not has_vertical_dim(xr.Dataset())


@pytest.mark.parametrize("frequency_dim", FREQUENCY_DIMS)
def test_has_frequency_dim(
    frequency_dim,
    orbit_dataarray: xr.DataArray,
) -> None:
    """Test the has_frequency_dim function."""
    da = orbit_dataarray.expand_dims({frequency_dim: 2})
    da_extra = da.expand_dims({"extra": 2})

    #### Check with strict=False (the default)
    assert has_frequency_dim(da)
    assert has_frequency_dim(da_extra)

    assert not has_frequency_dim(orbit_dataarray)

    #### Check with strict=True
    assert not has_frequency_dim(da, strict=True)
    assert not has_frequency_dim(da_extra, strict=True)
    assert not has_frequency_dim(orbit_dataarray, strict=True)

    # Check only frequency dimension
    da_frequency = da.isel(cross_track=0, along_track=0)
    assert has_frequency_dim(da_frequency, strict=True)
    assert has_frequency_dim(da_frequency, strict=False)

    # Check dataset "any" condition
    ds = xr.Dataset()
    ds["var"] = da
    ds["empty"] = xr.DataArray()
    assert has_frequency_dim(ds)

    # Check no frequency dimension in dataset
    assert not has_frequency_dim(xr.Dataset())


@pytest.mark.parametrize("frequency_dim", FREQUENCY_DIMS)
def test_check_has_frequency_dim(
    frequency_dim,
    orbit_dataarray: xr.DataArray,
) -> None:
    """Test check_has_frequency_dim function."""
    da = orbit_dataarray.expand_dims({frequency_dim: 2})

    # Should not raise exception
    check_has_frequency_dim(da)
    check_has_frequency_dim(xr.Dataset({"var": da}))

    # Should raise an exception
    with pytest.raises(ValueError):
        check_has_frequency_dim(xr.DataArray())
    with pytest.raises(ValueError):
        check_has_frequency_dim(xr.Dataset())


def test_check_has_vertical_dim(
    orbit_spatial_3d_dataarray: xr.DataArray,
) -> None:
    """Test check_has_vertical_dim function."""
    # Should not raise exception
    check_has_vertical_dim(orbit_spatial_3d_dataarray)
    check_has_vertical_dim(xr.Dataset({"var": orbit_spatial_3d_dataarray}))

    # Should raise an exception
    with pytest.raises(ValueError):
        check_has_vertical_dim(xr.DataArray())
    with pytest.raises(ValueError):
        check_has_frequency_dim(xr.Dataset())


def test_check_has_spatial_dim(
    orbit_dataarray: xr.DataArray,
) -> None:
    """Test check_has_spatial_dim function."""
    # Should not raise exception
    check_has_spatial_dim(orbit_dataarray)
    check_has_spatial_dim(xr.Dataset({"var": orbit_dataarray}))

    # Should raise an exception
    with pytest.raises(ValueError):
        check_has_spatial_dim(xr.DataArray())
    with pytest.raises(ValueError):
        check_has_spatial_dim(xr.Dataset())


####-------------------------------------------------------------------------------------
#######################
#### GRID vs ORBIT ####
#######################


def test_is_expected_spatial_dims() -> None:
    """Test _is_expected_spatial_dims function."""
    assert _is_expected_spatial_dims(["y", "x"])

    # Orbit
    assert _is_expected_spatial_dims(["cross_track", "along_track"])

    # Grid
    assert _is_expected_spatial_dims(["lon", "lat"])
    assert _is_expected_spatial_dims(["latitude", "longitude"])

    # Invalid
    assert not _is_expected_spatial_dims(["other", "other"])


def test_is_orbit(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test is_orbit function."""
    assert is_orbit(orbit_dataarray)
    assert is_orbit(orbit_dataarray.isel(along_track=0))
    assert is_orbit(orbit_dataarray.isel(cross_track=0))  # nadir-view

    # Check with other dimensions names
    assert is_orbit(orbit_dataarray.rename({"lon": "longitude", "lat": "latitude"}))
    assert is_orbit(orbit_dataarray.rename({"cross_track": "y", "along_track": "x"}))
    assert is_orbit(orbit_dataarray.isel(along_track=0).rename({"cross_track": "y"}))
    assert is_orbit(orbit_dataarray.isel(cross_track=0).rename({"along_track": "x"}))

    # Check grid is not confound with orbit
    assert not is_orbit(grid_dataarray.isel(lon=0))
    assert not is_orbit(grid_dataarray.isel(lat=0))
    assert not is_orbit(xr.DataArray())

    # Check also strange edge cases
    assert not is_orbit(grid_dataarray.isel(lat=0).rename({"lon": "x"}))
    assert not is_orbit(grid_dataarray.isel(lon=0).rename({"lat": "y"}))
    assert not is_orbit(grid_dataarray.isel(lon=0).rename({"lat": "cross_track"}))
    assert not is_orbit(grid_dataarray.isel(lon=0).rename({"lat": "along_track"}))

    # With one dimensional longitude
    n_x = 10
    n_y = 20
    x = np.arange(n_x)
    y = np.arange(n_y)
    data = np.random.default_rng().random((n_x, n_y))
    invalid_da = xr.DataArray(data, coords={"x": x, "y": y})
    assert not is_orbit(invalid_da)

    # Assert without coordinates
    assert not is_orbit(grid_dataarray.drop_vars(["lon", "lat"]))
    assert not is_orbit(orbit_dataarray.drop_vars(["lon", "lat"]))


def test_is_grid(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test is_grid function."""
    assert is_grid(grid_dataarray)
    assert not is_grid(grid_dataarray.isel(lon=0))
    assert not is_grid(grid_dataarray.isel(lat=0))

    # Check with other dimensions names
    assert is_grid(grid_dataarray.rename({"lon": "longitude", "lat": "latitude"}))
    assert is_grid(grid_dataarray.rename({"lon": "x", "lat": "y"}))

    # Check orbit is not confound with grid
    assert not is_grid(orbit_dataarray)
    assert not is_grid(orbit_dataarray.isel(along_track=0))
    assert not is_grid(orbit_dataarray.isel(cross_track=0))
    assert not is_grid(xr.DataArray())

    # Assert without coordinates
    assert not is_grid(grid_dataarray.drop_vars(["lon", "lat"]))
    assert not is_grid(orbit_dataarray.drop_vars(["lon", "lat"]))


def test_check_is_orbit(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test check_is_orbit function."""
    # Should not raise exception
    check_is_orbit(orbit_dataarray)

    # Should raise an exception
    for invalid in [grid_dataarray, xr.DataArray()]:
        with pytest.raises(ValueError):
            check_is_orbit(invalid)


def test_check_is_grid(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test check_is_grid function."""
    # Should not raise exception
    check_is_grid(grid_dataarray)

    # Should raise an exception
    for invalid in [orbit_dataarray, xr.DataArray()]:
        with pytest.raises(ValueError):
            check_is_grid(invalid)


def test_check_is_gpm_object(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test check_is_gpm_object function."""
    # Should not raise exception
    check_is_grid(grid_dataarray)
    check_is_orbit(orbit_dataarray)

    # Should raise an exception
    with pytest.raises(ValueError):
        check_is_gpm_object(xr.DataArray())


def test_check_has_cross_track_dim(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test check_is_gpm_object function."""
    # Should not raise exception
    check_has_cross_track_dim(orbit_dataarray)

    # Should raise exception
    with pytest.raises(ValueError):
        check_has_cross_track_dim(grid_dataarray)

    with pytest.raises(ValueError):
        check_has_cross_track_dim(xr.DataArray())


def test_check_has_along_track_dim(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
) -> None:
    """Test check_is_gpm_object function."""
    # Should not raise exception
    check_has_along_track_dim(orbit_dataarray)

    # Should raise an exception
    with pytest.raises(ValueError):
        check_has_along_track_dim(grid_dataarray)

    with pytest.raises(ValueError):
        check_has_along_track_dim(xr.DataArray())


####-------------------------------------------------------------------------------------
#######################
#### ORBIT TYPES   ####
#######################


class TestIsSpatial2D:
    """Test the is_spatial_2d function."""

    def test_dataarray(
        self,
        orbit_dataarray: xr.DataArray,
        grid_dataarray: xr.DataArray,
        orbit_cross_section_dataarray: xr.DataArray,
        grid_cross_section_dataarray: xr.DataArray,
        orbit_spatial_3d_dataarray: xr.DataArray,
        grid_spatial_3d_dataarray: xr.DataArray,
    ) -> None:

        #### Check with strict=True (the default)
        assert is_spatial_2d(grid_dataarray)
        assert is_spatial_2d(orbit_dataarray)

        assert not is_spatial_2d(grid_cross_section_dataarray)
        assert not is_spatial_2d(orbit_cross_section_dataarray)

        assert not is_spatial_2d(orbit_spatial_3d_dataarray)
        assert not is_spatial_2d(grid_spatial_3d_dataarray)

        assert not is_spatial_2d(xr.DataArray())

        #### Check with strict=False
        assert is_spatial_2d(grid_dataarray, strict=False)
        assert is_spatial_2d(orbit_dataarray, strict=False)

        assert not is_spatial_2d(orbit_spatial_3d_dataarray, strict=False)  # vertical not allowed !
        assert not is_spatial_2d(grid_spatial_3d_dataarray, strict=False)  # vertical not allowed !

        assert not is_spatial_2d(grid_cross_section_dataarray, strict=False)
        assert not is_spatial_2d(orbit_cross_section_dataarray, strict=False)

        # Check is spatial 2D if has extra dimensions which are not "vertical"
        da_extra = orbit_dataarray.expand_dims({"extra": 2})
        assert is_spatial_2d(da_extra, strict=False)

        # Check squeeze condition
        da_nadir = orbit_dataarray.isel(along_track=[0])  # along_track size: 1
        assert not is_spatial_2d(da_nadir)
        assert not is_spatial_2d(da_nadir, strict=False)
        assert is_spatial_2d(
            da_nadir,
            squeeze=False,
        )  # with squeeze=False, along_track dim of size 1 is considered

    def test_dataset(
        self,
        orbit_dataarray: xr.DataArray,
        orbit_dataset: xr.Dataset,
        grid_dataset: xr.Dataset,
        orbit_spatial_3d_dataarray,
        orbit_spatial_3d_dataset: xr.Dataset,
        grid_spatial_3d_dataset: xr.Dataset,
        orbit_cross_section_dataset: xr.Dataset,
        grid_cross_section_dataset: xr.Dataset,
    ) -> None:

        #### Check with strict=True (the default)
        assert is_spatial_2d(grid_dataset)
        assert is_spatial_2d(orbit_dataset)

        assert not is_spatial_2d(orbit_spatial_3d_dataset)
        assert not is_spatial_2d(grid_spatial_3d_dataset)
        assert not is_spatial_2d(orbit_cross_section_dataset)
        assert not is_spatial_2d(grid_cross_section_dataset)

        assert not is_spatial_2d(xr.Dataset())

        #### Check with strict=False
        assert is_spatial_2d(grid_dataset, strict=False)
        assert is_spatial_2d(orbit_dataset, strict=False)

        assert not is_spatial_2d(orbit_spatial_3d_dataset, strict=False)  # vertical not allowed !
        assert not is_spatial_2d(grid_spatial_3d_dataset, strict=False)  # vertical not allowed !

        assert not is_spatial_2d(orbit_cross_section_dataset, strict=False)
        assert not is_spatial_2d(grid_cross_section_dataset, strict=False)

        # Check that all match the conditions
        ds = orbit_dataset
        ds["var_3d"] = orbit_dataarray.expand_dims({"extra": 2})

        assert is_spatial_2d(ds, strict=False)
        assert not is_spatial_2d(ds, strict=True)

        ds = orbit_dataset
        ds["empty_var"] = xr.DataArray()
        assert not is_spatial_2d(ds, strict=False)
        assert not is_spatial_2d(ds, strict=True)


class TestIsSpatial3D:
    """Test the is_spatial_3d function."""

    def test_dataarray(
        self,
        orbit_dataarray: xr.DataArray,
        grid_dataarray: xr.DataArray,
        orbit_cross_section_dataarray: xr.DataArray,
        grid_cross_section_dataarray: xr.DataArray,
        orbit_spatial_3d_dataarray: xr.DataArray,
        grid_spatial_3d_dataarray: xr.DataArray,
    ) -> None:

        #### Check with strict=True (the default)
        assert is_spatial_3d(grid_spatial_3d_dataarray)
        assert is_spatial_3d(orbit_spatial_3d_dataarray)

        assert not is_spatial_3d(grid_cross_section_dataarray)
        assert not is_spatial_3d(orbit_cross_section_dataarray)
        assert not is_spatial_3d(orbit_dataarray)
        assert not is_spatial_3d(grid_dataarray)
        assert not is_spatial_3d(xr.DataArray())

        #### Check with strict=False
        assert is_spatial_3d(orbit_spatial_3d_dataarray, strict=False)
        assert is_spatial_3d(grid_spatial_3d_dataarray, strict=False)

        assert not is_spatial_3d(grid_dataarray, strict=False)
        assert not is_spatial_3d(orbit_dataarray, strict=False)
        assert not is_spatial_3d(grid_cross_section_dataarray, strict=False)
        assert not is_spatial_3d(orbit_cross_section_dataarray, strict=False)

        # Check spatial 2D with extra dimension not vertical
        da = orbit_dataarray.expand_dims({"extra": 2})
        assert not is_spatial_3d(da, strict=False)
        assert not is_spatial_3d(da, strict=True)

        # Check spatial 3D with extra dimension not vertical
        da = orbit_spatial_3d_dataarray.expand_dims({"extra": 2})
        assert is_spatial_3d(da, strict=False)
        assert not is_spatial_3d(da, strict=True)

    def test_dataset(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
        orbit_dataset: xr.Dataset,
        grid_dataset: xr.Dataset,
        orbit_spatial_3d_dataset: xr.Dataset,
        grid_spatial_3d_dataset: xr.Dataset,
        orbit_cross_section_dataset: xr.Dataset,
        grid_cross_section_dataset: xr.Dataset,
    ) -> None:

        #### Check with strict=True (the default)
        assert is_spatial_3d(grid_spatial_3d_dataset)
        assert is_spatial_3d(orbit_spatial_3d_dataset)

        assert not is_spatial_3d(grid_dataset)
        assert not is_spatial_3d(orbit_dataset)
        assert not is_spatial_3d(orbit_cross_section_dataset)
        assert not is_spatial_3d(grid_cross_section_dataset)
        assert not is_spatial_3d(xr.Dataset())

        #### Check with strict=False
        assert is_spatial_3d(orbit_spatial_3d_dataset, strict=False)
        assert is_spatial_3d(grid_spatial_3d_dataset, strict=False)

        assert not is_spatial_3d(grid_dataset, strict=False)
        assert not is_spatial_3d(orbit_dataset, strict=False)
        assert not is_spatial_3d(orbit_cross_section_dataset, strict=False)
        assert not is_spatial_3d(grid_cross_section_dataset, strict=False)

        # Check that all match the conditions
        ds = orbit_spatial_3d_dataset
        ds["var_3d"] = orbit_spatial_3d_dataarray.expand_dims({"extra": 2})
        assert is_spatial_3d(ds, strict=False)
        assert not is_spatial_3d(ds, strict=True)

        ds = orbit_spatial_3d_dataset
        ds["empty_var"] = xr.DataArray()
        assert not is_spatial_3d(ds, strict=True)
        assert not is_spatial_3d(ds, strict=False)


class TestIsCrossSection:
    """Test the is_cross_section function."""

    def test_dataarray(
        self,
        orbit_dataarray: xr.DataArray,
        grid_dataarray: xr.DataArray,
        orbit_cross_section_dataarray: xr.DataArray,
        grid_cross_section_dataarray: xr.DataArray,
        orbit_spatial_3d_dataarray: xr.DataArray,
        grid_spatial_3d_dataarray: xr.DataArray,
    ) -> None:

        #### Check with strict=True (the default)
        assert is_cross_section(orbit_cross_section_dataarray)
        assert is_cross_section(grid_cross_section_dataarray)

        assert not is_cross_section(grid_spatial_3d_dataarray)
        assert not is_cross_section(orbit_spatial_3d_dataarray)
        assert not is_cross_section(orbit_dataarray)
        assert not is_cross_section(grid_dataarray)
        assert not is_cross_section(xr.DataArray())

        #### Check with strict=False
        assert is_cross_section(grid_cross_section_dataarray, strict=False)
        assert is_cross_section(orbit_cross_section_dataarray, strict=False)

        assert not is_cross_section(grid_dataarray, strict=False)
        assert not is_cross_section(orbit_dataarray, strict=False)
        assert not is_cross_section(grid_spatial_3d_dataarray, strict=False)
        assert not is_cross_section(orbit_spatial_3d_dataarray, strict=False)

        # Check cross-section with extra dimension not spatial
        da = orbit_cross_section_dataarray.expand_dims({"extra": 2})
        assert is_cross_section(da, strict=False)
        assert not is_cross_section(da, strict=True)

        # Check when spatial or vertical dim is not present
        da = orbit_cross_section_dataarray.isel(cross_track=0)
        assert not is_cross_section(da, strict=False)
        assert not is_cross_section(da, strict=True)

        da = orbit_cross_section_dataarray.isel(range=0)
        assert not is_cross_section(da, strict=False)
        assert not is_cross_section(da, strict=True)

    def test_dataset(
        self,
        orbit_cross_section_dataarray: xr.DataArray,
        orbit_dataset: xr.Dataset,
        grid_dataset: xr.Dataset,
        orbit_spatial_3d_dataarray,
        orbit_spatial_3d_dataset: xr.Dataset,
        grid_spatial_3d_dataset: xr.Dataset,
        orbit_cross_section_dataset: xr.Dataset,
        grid_cross_section_dataset: xr.Dataset,
    ) -> None:

        #### Check with strict=True (the default)
        assert is_cross_section(grid_cross_section_dataset)
        assert is_cross_section(orbit_cross_section_dataset)

        assert not is_cross_section(grid_dataset)
        assert not is_cross_section(orbit_dataset)
        assert not is_cross_section(orbit_spatial_3d_dataset)
        assert not is_cross_section(grid_spatial_3d_dataset)
        assert not is_cross_section(xr.Dataset())

        #### Check with strict=False
        assert is_cross_section(orbit_cross_section_dataset, strict=False)
        assert is_cross_section(grid_cross_section_dataset, strict=False)

        # - Check the extra dimensions is not another spatial dimension
        assert not is_cross_section(grid_dataset, strict=False)
        assert not is_cross_section(orbit_dataset, strict=False)
        assert not is_cross_section(orbit_spatial_3d_dataset, strict=False)
        assert not is_cross_section(grid_spatial_3d_dataset, strict=False)

        # Check that all match the conditions
        ds = orbit_cross_section_dataset
        ds["var_3d"] = orbit_cross_section_dataarray.expand_dims({"extra": 2})
        assert is_cross_section(ds, strict=False)
        assert not is_cross_section(ds, strict=True)

        ds = orbit_cross_section_dataset
        ds["empty_var"] = xr.DataArray()
        assert not is_cross_section(ds, strict=True)
        assert not is_cross_section(ds, strict=False)


def test_check_is_spatial_2d(
    orbit_dataarray: xr.DataArray,
    orbit_dataset: xr.Dataset,
) -> None:
    """Test check_is_spatial_2d function."""
    # Should not raise exception
    check_is_spatial_2d(orbit_dataarray)
    check_is_spatial_2d(orbit_dataset)

    # Should raise an exception
    with pytest.raises(ValueError):
        check_is_spatial_2d(xr.DataArray())


def test_check_is_spatial_3d(
    orbit_dataarray: xr.DataArray,
    orbit_spatial_3d_dataarray: xr.DataArray,
    orbit_spatial_3d_dataset: xr.Dataset,
) -> None:
    """Test check_is_spatial_3d function."""
    # Should raise an exception
    with pytest.raises(ValueError):
        check_is_spatial_3d(orbit_dataarray)

    # Should not raise exception
    check_is_spatial_3d(orbit_spatial_3d_dataarray)
    check_is_spatial_3d(orbit_spatial_3d_dataset)


def test_check_is_cross_section(
    orbit_dataarray: xr.DataArray,
    orbit_cross_section_dataarray: xr.DataArray,
    orbit_cross_section_dataset: xr.Dataset,
) -> None:
    """Test check_is_cross_section function."""
    # Should raise an exception
    with pytest.raises(ValueError):
        check_is_cross_section(orbit_dataarray)

    # Should not raise exception
    check_is_cross_section(orbit_cross_section_dataarray)
    check_is_cross_section(orbit_cross_section_dataset)


####-----------------------------------------------------------------------------------------------------------------.
###############################
#### Variables information ####
###############################


class TestGetVariables:
    def test_spatial_2d(self, orbit_dataset_collection: xr.Dataset, grid_dataset_collection: xr.Dataset) -> None:
        """Test get_spatial_2d_variables function."""
        # ORBIT
        assert get_spatial_2d_variables(orbit_dataset_collection) == ["bin_variable", "variableBin", "variable_2d"]
        # GRID
        assert get_spatial_2d_variables(grid_dataset_collection) == ["variable_2d"]

    def test_spatial_3d(self, orbit_dataset_collection: xr.Dataset, grid_dataset_collection: xr.Dataset) -> None:
        """Test get_spatial_3d_variables function."""
        # ORBIT
        assert get_spatial_3d_variables(orbit_dataset_collection) == ["variable_3d"]
        # GRID
        assert get_spatial_3d_variables(grid_dataset_collection) == ["variable_3d"]

    def test_cross_section(self, orbit_dataset_collection: xr.Dataset, grid_dataset_collection: xr.Dataset) -> None:
        """Test get_cross_section_variables function."""
        # ORBIT
        assert get_cross_section_variables(orbit_dataset_collection) == []
        assert get_cross_section_variables(orbit_dataset_collection.isel(along_track=0)) == ["variable_3d"]
        assert get_cross_section_variables(orbit_dataset_collection.isel(cross_track=0)) == ["variable_3d"]
        # GRID
        assert get_cross_section_variables(grid_dataset_collection) == []
        assert get_cross_section_variables(grid_dataset_collection.isel(lat=0)) == ["variable_3d"]
        assert get_cross_section_variables(grid_dataset_collection.isel(lon=0)) == ["variable_3d"]

    def test_frequency_variable(
        self,
        orbit_dataset_collection: xr.Dataset,
        grid_dataset_collection: xr.Dataset,
    ) -> None:
        """Test get_frequency_variables function."""
        # ORBIT
        assert get_frequency_variables(orbit_dataset_collection) == ["variable_frequency"]
        # GRID
        assert get_frequency_variables(grid_dataset_collection) == ["variable_frequency"]

    def test_vertical_variables(
        self,
        orbit_dataset_collection: xr.Dataset,
        grid_dataset_collection: xr.Dataset,
    ) -> None:
        """Test get_vertical_variables function."""
        # ORBIT
        assert get_vertical_variables(orbit_dataset_collection) == ["variable_3d"]
        # GRID
        assert get_vertical_variables(grid_dataset_collection) == ["variable_3d"]

    def test_bin_variables(self, orbit_dataset_collection: xr.Dataset) -> None:
        """Test get_bin_variables function."""
        # ORBIT
        assert get_bin_variables(orbit_dataset_collection) == ["bin_variable", "variableBin"]
