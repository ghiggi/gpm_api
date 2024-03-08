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


# Private functions ############################################################


def test__get_available_frequency_dims() -> None:
    """Test _get_available_frequency_dims function"""

    da = xr.DataArray(np.zeros((0,)), dims=["other"])
    assert checks._get_available_frequency_dims(da) == ()

    da = xr.DataArray(np.zeros((0, 0)), dims=["other", "radar_frequency"])
    assert checks._get_available_frequency_dims(da) == ("radar_frequency",)

    da = xr.DataArray(np.zeros((0, 0)), dims=["other", "pmw_frequency"])
    assert checks._get_available_frequency_dims(da) == ("pmw_frequency",)


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
