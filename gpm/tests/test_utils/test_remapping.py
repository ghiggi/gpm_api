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
"""This module tests coordinates transformations and remapping utilities."""
import dask.array as da
import numpy as np
import pyproj
import pytest
import xarray as xr

from gpm.utils.remapping import reproject_coords


class TestReprojectCoords:
    """Test class for the reproject_coords function."""

    src_crs = pyproj.Proj(proj="geocent").crs
    dst_crs = pyproj.Proj(proj="latlong").crs

    x = np.array([[6378137.0, 0]])
    y = np.array([[0, 6378137.0]])
    z = np.array([[0, 0]])

    # self = TestReprojectCoords

    @pytest.mark.parametrize("z_fixture", [None, z])
    def test_numpy_array(self, z_fixture):
        """
        Test reproject_coords with numpy arrays.

        Tests if the function works correctly with numpy arrays as input.
        The function is tested with and without providing a 'z' array.
        """
        x1, y1, z1 = reproject_coords(x=self.x, y=self.y, z=z_fixture, src_crs=self.src_crs, dst_crs=self.dst_crs)
        assert x1.shape == y1.shape == z1.shape, "Shape mismatch among x, y, z"
        assert x1.shape == (1, 2)
        np.testing.assert_allclose(x1, np.array([[0.0, 90.0]]), atol=1e-6)
        np.testing.assert_allclose(y1, np.array([[0.0, 0.0]]), atol=1e-6)
        np.testing.assert_allclose(z1, np.array([[0.0, 0.0]]), atol=1e-6)

    @pytest.mark.parametrize("z_fixture", [None, da.from_array(np.array([0, 0]), chunks=(1,))])
    def test_dask_array_1d(self, z_fixture):
        """
        Test reproject_coords with 1D dask arrays.

        Tests if the function works correctly with 1D dask arrays as input.
        """
        x = da.from_array(np.array([6378137.0, 0]), chunks=(1,))
        y = da.from_array(np.array([0, 6378137.0]), chunks=(1,))
        x1, y1, z1 = reproject_coords(x=x, y=y, z=z_fixture, src_crs=self.src_crs, dst_crs=self.dst_crs)
        assert x1.shape == (2,)
        assert x1.shape == y1.shape == z1.shape, "Shape mismatch among x, y, z"
        assert x1.chunks == ((1, 1),)
        np.testing.assert_allclose(x1.compute(), np.array([0.0, 90.0]), atol=1e-6)
        np.testing.assert_allclose(y1.compute(), np.array([0.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(z1.compute(), np.array([0.0, 0.0]), atol=1e-6)

    @pytest.mark.parametrize("z_fixture", [None, da.from_array(z)])
    def test_dask_array_2d(self, z_fixture):
        """
        Test reproject_coords with 2D dask arrays.

        Tests if the function works correctly with 2D dask arrays as input.
        """
        x_dask = da.from_array(self.x, chunks=(1, 2))
        y_dask = da.from_array(self.y, chunks=(1, 2))

        x1, y1, z1 = reproject_coords(x=x_dask, y=y_dask, z=z_fixture, src_crs=self.src_crs, dst_crs=self.dst_crs)
        assert x1.shape == y1.shape == z1.shape, "Shape mismatch among x, y, z"
        assert x1.shape == (1, 2)
        assert x1.chunks == ((1,), (2,))
        np.testing.assert_allclose(x1.compute(), np.array([[0.0, 90.0]]), atol=1e-6)
        np.testing.assert_allclose(y1.compute(), np.array([[0.0, 0.0]]), atol=1e-6)
        np.testing.assert_allclose(z1.compute(), np.array([[0.0, 0.0]]), atol=1e-6)

    @pytest.mark.parametrize("z_fixture", [None, xr.DataArray(da.from_array(z, chunks=(1, 2)), dims=["lat", "lon"])])
    def test_xarray_dask(self, z_fixture):
        """
        Test reproject_coords with xarray DataArrays using dask backend.

        Tests if the function works correctly with xarray DataArrays that use a dask backend.
        """
        x_dask = xr.DataArray(da.from_array(self.x, chunks=(1, 2)), dims=["lat", "lon"])
        y_dask = xr.DataArray(da.from_array(self.y, chunks=(1, 2)), dims=["lat", "lon"])

        x1, y1, z1 = reproject_coords(x=x_dask, y=y_dask, z=z_fixture, src_crs=self.src_crs, dst_crs=self.dst_crs)
        assert x1.shape == y1.shape == z1.shape, "Shape mismatch among x, y, z"
        assert x1.shape == (1, 2)
        assert isinstance(x1, xr.DataArray)
        assert x1.data.chunks == ((1,), (2,))

        np.testing.assert_allclose(x1.compute(), np.array([[0.0, 90.0]]), atol=1e-6)
        np.testing.assert_allclose(y1.compute(), np.array([[0.0, 0.0]]), atol=1e-6)
        np.testing.assert_allclose(z1.compute(), np.array([[0.0, 0.0]]), atol=1e-6)

    @pytest.mark.parametrize("z_fixture", [None, xr.DataArray(z, dims=["lat", "lon"])])
    def test_xarray_numpy(self, z_fixture):
        """
        Test reproject_coords with xarray DataArrays using numpy backend.

        Tests if the function works correctly with xarray DataArrays that use a numpy backend.
        """
        x_numpy = xr.DataArray(self.x, dims=["lat", "lon"])
        y_numpy = xr.DataArray(self.y, dims=["lat", "lon"])

        x1, y1, z1 = reproject_coords(x=x_numpy, y=y_numpy, z=z_fixture, src_crs=self.src_crs, dst_crs=self.dst_crs)
        assert x1.shape == y1.shape == z1.shape, "Shape mismatch among x, y, z"
        assert x1.shape == (1, 2)
        assert isinstance(x1, xr.DataArray)
        np.testing.assert_allclose(x1, np.array([[0.0, 90.0]]), atol=1e-6)
        np.testing.assert_allclose(y1, np.array([[0.0, 0.0]]), atol=1e-6)
        np.testing.assert_allclose(z1, np.array([[0.0, 0.0]]), atol=1e-6)

    def test_invalid_crs(self):
        """
        Test reproject_coords with invalid CRS types.

        Ensures that the function raises a TypeError when invalid CRS types are provided.
        """
        with pytest.raises(TypeError):
            reproject_coords(x=self.x, y=self.y, src_crs="invalid_crs", dst_crs=self.dst_crs)

        with pytest.raises(TypeError):
            reproject_coords(
                x=self.x,
                y=self.y,
                src_crs=pyproj.Proj(proj="geocent"),
                dst_crs=self.dst_crs,  # .crs required
            )

    def test_missing_crs(self):
        """Test reproject_coords with missing CRS."""
        with pytest.raises(ValueError):
            reproject_coords(x=self.x, y=self.y, dst_crs=self.dst_crs)

        with pytest.raises(ValueError):
            reproject_coords(
                x=self.x,
                y=self.y,
                src_crs=self.src_crs,
            )

    def test_shape_mismatch(self):
        """
        Test reproject_coords with mismatched input shapes.

        Ensures that the function raises a ValueError when input arrays x and y have different shapes.
        """
        y_mismatch = np.array([[0]])
        with pytest.raises(ValueError):
            reproject_coords(x=self.x, y=y_mismatch, src_crs=self.src_crs, dst_crs=self.dst_crs)

    def test_type_mismatch(self):
        """
        Test reproject_coords with different input types for x and y.

        Ensures that the function raises a TypeError when x and y are not of the same type.
        """
        y_list = [[0, 6378137.0]]
        with pytest.raises(TypeError):
            reproject_coords(x=self.x, y=y_list, src_crs=self.src_crs, dst_crs=self.dst_crs)
