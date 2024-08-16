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
"""This module test the subsetting utilities."""

import numpy as np
import pytest
import xarray as xr

from gpm.tests.utils.fake_datasets import get_orbit_dataarray
from gpm.utils.subsetting import align_along_track, align_cross_track


class TestSel:
    """Test gpm.sel method."""

    def test_gpm_id(self):
        """Test gpm.sel with gpm_id coordinate."""
        n_along_track = 10
        n_cross_track = 5
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=20,
            end_lat=15,
            width=1e6,
            n_along_track=n_along_track,
            n_cross_track=n_cross_track,
        )
        da.data[:, 0:10] = np.arange(0, 10)

        # Test subsetting with gpm_id unordered
        da_subset = da.gpm.sel(gpm_id=["0-2", "0-5", "0-1"])
        assert da_subset["gpm_id"].data.tolist() == ["0-2", "0-5", "0-1"]
        np.testing.assert_allclose(da_subset.isel(cross_track=0).data, [2, 5, 1])

        # Test subsetting by gpm_id DataArray
        da_sel = da.isel(along_track=slice(0, 3))["gpm_id"]
        da_subset = da.gpm.sel(gpm_id=da_sel)
        assert da_subset["gpm_id"].data.tolist() == ["0-0", "0-1", "0-2"]
        np.testing.assert_allclose(da_subset.isel(cross_track=0).data, [0, 1, 2])

        # Test subsetting by gpm_id slice (stop is inclusive !)
        da_subset = da.gpm.sel(gpm_id=slice("0-2", "0-4"))
        assert da_subset["gpm_id"].data.tolist() == ["0-2", "0-3", "0-4"]
        np.testing.assert_allclose(da_subset.isel(cross_track=0).data, [2, 3, 4])

        # Test subsetting by non-existing gpm_ids
        # - No error raised with slice
        assert da.gpm.sel(gpm_id=slice("1-2", "1-4")).size == 0  # no error
        # - Error is is raised with single or list values
        with pytest.raises(KeyError):
            da.gpm.sel(gpm_id=["1-2", "1-4"])
        with pytest.raises(KeyError):
            da.gpm.sel(gpm_id=["0-2", "1-4"])
        with pytest.raises(KeyError):
            da.gpm.sel(gpm_id="1-4")

    def test_time(self):
        """Test gpm.sel with time coordinate."""
        n_along_track = 10
        n_cross_track = 5
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=20,
            end_lat=15,
            width=1e6,
            n_along_track=n_along_track,
            n_cross_track=n_cross_track,
        )
        da.data[:, 0:10] = np.arange(0, 10)
        # da["time"]

        # Test subsetting with time unordered (as string)
        time_subset = ["2000-01-01T00:00:06", "2000-01-01T00:00:04"]
        da_subset = da.gpm.sel(time=["2000-01-01T00:00:06", "2000-01-01T00:00:04"])
        np.testing.assert_allclose(da_subset["time"].data.tolist(), np.array(time_subset, dtype="M8[ns]").tolist())
        np.testing.assert_allclose(da_subset.isel(cross_track=0).data, [6, 4])

        # Test subsetting with time as np.datetime 64
        da_subset = da.gpm.sel(time=np.array(time_subset, dtype="M8[ns]"))
        np.testing.assert_allclose(da_subset["time"].data.tolist(), np.array(time_subset, dtype="M8[ns]").tolist())
        np.testing.assert_allclose(da_subset.isel(cross_track=0).data, [6, 4])

        # Test subsetting by time DataArray
        da_sel = da.isel(along_track=slice(0, 3))["time"]
        da_subset = da.gpm.sel(time=da_sel)
        np.testing.assert_allclose(da_subset.isel(cross_track=0).data, [0, 1, 2])

        # Test subsetting by time slice (stop is inclusive !)
        da_subset = da.gpm.sel(time=slice("2000-01-01T00:00:04", "2000-01-01T00:00:06"))
        np.testing.assert_allclose(da_subset.isel(cross_track=0).data, [4, 5, 6])

    def test_inexisting_coordinate(self):
        """Test gpm.sel with an inexisting coordinate."""
        n_along_track = 10
        n_cross_track = 5
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=20,
            end_lat=15,
            width=1e6,
            n_along_track=n_along_track,
            n_cross_track=n_cross_track,
        )
        with pytest.raises(ValueError):
            da.gpm.sel(inexisting=slice(0, 2))

    def test_dimension_not_coordinate(self):
        """Test gpm.sel with a dimension without coordinate."""
        n_along_track = 10
        n_cross_track = 5
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=20,
            end_lat=15,
            width=1e6,
            n_along_track=n_along_track,
            n_cross_track=n_cross_track,
        )
        with pytest.raises(ValueError) as excinfo:
            da.gpm.sel(along_track=slice(0, 2))
        assert "Can not subset with gpm.sel the dimension 'along_track' if it is not also a coordinate." in str(
            excinfo.value,
        )

    def test_dimension_coordinate(self):
        """Test gpm.sel with a dimension coordinate."""
        n_along_track = 10
        n_cross_track = 5
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=20,
            end_lat=15,
            width=1e6,
            n_along_track=n_along_track,
            n_cross_track=n_cross_track,
        )
        da.data[:, 0:10] = np.arange(0, 10)
        da = da.swap_dims({"along_track": "gpm_along_track_id"})
        da_subset = da.gpm.sel(gpm_along_track_id=slice(0, 2))
        np.testing.assert_allclose(da_subset.isel(cross_track=0).data, [0, 1, 2])

    def test_numerical_coordinate(self):
        """Test gpm.sel with a classical numerical coordinate."""
        n_along_track = 10
        n_cross_track = 5
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=20,
            end_lat=15,
            width=1e6,
            n_along_track=n_along_track,
            n_cross_track=n_cross_track,
        )
        da.data[:, 0:10] = np.arange(0, 10)
        da_subset = da.gpm.sel(gpm_along_track_id=slice(0, 2))
        np.testing.assert_allclose(da_subset.isel(cross_track=0).data, [0, 1, 2])

    def test_multiple_indexers_error(self):
        """Test gpm.sel with coordinates pointing to same dimension."""
        n_along_track = 10
        n_cross_track = 5
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=20,
            end_lat=15,
            width=1e6,
            n_along_track=n_along_track,
            n_cross_track=n_cross_track,
        )
        da.data[:, 0:10] = np.arange(0, 10)
        with pytest.raises(ValueError) as excinfo:
            da.gpm.sel(gpm_along_track_id=slice(0, 2), gpm_id=slice("0-0", "0-3"))
        assert "Multiple indexers point to the 'along_track' dimension." in str(excinfo.value)

    def test_multi_subsetting(self):
        """Test gpm.sel with multiple coordinates."""
        n_along_track = 10
        n_cross_track = 5
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=20,
            end_lat=15,
            width=1e6,
            n_along_track=n_along_track,
            n_cross_track=n_cross_track,
        )
        da.data = np.arange(0, 50).reshape(5, 10)
        da_subset = da.gpm.sel(gpm_along_track_id=slice(0, 2), gpm_cross_track_id=slice(0, 1))
        assert da_subset.shape == (2, 3)
        np.testing.assert_allclose(da_subset.isel(cross_track=0).data, [0, 1, 2])
        np.testing.assert_allclose(da_subset.isel(cross_track=1).data, [10, 11, 12])


class TestIsel:
    """Test gpm.isel method."""

    def test_with_1d_coordinate(self):
        """Test gpm.isel with a classical 1D coordinate."""
        n_along_track = 10
        n_cross_track = 5
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=20,
            end_lat=15,
            width=1e6,
            n_along_track=n_along_track,
            n_cross_track=n_cross_track,
        )
        da.data[:, 0:10] = np.arange(0, 10)

        # Slice
        da_subset = da.gpm.isel(gpm_id=slice(0, 2))
        np.testing.assert_allclose(da_subset.isel(cross_track=0).data, [0, 1])
        # List
        da_subset = da.gpm.isel(gpm_id=[3, 2, -1])
        np.testing.assert_allclose(da_subset.isel(cross_track=0).data, [3, 2, 9])
        # Value
        da_subset = da.gpm.isel(gpm_id=-1)
        np.testing.assert_allclose(da_subset.isel(cross_track=0).data, [9])
        # Out of index
        with pytest.raises(IndexError):
            da.gpm.isel(gpm_id=10)

    def test_with_2d_coordinate(self):
        """Test gpm.isel with a 2D coordinate raise an error."""
        n_along_track = 10
        n_cross_track = 5
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=20,
            end_lat=15,
            width=1e6,
            n_along_track=n_along_track,
            n_cross_track=n_cross_track,
        )
        da.data[:, 0:10] = np.arange(0, 10)

        with pytest.raises(ValueError) as excinfo:
            da.gpm.isel(lon=slice(0, 2))
        assert "'lon' is not a 1D non-dimensional coordinate." in str(excinfo.value)

    def test_with_dimension(self):
        """Test gpm.isel with a dimension."""
        n_along_track = 10
        n_cross_track = 5
        da = get_orbit_dataarray(
            start_lon=0,
            start_lat=0,
            end_lon=20,
            end_lat=15,
            width=1e6,
            n_along_track=n_along_track,
            n_cross_track=n_cross_track,
        )
        da.data[:, 0:10] = np.arange(0, 10)

        da_subset = da.gpm.isel(along_track=slice(0, 2))
        np.testing.assert_allclose(da_subset.isel(cross_track=0).data, [0, 1])


def test_align_along_track():
    """Test gpm.align_along_track function."""
    n_along_track = 10
    n_cross_track = 5
    da = get_orbit_dataarray(
        start_lon=0,
        start_lat=0,
        end_lon=20,
        end_lat=15,
        width=1e6,
        n_along_track=n_along_track,
        n_cross_track=n_cross_track,
    )
    da.data[:, 0:10] = np.arange(0, 10)
    ds = da.to_dataset(name="var")

    ds_subset = ds.isel(along_track=slice(6, 10))
    da_subset = da.isel(along_track=slice(5, 8))

    # Alignment occurs using gpm-id !
    list_objs = align_along_track(da_subset, da, ds, ds_subset)
    # Return all objects
    assert len(list_objs) == 4
    assert isinstance(list_objs[0], xr.DataArray)
    assert isinstance(list_objs[2], xr.Dataset)

    # Check all same gpm_id
    for xr_obj in list_objs:
        assert xr_obj["gpm_id"].data.tolist() == ["0-6", "0-7"]

    # Check case when no intersection
    ds_subset = ds.isel(along_track=slice(0, 3))
    da_subset = da.isel(along_track=slice(7, 10))
    with pytest.raises(ValueError) as excinfo:
        list_objs = align_along_track(da_subset, da, ds, ds_subset)
    assert "No common gpm_id." in str(excinfo.value)

    # Check case when gpm_id is not present
    ds_subset = ds.drop_vars("gpm_id")
    da_subset = da.drop_vars("gpm_id")
    with pytest.raises(ValueError) as excinfo:
        list_objs = align_along_track(da_subset, da, ds, ds_subset)
    assert "The xarray objects does not have the 'gpm_id' coordinate. Impossible to align." in str(excinfo.value)


def test_align_cross_track():
    """Test gpm.align_cross_track function."""
    n_along_track = 5
    n_cross_track = 10
    da = get_orbit_dataarray(
        start_lon=0,
        start_lat=0,
        end_lon=20,
        end_lat=15,
        width=1e6,
        n_along_track=n_along_track,
        n_cross_track=n_cross_track,
    )
    da.data[0:10, :] = np.repeat(np.arange(0, n_cross_track), n_along_track).reshape(n_cross_track, n_along_track)
    ds = da.to_dataset(name="var")

    ds_subset = ds.isel(cross_track=[7, 6])  # unordered to check it sorts !
    da_subset = da.isel(cross_track=[8, 7, 6, 5])  # unordered to check it sorts !

    # Alignment occurs using gpm-id !
    list_objs = align_cross_track(da_subset, da, ds, ds_subset)
    # Return all objects
    assert len(list_objs) == 4
    assert isinstance(list_objs[0], xr.DataArray)
    assert isinstance(list_objs[2], xr.Dataset)

    # Check all same gpm_cross_track_id (sorted !)
    for xr_obj in list_objs:
        assert xr_obj["gpm_cross_track_id"].data.tolist() == [6, 7]

    # Check case when empty
    ds_subset = ds.isel(cross_track=slice(0, 3))
    da_subset = da.isel(cross_track=slice(7, 10))
    with pytest.raises(ValueError) as excinfo:
        list_objs = align_cross_track(da_subset, da, ds, ds_subset)
    assert "No common gpm_cross_track_id." in str(excinfo.value)

    # Check case when gpm_id is not present
    ds_subset = ds.drop_vars("gpm_cross_track_id")
    da_subset = da.drop_vars("gpm_cross_track_id")
    with pytest.raises(ValueError) as excinfo:
        list_objs = align_cross_track(da_subset, da, ds, ds_subset)
    assert "The xarray objects does not have the 'gpm_cross_track_id' coordinate. Impossible to align." in str(
        excinfo.value,
    )
