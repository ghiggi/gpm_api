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
"""This module test the visualization transect utilities."""

import platform

import numpy as np
import pytest
import xarray as xr

from gpm.tests.test_visualization.utils import (
    get_test_name,
    save_and_check_figure,
    skip_tests_if_no_data,
)
from gpm.tests.utils.fake_datasets import get_orbit_dataarray
from gpm.visualization.profile import plot_transect, plot_transect_line


@pytest.fixture()
def orbit_spatial_3d_dataarray() -> xr.DataArray:
    """Return a 3D orbit dataset."""
    orbit_dataarray_3d = get_orbit_dataarray(
        start_lon=0,
        start_lat=0,
        end_lon=20,
        end_lat=15,
        width=1e6,
        n_along_track=20,
        n_cross_track=49,  # imitate 2A-DPR
        n_range=4,
    )
    return orbit_dataarray_3d


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Minor figure differences on Windows",
)
skip_tests_if_no_data()


class TestPlotTransect:
    """Test the plot_transect function."""

    def test_cross_track(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting cross-track transect."""
        p = plot_transect(orbit_spatial_3d_dataarray.isel(along_track=0))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_along_track(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting along-track transect."""
        p = plot_transect(orbit_spatial_3d_dataarray.isel(cross_track=0))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_with_height(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting transect with height on y axis."""
        p = plot_transect(orbit_spatial_3d_dataarray.isel(along_track=0), y="height", x="lon")
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_with_alpha_array(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting transect with alpha array."""
        da = orbit_spatial_3d_dataarray.isel(along_track=0)
        alpha = np.ones(da.shape) * 0.5
        p = plot_transect(da, y="height", x="lon", alpha=alpha)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_with_rgb_imshow(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting RGB transect (with imshow)."""
        da_rgb_3d = orbit_spatial_3d_dataarray.expand_dims({"rgb": 3}).transpose(..., "rgb")
        da_rgb = da_rgb_3d.isel(along_track=0)
        p = plot_transect(da_rgb, y="range", rgb="rgb")
        save_and_check_figure(figure=p.figure, name=get_test_name())

    # TODO: implement RGB for pcolormesh
    # TODO: if rgb=True, 1D-coord not available as x/y plot !
    # --> _infer_xy_labels_3d, _infer_xy_labels

    # def test_with_rgb_pcolormesh(
    #     self,
    #     orbit_spatial_3d_dataarray: xr.DataArray,
    # ) -> None:
    #     """Test plotting RGB transect (with pcolormesh)."""
    #     orbit_spatial_3d_dataarray = orbit_spatial_3d_dataarray.expand_dims({"rgb": 3}).transpose(...,"rgb")
    #     p = plot_transect(orbit_spatial_3d_dataarray.isel(along_track=0), y="height", x="lon")
    #     save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_with_nan_coordinates(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting transect with nan coordinates."""
        da_transect = orbit_spatial_3d_dataarray.isel(along_track=0)
        da_transect["height"].data[0:2, 0:2] = np.nan
        p = plot_transect(da_transect, y="height", x="lon")
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_with_nan_coordinates_alpha_array(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting transect with nan coordinates and alpha array."""
        da_transect = orbit_spatial_3d_dataarray.isel(along_track=0)
        da_transect["height"].data[0:2, 0:2] = np.nan
        alpha = np.ones(da_transect.shape) * 0.5
        p = plot_transect(da_transect, y="height", x="lon", alpha=alpha)
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_with_height_km(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting transect with height in kilometers on y axis."""
        p = plot_transect(orbit_spatial_3d_dataarray.isel(along_track=0), y="height_km", x="lat")
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_with_horizontal_distance(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting cross-track transect with horizontal distance on x axis."""
        p = plot_transect(orbit_spatial_3d_dataarray.isel(along_track=0), y="range", x="horizontal_distance")
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_raise_error_with_invalid_coordinates(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test invalid y axis options."""
        # Not existing y coordinate
        with pytest.raises(ValueError):
            plot_transect(orbit_spatial_3d_dataarray.isel(along_track=0), y="inexisting")

        # No pre-computed y coordinate
        with pytest.raises(ValueError):
            plot_transect(orbit_spatial_3d_dataarray.isel(along_track=0), y="range_distance_from_satellite_km")

        # No pre-computed y coordinate
        with pytest.raises(ValueError):
            plot_transect(orbit_spatial_3d_dataarray.isel(along_track=0), x="inexisting")


def test_plot_transect_line(
    orbit_spatial_3d_dataarray: xr.DataArray,
):
    p = plot_transect_line(orbit_spatial_3d_dataarray.isel(along_track=10), color="darkblue", add_direction=True)
    save_and_check_figure(figure=p.figure, name=get_test_name())
