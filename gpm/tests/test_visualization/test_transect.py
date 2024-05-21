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
        """Test plotting orbit data."""
        p = plot_transect(orbit_spatial_3d_dataarray.isel(along_track=0))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_along_track(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data."""
        p = plot_transect(orbit_spatial_3d_dataarray.isel(cross_track=0))
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_with_height(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data."""
        p = plot_transect(orbit_spatial_3d_dataarray.isel(along_track=0), y="height", x="lon")
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_with_nan_coordinates(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data."""
        da_transect = orbit_spatial_3d_dataarray.isel(along_track=0)
        da_transect["height"].data[0:2, 0:2] = np.nan
        p = plot_transect(da_transect, y="height", x="lon")
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_with_height_km(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data."""
        p = plot_transect(orbit_spatial_3d_dataarray.isel(along_track=0), y="height_km", x="lat")
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_with_horizontal_distance(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data."""
        p = plot_transect(orbit_spatial_3d_dataarray.isel(along_track=0), y="range", x="horizontal_distance")
        save_and_check_figure(figure=p.figure, name=get_test_name())

    def test_raise_error_with_invalid_coordinates(
        self,
        orbit_spatial_3d_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data."""
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
