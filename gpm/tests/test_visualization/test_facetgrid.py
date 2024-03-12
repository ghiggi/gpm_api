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
import xarray as xr

from gpm.visualization import plot
from gpm.tests.test_visualization.utils import (
    expand_dims,
    get_test_name,
    save_and_check_figure,
)


EXTRA_DIM = "auxiliary_dim"
EXTRA_DIM_2 = "auxiliary_dim_2"


# Fixtures ####################################################################


@pytest.fixture
def orbit_dataarray_4_frames(
    orbit_dataarray: xr.DataArray,
) -> xr.DataArray:
    """Return a dataarray with one extra dimension of shape 4"""

    return expand_dims(orbit_dataarray, 4, dim=EXTRA_DIM)


@pytest.fixture
def orbit_dataarray_6_frames(
    orbit_dataarray: xr.DataArray,
) -> xr.DataArray:
    """Return a dataarray with one extra dimension of shape 6"""

    return expand_dims(orbit_dataarray, 6, dim=EXTRA_DIM)


@pytest.fixture
def orbit_dataarray_2x2_frames(
    orbit_dataarray: xr.DataArray,
) -> xr.DataArray:
    """Return a dataarray with two extra dimensions of shape 2"""

    orbit_dataarray = expand_dims(orbit_dataarray, 2, dim=EXTRA_DIM)
    orbit_dataarray = expand_dims(orbit_dataarray, 2, dim=EXTRA_DIM_2)
    return orbit_dataarray


@pytest.fixture
def grid_dataarray_4_frames(
    grid_dataarray: xr.DataArray,
) -> xr.DataArray:
    """Return a dataarray with one extra dimension of shape 4"""

    return expand_dims(grid_dataarray, 4, dim=EXTRA_DIM)


# Tests #######################################################################


class TestPlotMap:
    """Test the plot_map function while using the facetgrid module"""

    def test_orbit(
        self,
        orbit_dataarray_6_frames: xr.DataArray,
    ) -> None:
        """Test plotting orbit data using col argument"""

        p = plot.plot_map(orbit_dataarray_6_frames, col=EXTRA_DIM, col_wrap=3)
        p.set_title("Title")
        save_and_check_figure(figure=p.fig, name=get_test_name())

    def test_orbit_row(
        self,
        orbit_dataarray_6_frames: xr.DataArray,
    ) -> None:
        """Test plotting orbit data using row argument"""

        p = plot.plot_map(orbit_dataarray_6_frames, row=EXTRA_DIM, col_wrap=3)
        save_and_check_figure(figure=p.fig, name=get_test_name())

    def test_orbit_col_row(
        self,
        orbit_dataarray_2x2_frames: xr.DataArray,
    ) -> None:
        """Test plotting orbit data using row and col arguments"""

        # Calling with call_wrap=1, which should be ignored
        p = plot.plot_map(orbit_dataarray_2x2_frames, col=EXTRA_DIM, row=EXTRA_DIM_2)
        save_and_check_figure(figure=p.fig, name=get_test_name())

    def test_orbit_no_col_row(
        self,
        orbit_dataarray_4_frames: xr.DataArray,
    ) -> None:
        """Test plotting orbit data without valid col or row argument"""

        with pytest.raises(ValueError):
            plot.plot_map(orbit_dataarray_4_frames, col=None)

    def test_orbit_one_empty_subplot(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with one empty subplot"""

        orbit_dataarray = expand_dims(orbit_dataarray, 3, dim=EXTRA_DIM)
        p = plot.plot_map(orbit_dataarray, col=EXTRA_DIM, col_wrap=2)
        save_and_check_figure(figure=p.fig, name=get_test_name())

    def test_orbit_extent(
        self,
        orbit_dataarray_4_frames: xr.DataArray,
    ) -> None:
        """Test plotting orbit data while specifying extent"""

        extent = [0, 20, 0, 20]
        p = plot.plot_map(
            orbit_dataarray_4_frames, col=EXTRA_DIM, col_wrap=2, optimize_layout=False
        )
        p.remove_title_dimension_prefix()
        p.set_extent(extent)
        save_and_check_figure(figure=p.fig, name=get_test_name())

    def test_orbit_non_unique_index(
        self,
        orbit_dataarray_4_frames: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with non-unique channel index"""

        orbit_dataarray_4_frames[EXTRA_DIM] = [0, 1, 1, 2]
        with pytest.raises(ValueError):
            plot.plot_map(orbit_dataarray_4_frames, col=EXTRA_DIM, col_wrap=2)

    def test_orbit_no_colorbar(
        self,
        orbit_dataarray_4_frames: xr.DataArray,
    ) -> None:
        """Test plotting orbit data without colorbar"""

        p = plot.plot_map(orbit_dataarray_4_frames, col=EXTRA_DIM, col_wrap=2, add_colorbar=False)
        save_and_check_figure(figure=p.fig, name=get_test_name())

    def test_orbit_horizontal_colorbar(
        self,
        orbit_dataarray_4_frames: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with a horizontal colorbar"""

        cbar_kwargs = {"orientation": "horizontal"}
        p = plot.plot_map(
            orbit_dataarray_4_frames, col=EXTRA_DIM, col_wrap=2, cbar_kwargs=cbar_kwargs
        )
        save_and_check_figure(figure=p.fig, name=get_test_name())

    def test_orbit_colorbar_ticks(
        self,
        orbit_dataarray_4_frames: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with custom colorbar tick labels"""

        cbar_kwargs = {"ticks": [0.1, 0.2, 0.4, 0.6, 0.8], "ticklabels": [42.5, 43, 44, 45, 46]}
        p = plot.plot_map(
            orbit_dataarray_4_frames, col=EXTRA_DIM, col_wrap=2, cbar_kwargs=cbar_kwargs
        )
        save_and_check_figure(figure=p.fig, name=get_test_name())

    def test_orbit_extended_colorbar(
        self,
        orbit_dataarray_4_frames: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with extended colorbar"""

        cbar_kwargs = {"extend": "both", "extendfrac": 0.03}
        p = plot.plot_map(
            orbit_dataarray_4_frames, col=EXTRA_DIM, col_wrap=2, cbar_kwargs=cbar_kwargs
        )
        save_and_check_figure(figure=p.fig, name=get_test_name())

    def test_grid(
        self,
        grid_dataarray_4_frames: xr.DataArray,
    ) -> None:
        """Test plotting orbit data"""

        p = plot.plot_map(grid_dataarray_4_frames, col=EXTRA_DIM, col_wrap=2)
        save_and_check_figure(figure=p.fig, name=get_test_name())


class TestPlotImage:
    """Test the plot_image function while using the facetgrid module"""

    def test_orbit(
        self,
        orbit_dataarray_4_frames: xr.DataArray,
    ) -> None:
        """Test plotting orbit data"""

        p = plot.plot_image(orbit_dataarray_4_frames, col=EXTRA_DIM, col_wrap=2)
        save_and_check_figure(figure=p.fig, name=get_test_name())

    def test_orbit_high_aspect_ratio(
        self,
        orbit_dataarray_4_frames: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with high aspect ratio (that gets corrected)"""

        p = plot.plot_image(orbit_dataarray_4_frames, col=EXTRA_DIM, col_wrap=2, facet_aspect=2)
        save_and_check_figure(figure=p.fig, name=get_test_name())

    def test_grid(
        self,
        grid_dataarray_4_frames: xr.DataArray,
    ) -> None:
        """Test plotting orbit data"""

        p = plot.plot_image(grid_dataarray_4_frames, col=EXTRA_DIM, col_wrap=2)
        save_and_check_figure(figure=p.fig, name=get_test_name())
