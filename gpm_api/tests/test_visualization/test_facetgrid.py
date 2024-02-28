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
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr

from gpm_api.visualization import plot
from utils import get_test_name, save_and_check_figure


class TestPlotMap:
    """Test the plot_map function while using the facetgrid module"""

    channel = "channel"

    def expamd_dims(
        self,
        dataarray: xr.DataArray,
        size: int,
    ) -> xr.DataArray:
        """Expand dimensions of a dataarray"""

        dataarray = dataarray.expand_dims(dim={self.channel: size})
        np.random.seed(0)
        dataarray.data = np.random.rand(*dataarray.data.shape)
        return dataarray

    def test_orbit(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data"""

        orbit_dataarray = self.expamd_dims(orbit_dataarray, 6)
        p = plot.plot_map(orbit_dataarray, col=self.channel, col_wrap=3)
        save_and_check_figure(figure=p.fig, name=get_test_name())

    def test_orbit_row(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data using row argument"""

        orbit_dataarray = self.expamd_dims(orbit_dataarray, 6)
        p = plot.plot_map(orbit_dataarray, row=self.channel, col_wrap=3)
        save_and_check_figure(figure=p.fig, name=get_test_name())

    def test_orbit_one_empty_subplot(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data with one empty subplot"""

        orbit_dataarray = self.expamd_dims(orbit_dataarray, 3)
        p = plot.plot_map(orbit_dataarray, col=self.channel, col_wrap=2)
        save_and_check_figure(figure=p.fig, name=get_test_name())

    def test_orbit_extent(
        self,
        orbit_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data while specifying extent"""

        orbit_dataarray = self.expamd_dims(orbit_dataarray, 4)
        extent = [0, 20, 0, 20]
        p = plot.plot_map(orbit_dataarray, col=self.channel, col_wrap=2)
        p.remove_title_dimension_prefix()
        p.set_extent(extent)
        save_and_check_figure(figure=p.fig, name=get_test_name())

    def test_grid(
        self,
        grid_dataarray: xr.DataArray,
    ) -> None:
        """Test plotting orbit data"""

        grid_dataarray = self.expamd_dims(grid_dataarray, 4)
        p = plot.plot_map(grid_dataarray, col=self.channel, col_wrap=2)
        save_and_check_figure(figure=p.fig, name=get_test_name())
