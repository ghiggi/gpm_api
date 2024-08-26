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
import platform

import pytest
import xarray as xr

from gpm.tests.test_visualization.utils import (
    get_test_name,
    save_and_check_figure,
    skip_tests_if_no_data,
)
from gpm.visualization import orbit

# Fixtures imported from gpm.tests.conftest:
# - orbit_dataarray


pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="Minor figure differences on Windows",
)
skip_tests_if_no_data()


VARIABLE = "variable"


# Fixtures #####################################################################


@pytest.fixture
def orbit_dataset(
    orbit_dataarray: xr.DataArray,
) -> xr.Dataset:
    """Return a dataset with a single variable."""
    return xr.Dataset({VARIABLE: orbit_dataarray})


# Tests ########################################################################


def test_plot_swath(
    orbit_dataset: xr.Dataset,
) -> None:
    """Test the plot_swath function."""
    p = orbit.plot_swath(orbit_dataset)
    save_and_check_figure(figure=p.figure, name=get_test_name())


def test_plot_swath_lines(
    orbit_dataset: xr.Dataset,
) -> None:
    """Test the plot_swath_lines function."""
    p = orbit.plot_swath_lines(orbit_dataset)
    save_and_check_figure(figure=p.figure, name=get_test_name())
