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
"""This module provide utility functions used in the unit tests for visualization."""

import inspect
import os
import tempfile

import numpy as np
import pytest
import xarray as xr
from matplotlib import (
    figure as mpl_figure,
)
from matplotlib import (
    image as mpl_image,
)
from matplotlib import (
    pyplot as plt,
)

from gpm import _root_path

plots_dir_path = os.path.join(_root_path, "gpm", "tests", "data", "plots")
image_extension = ".png"
mse_tolerance = 5e-3


def skip_tests_if_no_data() -> None:
    """Skip tests if the test data does not exist."""
    if not os.path.exists(plots_dir_path):
        pytest.skip(
            "Test images not found. Please run `git submodule update --init`. to clone existing test data.",
            allow_module_level=True,
        )


def save_and_check_figure(
    figure: mpl_figure.Figure | None = None,
    name: str = "",
) -> None:
    """Save the current figure to a temporary location and compare it to the reference figure.

    If the reference figure does not exist, it is created and the test is skipped.
    """
    if figure is None:
        figure = plt.gcf()

    # Save reference figure if it does not exist
    reference_path = os.path.join(plots_dir_path, name + image_extension)

    if not os.path.exists(reference_path):
        figure.savefig(reference_path)
        pytest.skip(
            "Reference figure did not exist. Created it. To clone existing test data,"
            "run `git submodule update --init`.",
        )

    # Save current figure to temporary file
    with tempfile.NamedTemporaryFile(suffix=image_extension, delete=False) as tmp_file:
        figure.savefig(tmp_file.name)

        # Compare reference and temporary file
        reference = mpl_image.imread(reference_path)
        tmp = mpl_image.imread(tmp_file.name)

        mse = np.mean((reference - tmp) ** 2)
        assert (
            mse < mse_tolerance
        ), f"Figure {tmp_file.name} is not the same as {name}{image_extension}. MSE {mse} > {mse_tolerance}"

        # Remove temporary file if comparison was successful
        tmp_file.close()
        os.remove(tmp_file.name)
        plt.close()


def get_test_name() -> str:
    """Get a unique name for the calling function.

    If the function is a method of a class, pass the class instance as argument (self).
    """
    inspect_stack = inspect.stack()

    # Add module name
    calling_module = inspect.getmodule(inspect_stack[1][0])
    if calling_module is None:
        raise ValueError("This function must be called from a module.")

    name_parts = [calling_module.__name__]

    # Add class name (if called from a class method)
    class_instance = inspect_stack[1].frame.f_locals.get("self", None)
    if class_instance is not None:
        name_parts.append(class_instance.__class__.__name__)

    # Add function name
    name_parts.append(inspect_stack[1][3])

    return "-".join(name_parts)


def expand_dims(
    dataarray: xr.DataArray,
    size: int,
    dim: str,
    axis: int | None = None,
) -> xr.DataArray:
    """Expand dimensions of a dataarray and fill with random data."""
    dataarray = dataarray.expand_dims(dim={dim: size}, axis=axis)
    rng = np.random.default_rng(seed=0)
    dataarray.data = rng.random(dataarray.data.shape)
    return dataarray
