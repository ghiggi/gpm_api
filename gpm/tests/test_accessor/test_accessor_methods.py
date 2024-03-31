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
"""This module defines test units for GPM-API accessor."""

import importlib
import inspect
import re
from typing import Callable

import numpy as np
import pytest
import xarray as xr
from pytest_mock import MockFixture

from gpm.accessor.methods import GPM_Base_Accessor, GPM_DataArray_Accessor, GPM_Dataset_Accessor


def get_class_methods(accessor_class) -> dict[str, Callable]:
    """Get methods from class"""

    method_tuples_list = inspect.getmembers(accessor_class, inspect.isfunction)
    return {name: function for name, function in method_tuples_list if not name.startswith("_")}


base_accessor_methods_dict = get_class_methods(GPM_Base_Accessor)
dataset_accessor_methods_dict = get_class_methods(GPM_Dataset_Accessor)
dataarray_accessor_methods_dict = get_class_methods(GPM_DataArray_Accessor)
base_accessor_methods = list(base_accessor_methods_dict.values())
dataset_accessor_methods = [
    v for v in dataset_accessor_methods_dict.values() if v not in base_accessor_methods
]
dataarray_accessor_methods = [
    v for v in dataarray_accessor_methods_dict.values() if v not in base_accessor_methods
]
accessor_methods = base_accessor_methods + dataset_accessor_methods + dataarray_accessor_methods


def get_arguments_list(function: Callable, remove_self: bool = True) -> list:
    """Get list of arguments of a function"""

    signature = inspect.signature(function)
    arguments = list(signature.parameters.keys())

    if remove_self and arguments[0] == "self":
        arguments = arguments[1:]

    return arguments


def get_function_location(function: Callable) -> str:
    """Get useful debug information about a function"""

    return f"{function.__module__}.{function.__qualname__}"


def get_imported_gpm_method_path(function: Callable) -> tuple[str, str]:
    """Get path of imported gpm method in accessor method source code (format is "module.method"))"""

    source = inspect.getsource(function)
    import_pattern = re.compile(r"from (\S+) import (\S+)")
    match = import_pattern.search(source)

    if match:
        module = match.group(1)
        method_name = match.group(2)
        return module, method_name

    else:
        raise ValueError(f"No import statement found in {get_function_location(function)}")


def get_imported_gpm_method(accessor_method: Callable) -> Callable:
    imported_module, imported_method_name = get_imported_gpm_method_path(accessor_method)
    module = importlib.import_module(imported_module)
    gpm_method = getattr(module, imported_method_name)
    return gpm_method


def get_default_arguments_dict(function: Callable) -> dict[str, object]:
    """Get default arguments of a function as a dictionary"""

    signature = inspect.signature(function)
    default_arguments = {}

    for key, value in signature.parameters.items():
        if value.default is not inspect.Parameter.empty:
            default_arguments[key] = value.default

    return default_arguments


def compare_default_arguments(
    accessor_method: Callable,
    reference_method: Callable,
):
    """Check that default arguments of accessor_method and reference_method are the same"""

    accessor_default_arguments = get_default_arguments_dict(accessor_method)
    reference_default_arguments = get_default_arguments_dict(reference_method)

    missing_arguments = set(reference_default_arguments.keys()) - set(
        accessor_default_arguments.keys()
    )
    assert (
        not missing_arguments
    ), f"Missing arguments in {get_function_location(accessor_method)}: {missing_arguments}"

    extra_arguments = set(accessor_default_arguments.keys()) - set(
        reference_default_arguments.keys()
    )
    assert (
        not extra_arguments
    ), f"Extra arguments in {get_function_location(accessor_method)}: {extra_arguments}"

    different_values = {
        key: (accessor_default_arguments[key], reference_default_arguments[key])
        for key in reference_default_arguments
        if accessor_default_arguments[key] != reference_default_arguments[key]
        and not (
            np.isnan(accessor_default_arguments[key]) and np.isnan(reference_default_arguments[key])
        )
    }
    assert (
        not different_values
    ), f"Different values in {get_function_location(accessor_method)}: {different_values}"


def mock_associated_gpm_method(
    accessor_method: Callable,
    mocker: MockFixture,
) -> None:
    imported_module, imported_method_name = get_imported_gpm_method_path(accessor_method)

    def mock_gpm_method(xr_obj, *args, **kwargs):
        args_dict = {arg: arg for arg in args}
        return {"xr_obj": xr_obj, **args_dict, **kwargs}

    mocker.patch(f"{imported_module}.{imported_method_name}", side_effect=mock_gpm_method)


class TestRegisterAccessor:
    """Test that accessor are registered by xarray"""

    def test_dataset(self) -> None:
        ds = xr.Dataset()
        assert hasattr(xr.Dataset, "gpm")
        assert hasattr(ds, "gpm")

    def test_dataarray(self) -> None:
        da = xr.DataArray()
        assert hasattr(xr.DataArray, "gpm")
        assert hasattr(da, "gpm")


@pytest.mark.parametrize("accessor_method", accessor_methods)
def test_default_arguments(
    accessor_method: Callable,
) -> None:
    """Test that default arguments are the same between accessor methods and gpm methods"""

    gpm_method = get_imported_gpm_method(accessor_method)
    compare_default_arguments(accessor_method, gpm_method)


@pytest.mark.parametrize("accessor_method", base_accessor_methods + dataset_accessor_methods)
def test_passed_arguments_dataset(
    mocker: MockFixture,
    accessor_method: Callable,
) -> None:
    """Test that arguments are passed correctly to gpm methods for datasets"""

    gpm_method_arguments = get_arguments_list(get_imported_gpm_method(accessor_method))
    mock_associated_gpm_method(accessor_method, mocker)

    # Create dictionary of accessor arguments
    accessor_arguments = get_arguments_list(accessor_method)
    args_kwargs_dict = {arg: arg for arg in accessor_arguments}
    ds = xr.Dataset()

    # Different behavior if dataarray variable is extracted from dataset
    if "variable" in accessor_arguments and "variable" not in gpm_method_arguments:
        da = xr.DataArray([0])  # Must not be empty, otherwise comparison with itself fails
        ds["variable"] = da
        expected = {"xr_obj": da, **args_kwargs_dict}
        del expected["variable"]

    else:
        expected = {"xr_obj": ds, **args_kwargs_dict}

    ds_accessor_method = getattr(ds.gpm, accessor_method.__name__)
    returned = ds_accessor_method(**args_kwargs_dict)

    assert (
        returned == expected
    ), f"Arguments not passed correctly in {get_function_location(accessor_method)}"


@pytest.mark.parametrize("accessor_method", base_accessor_methods + dataarray_accessor_methods)
def test_passed_arguments_dataarray(
    mocker: MockFixture,
    accessor_method: Callable,
) -> None:
    """Test that arguments are passed correctly to gpm methods for dataarrays"""

    mock_associated_gpm_method(accessor_method, mocker)

    # Create dictionary of accessor arguments
    accessor_arguments = get_arguments_list(accessor_method)
    args_kwargs_dict = {arg: arg for arg in accessor_arguments}
    da = xr.DataArray()
    da_accessor_method = getattr(da.gpm, accessor_method.__name__)

    expected = {"xr_obj": da, **args_kwargs_dict}
    returned = da_accessor_method(**args_kwargs_dict)

    assert (
        returned == expected
    ), f"Arguments not passed correctly in {get_function_location(accessor_method)}"
