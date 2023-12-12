import inspect
import importlib
import numpy as np
import pytest
import re
from typing import Callable, Dict, List, Tuple
import xarray as xr

import gpm_api  # Needed to register accessors
from gpm_api.accessor.methods import GPM_Base_Accessor, GPM_Dataset_Accessor, GPM_DataArray_Accessor


def get_class_methods(accessor_class) -> Dict[str, Callable]:
    """Get methods from class"""

    method_tuples_list = inspect.getmembers(accessor_class, inspect.isfunction)
    return {name: function for name, function in method_tuples_list if not name.startswith("_")}


base_accessor_methods_dict = get_class_methods(GPM_Base_Accessor)
dataset_accessor_methods_dict = get_class_methods(GPM_Dataset_Accessor)
dataarray_accessor_methods_dict = get_class_methods(GPM_DataArray_Accessor)
accessor_methods_dict = {
    **dataset_accessor_methods_dict,
    **dataarray_accessor_methods_dict,
    **base_accessor_methods_dict,
}
accessor_methods = list(accessor_methods_dict.values())


def get_arguments_list(function: Callable) -> list:
    """Get list of arguments of a function"""

    signature = inspect.signature(function)
    return list(signature.parameters.keys())


def get_function_location(function: Callable) -> str:
    """Get useful debug information about a function"""

    return f"{function.__module__}.{function.__qualname__}"


def get_imported_gpm_method_path(function: Callable) -> Tuple[str, str]:
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


def get_default_arguments_dict(function: Callable) -> Dict[str, object]:
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
    changed_default_arguments: Dict[str, object] = {},
):
    """Check that default arguments of accessor_method and reference_method are the same"""

    accessor_default_arguments = get_default_arguments_dict(accessor_method)
    reference_default_arguments = get_default_arguments_dict(reference_method)
    # modified_default_arguments = {**accessor_default_arguments, **changed_default_arguments}

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


class TestRegisterAccessor:
    """Test that accessor are registered by xarray"""

    def test_dataset(self) -> None:
        ds = xr.Dataset()
        assert hasattr(xr.Dataset, "gpm_api")
        assert hasattr(ds, "gpm_api")

    def test_dataarray(self) -> None:
        da = xr.DataArray()
        assert hasattr(xr.DataArray, "gpm_api")
        assert hasattr(da, "gpm_api")


@pytest.mark.parametrize("accessor_method", accessor_methods)
def test_default_arguments(
    accessor_method: Callable,
) -> None:
    """Test that default arguments are the same between accessor methods and gpm_api methods"""

    imported_module, imported_method_name = get_imported_gpm_method_path(accessor_method)
    module = importlib.import_module(imported_module)
    gpm_method = getattr(module, imported_method_name)
    compare_default_arguments(accessor_method, gpm_method)
