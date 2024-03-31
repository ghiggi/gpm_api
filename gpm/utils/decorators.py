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
"""This module contains functions decorators checking GPM-API object type."""
import functools

from gpm.checks import check_has_along_track_dimension as check_has_along_track_dimension_fun
from gpm.checks import check_has_cross_track_dimension as check_has_cross_track_dimension_fun
from gpm.checks import check_is_gpm_object as check_is_gpm_object_fun
from gpm.checks import check_is_grid as check_is_grid_fun
from gpm.checks import check_is_orbit as check_is_orbit_fun


def _get_xr_obj(args, kwargs):
    if "xr_obj" in kwargs:
        return kwargs["xr_obj"]
    return args[0]


def check_is_orbit(function):
    """Decorator function to check if input is a GPM ORBIT object. Raise ValueError if not."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        xr_obj = _get_xr_obj(args, kwargs)
        check_is_orbit_fun(xr_obj)
        return function(*args, **kwargs)

    return wrapper


def check_is_grid(function):
    """Decorator function to check if input is a GPM GRID object. Raise ValueError if not."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        xr_obj = _get_xr_obj(args, kwargs)
        check_is_grid_fun(xr_obj)
        return function(*args, **kwargs)

    return wrapper


def check_is_gpm_object(function):
    """Decorator function to check if input is a GPM object. Raise ValueError if not."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        xr_obj = _get_xr_obj(args, kwargs)
        check_is_gpm_object_fun(xr_obj)
        return function(*args, **kwargs)

    return wrapper


def check_has_cross_track_dimension(function):
    """Check that the cross-track dimension is available.

    If not available, raise an error.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        xr_obj = _get_xr_obj(args, kwargs)
        check_has_cross_track_dimension_fun(xr_obj)
        return function(*args, **kwargs)

    return wrapper


def check_has_along_track_dimension(function):
    """Check that the along-track dimension is available.

    If not available, raise an error.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        xr_obj = _get_xr_obj(args, kwargs)
        check_has_along_track_dimension_fun(xr_obj)
        return function(*args, **kwargs)

    return wrapper
