#!/usr/bin/env python3
"""
Created on Wed Feb  7 10:43:46 2024

@author: ghiggi
"""
import functools

from gpm_api.checks import check_has_along_track_dimension as check_has_along_track_dimension_fun
from gpm_api.checks import check_has_cross_track_dimension as check_has_cross_track_dimension_fun
from gpm_api.checks import check_is_gpm_object as check_is_gpm_object_fun
from gpm_api.checks import check_is_grid as check_is_grid_fun
from gpm_api.checks import check_is_orbit as check_is_orbit_fun


def check_is_orbit(function):
    """Decorator function to check if input is a GPM ORBIT object. Raise ValueError if not."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        xr_obj = args[0]
        check_is_orbit_fun(xr_obj)
        return function(*args, **kwargs)

    return wrapper


def check_is_grid(function):
    """Decorator function to check if input is a GPM GRID object. Raise ValueError if not."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        xr_obj = args[0]
        check_is_grid_fun(xr_obj)
        return function(*args, **kwargs)

    return wrapper


def check_is_gpm_object(function):
    """Decorator function to check if input is a GPM object. Raise ValueError if not."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        xr_obj = args[0]
        check_is_gpm_object_fun(xr_obj)
        return function(*args, **kwargs)

    return wrapper


def check_has_cross_track_dimension(function):
    """Check that the cross-track dimension is available.

    If not available, raise an error."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        # Write decorator function logic here
        xr_obj = args[0]
        check_has_cross_track_dimension_fun(xr_obj)
        return function(*args, **kwargs)

    return wrapper


def check_has_along_track_dimension(function):
    """Check that the along-track dimension is available.

    If not available, raise an error."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        # Write decorator function logic here
        xr_obj = args[0]
        check_has_along_track_dimension_fun(xr_obj)
        return function(*args, **kwargs)

    return wrapper
