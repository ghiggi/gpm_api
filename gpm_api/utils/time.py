#!/usr/bin/env python3
"""
Created on Mon Oct 31 13:24:41 2022

@author: ghiggi
"""
import numpy as np
import pandas as pd

from gpm_api.io.checks import (
    check_start_end_time,
    check_time,
)

####--------------------------------------------------------------------------.
############################
#### Subsetting by time ####
############################


def subset_by_time(xr_obj, start_time=None, end_time=None):
    """
    Filter a GPM xarray object by start_time and end_time.

    Parameters
    ----------
    xr_obj :
        A xarray object.
    start_time : datetime.datetime
        Start time.
        By default is None
    end_time : datetime.datetime
        End time.
        By default is None

    Returns
    -------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM xarray object

    """
    # Check inputs
    if start_time is not None:
        start_time = check_time(start_time)
    if end_time is not None:
        end_time = check_time(end_time)
    if start_time is not None and end_time is not None:
        start_time, end_time = check_start_end_time(start_time, end_time)

    # Get 1D dimension of the coordinate
    dim_coords = list(xr_obj["time"].dims)

    # If no dimension, it means that nothing to be subsetted
    # - Likely the time dimension has been dropped ...
    if len(dim_coords) == 0:
        return {}

    # If dimension > 1, not clear how to collapse to 1D the boolean array
    if len(dim_coords) != 1:
        raise ValueError(
            "Impossible to subset a non-dimensional time coordinate with dimension >=2."
        )

    dim_coord = dim_coords[0]

    # Subset by start_time
    if start_time is not None:
        isel_bool = xr_obj["time"] >= pd.Timestamp(start_time)
        if not np.any(isel_bool):
            raise ValueError(f"No timesteps to return with start_time {start_time}.")
        isel_dict = {dim_coord: isel_bool}
        xr_obj = xr_obj.isel(isel_dict)

    # Subset by end_time
    if end_time is not None:
        isel_bool = xr_obj["time"] <= pd.Timestamp(end_time)
        if not np.any(isel_bool):
            raise ValueError(f"No timesteps to return with end_time {end_time}.")
        isel_dict = {dim_coord: isel_bool}
        xr_obj = xr_obj.isel(isel_dict)

    return xr_obj


def subset_by_time_slice(xr_obj, slice):
    start_time = slice.start
    end_time = slice.stop
    return subset_by_time(xr_obj, start_time=start_time, end_time=end_time)


####--------------------------------------------------------------------------.
##################################
#### Time Dimension Sanitizer ####
##################################


def is_nat(timesteps):
    """Return a boolean array indicating timesteps which are NaT."""
    # pd.isnull(np.datetime64('NaT'))
    return pd.isnull(timesteps)


def has_nat(timesteps):
    """Return True if any of the timesteps is NaT."""
    return np.any(is_nat(timesteps))


def interpolate_nat(timesteps, method="linear", limit=5, limit_direction=None, limit_area=None):
    """
    Fill NaT values using an interpolation method.

    Notes:
    - Depending on the interpolation method (i.e. linear) the infilled values
      could have ns resolution.

    - For further information refers to
      https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html

    Parameters
    ----------
    method : str, default 'linear'
        Interpolation technique to use. One of:
        * 'linear': Treat the timesteps as equally spaced.
        * 'pad': Fill in NaTs using existing values.
        * 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'spline',
          'barycentric', 'polynomial': Passed to `scipy.interpolate.interp1d`.
          Both 'polynomial' and 'spline' require that you also specify an
          `order` (int), e.g. ``interpolate_nat(method='polynomial', order=5)``.
        * 'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima',
          'cubicspline': Wrappers around the SciPy interpolation methods of
          similar names. See `Notes` in
          https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
    limit : int, optional
        Maximum number of consecutive NaTs to fill. Must be greater than 0.
    limit_direction : {{'forward', 'backward', 'both'}}, Optional
        Consecutive NaTs will be filled in this direction.
        If limit is specified:
            * If 'method' is 'pad' or 'ffill', 'limit_direction' must be 'forward'.
            * If 'method' is 'backfill' or 'bfill', 'limit_direction' must be
              'backwards'.
        If 'limit' is not specified:
            * If 'method' is 'backfill' or 'bfill', the default is 'backward'
            * else the default is 'forward'
    limit_area : {{`None`, 'inside', 'outside'}}, default None
        If limit is specified, consecutive NaTs will be filled with this restriction.
        * ``None``: No fill restriction.
        * 'inside': Only fill NaTs surrounded by valid values (interpolate).
        * 'outside': Only fill NaTs outside valid values (extrapolate).
    **kwargs : optional
        Keyword arguments to pass on to the interpolating function.

     Returns
    -------
    timesteps, np.array
        Timesteps array of type datetime64[ns]

    """
    # Retrieve input dtype
    timesteps_dtype = timesteps.dtype
    # Convert timesteps to float
    timesteps_num = timesteps.astype(float)
    # Convert NaT to np.nan
    timesteps_num[is_nat(timesteps)] = np.nan
    # Create pd.Series
    series = pd.Series(timesteps_num)
    # Estimate NaT value
    series = series.interpolate(
        method=method,
        limit=limit,
        limit_direction=limit_direction,
        limit_area=limit_area,
    )
    # Convert back to numpy array input dtype
    timesteps = series.to_numpy().astype(timesteps_dtype)
    return timesteps


def infill_timesteps(timesteps, limit):
    """Infill missing timesteps if less than <limit> consecutive."""
    # Check at least two time steps available to infill
    if len(timesteps) <= 2:
        return timesteps

    # Interpolate if maximum <limit> timesteps are missing
    timesteps = interpolate_nat(timesteps, method="linear", limit=limit, limit_area="inside")

    # Check if there are still residual NaT
    if np.any(is_nat(timesteps)):
        raise ValueError(f"More than {limit} consecutive timesteps are NaT.")

    return timesteps


def ensure_time_validity(xr_obj, limit=10):
    """
    Attempt to correct the time coordinate if less than 'limit' consecutive NaT values are present.

    It raise a ValueError if more than consecutive NaT occurs.

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM xarray object.

    Returns
    -------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM xarray object..

    """
    timesteps = xr_obj["time"].values
    timesteps = infill_timesteps(timesteps, limit=limit)
    if "time" not in list(xr_obj.dims):
        xr_obj["time"].data = timesteps
    else:
        xr_obj = xr_obj.assign_coords({"time": timesteps})

    return xr_obj


####--------------------------------------------------------------------------.
