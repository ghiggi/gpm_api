#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:24:41 2022

@author: ghiggi
"""
import numpy as np 
import pandas as pd 
from gpm_api.io.checks import (
    check_time, 
    check_start_end_time,
)

# NOTE! GPM ORBIT time array is most often non-unique ! 
# --> Selection by time must be performed with 

ORBIT_TIME_TOLERANCE = np.timedelta64(3, 's')

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
        raise ValueError("Impossible to subset a non-dimensional time coordinate with dimension >=2.")
        
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
############################
#### Checks & Sanitizer ####
############################
def is_nat(timesteps): 
    """Return a boolean array indicating timesteps which are NaT."""
    # pd.isnull(np.datetime64('NaT'))
    return pd.isnull(timesteps)
      

def has_nat(timesteps): 
    """Return True if any of the timesteps is NaT."""
    return np.any(is_nat(timesteps))


def interpolate_nat(timesteps, 
                    method='linear',
                    limit=5,
                    limit_direction=None, 
                    limit_area=None): 
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
    series = series.interpolate(method=method, 
                                limit=limit, 
                                limit_direction=limit_direction,
                                limit_area=limit_area)
    # Convert back to numpy array input dtype 
    timesteps = series.to_numpy().astype(timesteps_dtype)
    return timesteps 

    
def _check_regular_timesteps(timesteps, tolerance=None, verbose=True):
    """
    Check no missing timesteps for longer than 'tolerance' seconds. 
    
    Parameters
    ----------
    timesteps : np.array
        Array of timesteps.
    tolerance : np.timedelta, optional
        The default is None.
        If None, it use the first 2 timesteps to derive the tolerance timedelta.
        Otherwise, it use the specified np.timedelta.

    Returns
    -------
    
    None.

    """
    # Infer tolerance if not specified
    if tolerance is None: 
        # If less than 2 timesteps, check nothing
        if len(timesteps) < 2: 
            return None
        # Otherwise infer t_res using first two timesteps
        tolerance = np.diff(timesteps[0:2])[0]
        
    # Cast timesteps to second accuracy
    timesteps = timesteps.astype("M8[s]")
    # Identify occurence of non-regular timesteps
    is_regular = np.diff(timesteps) <= tolerance
    # If non-regular timesteps occurs, reports the problems
    if not np.all(is_regular): 
        indices_missing = np.argwhere(~is_regular).flatten()           
        list_discontinuous = [timesteps[i:i+2]   for i in indices_missing]
        first_timestep_problem = list_discontinuous[0][0]
        # Display non-regular time interval 
        if verbose: 
            for start, stop in list_discontinuous:
                print(f"- Missing data between {start} and {stop}")
        # Raise error and highligh first non-regular timestep
        raise ValueError(f"There are non-regular timesteps starting from {first_timestep_problem}")
        

def check_regular_timesteps(xr_obj, verbose=True):
    """Checks GPM object has regular "time" coordinate."""
    from gpm_api.utils.geospatial import is_orbit, is_grid 
    timesteps = xr_obj['time'].values
    if is_orbit(xr_obj): 
        _check_regular_timesteps(timesteps, tolerance=ORBIT_TIME_TOLERANCE, verbose=verbose)
    elif is_grid(xr_obj): 
        _check_regular_timesteps(timesteps, tolerance=None, verbose=verbose)
    else: 
        raise ValueError("Unrecognized GPM xarray object.")
        
        
def _has_regular_orbit(xr_obj):
    """Check if GPM orbit object has regular timesteps."""
    # Retrieve timesteps 
    timesteps = xr_obj['time'].values.copy() # Avoid inplace remplacement of original data
    try: 
        _check_regular_timesteps(timesteps, tolerance=ORBIT_TIME_TOLERANCE, verbose=False)
        flag = True
    except ValueError:
        flag = False 
    return flag 
    
    
def _has_regular_grid(xr_obj):
    """Check if GPM grid object has regular timesteps."""
    # Retrieve timesteps 
    timesteps = xr_obj['time'].values
    if len(np.unique(np.diff(timesteps))) != 1: 
        return False 
    else:
        return True 
    
    
def has_regular_timesteps(xr_obj):
    """Checks GPM object has regular "time" coordinate.
    
    If there are NaT values in the time array, it returns False.    
    """
    from gpm_api.utils.geospatial import is_orbit, is_grid 
    if is_orbit(xr_obj): 
        return _has_regular_orbit(xr_obj) 
    elif is_grid(xr_obj): 
        return _has_regular_grid(xr_obj) 
    else: 
        raise ValueError("Unrecognized GPM xarray object.")


def has_consecutive_granules(xr_obj):
    """Checks GPM object is composed of consecutive granules."""
    from gpm_api.utils.geospatial import is_orbit, is_grid 
    if is_orbit(xr_obj):
        granule_ids =  xr_obj['gpm_granule_id'].data
        if np.all(np.diff(granule_ids) <= 1):
            return True
        else: 
            return False 
    if is_grid(xr_obj):
        return _has_regular_grid(xr_obj)
    else: 
        raise ValueError("Unrecognized GPM xarray object.")
        
        
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
    timesteps = xr_obj['time'].values
    # Interpolate if maximum 10 timesteps are missing 
    timesteps = interpolate_nat(timesteps, method='linear', limit=limit, limit_area="inside")
    
    # Check if there are still residual NaT 
    if np.any(is_nat(timesteps)): 
        raise ValueError("More than 10 consecutive timesteps are NaT.")
    
    # Update timesteps xarray object 
    xr_obj['time'].data = timesteps   
    
    return xr_obj


def get_regular_time_slices(xr_obj, tolerance=None):
    """
    Return a list of time slices which are regular 
    
    [slice(start,stop), slice(start,stop),...]

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM xarray object.
    tolerance : np.timedelta, optional
        The timedelta tolerance to define regular vs. non-regular timesteps.
        The default is None.
        If GPM GRID object, it uses the first 2 timesteps to derive the tolerance timedelta.
        If GPM ORBIT object, it uses the gpm_api.utils.time.ORBIT_TIME_TOLERANCE
        
    Returns
    -------
    list_slices : list
        List of slice object to select regular time intervals.
    """
    # Note: start and stop could be changes as str format (is accepted by subset_by_time*)
    from gpm_api.utils.geospatial import is_orbit, is_grid 
    
    # Retrieve timestep 
    timesteps = xr_obj['time'].values
    timesteps = timesteps.astype("M8[s]")
    
    # If less than 2 timesteps, check nothing
    if len(timesteps) < 2: 
        list_slices = [slice(start=timesteps[0], end=timesteps[0])]
        return list_slices
    
    # Infer tolerance if not specified
    if tolerance is None: 
        if is_orbit(xr_obj): 
            tolerance=ORBIT_TIME_TOLERANCE
        elif is_grid(xr_obj): 
            # Infer t_res using first two timesteps
            tolerance = np.diff(timesteps[0:2])[0]    
        else:   
            raise ValueError("Unrecognized GPM xarray object.")
            
    # Identify occurence of non-regular timesteps
    is_regular = np.diff(timesteps) <= tolerance
    
    # If non-regular timesteps occurs, reports the regular slices
    if not np.all(is_regular): 
        # Retrieve last indices where is regular 
        indices_start_missing = np.argwhere(~is_regular).flatten()     
        list_slices = []
        for i, idx in enumerate(indices_start_missing): 
            # Retrieve start 
            if i == 0: 
                start = timesteps[0] 
            else: 
                previous_start_idx = indices_start_missing[i-1]
                start = timesteps[previous_start_idx+1]
            # Retrieve stop 
            stop = timesteps[idx] 
            slc = slice(start, stop)
            list_slices.append(slc)
            
        # If last idx is not one of the 2 last timestep, add last interval 
        if idx !=  len(timesteps) - 2:
           start = timesteps[idx+1]
           stop = timesteps[-1]
           slc = slice(start, stop)
           list_slices.append(slc) 
           
    # Otherwise return start and end time of timesteps 
    else: 
        list_slices = [slice(timesteps[0], timesteps[-1])]       
    return list_slices
    
    
 

