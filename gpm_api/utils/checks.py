#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 18:41:48 2022

@author: ghiggi
"""
import numpy as np
import xarray as xr
from gpm_api.utils.slices import get_contiguous_true_slices, filter_slices_by_size
from gpm_api.utils.geospatial import is_orbit, is_grid


ORBIT_TIME_TOLERANCE = np.timedelta64(3, "s")


def check_is_xarray(x):
    if not isinstance(x, (xr.DataArray, xr.Dataset)):
        raise TypeError("Expecting a xr.Dataset or xr.DataArray.")


def check_is_xarray_dataarray(x):
    if not isinstance(x, xr.DataArray):
        raise TypeError("Expecting a xr.DataArray.")


def check_is_xarray_dataset(x):
    if not isinstance(x, xr.Dataset):
        raise TypeError("Expecting a xr.Dataset.")


def check_is_orbit(xr_obj):
    "Check is a GPM orbit object."
    if not is_orbit(xr_obj):
        raise ValueError("Expecting a GPM ORBIT object.")


def check_is_grid(xr_obj):
    "Check is a GPM grid object."
    if not is_grid(xr_obj):
        raise ValueError("Expecting a GPM GRID object.")


def check_is_spatial_2D_field(da):
    from .geospatial import is_spatial_2D_field

    if not is_spatial_2D_field(da):
        raise ValueError("Expecting a 2D GPM field.")


####--------------------------------------------------------------------------.
############################
#### Regular timesteps  ####
############################


def _get_timesteps(xr_obj):
    """Get timesteps with second precision from xarray object."""
    timesteps = xr_obj["time"].values
    timesteps = timesteps.astype("M8[s]")
    return timesteps


def _infer_time_tolerance(xr_obj):
    """Infer time interval tolerance between timesteps."""
    from gpm_api.utils.geospatial import is_orbit, is_grid

    # For GPM ORBIT objects, use the ORBIT_TIME_TOLERANCE
    if is_orbit(xr_obj):
        tolerance = ORBIT_TIME_TOLERANCE
    # For GPM GRID objects, infer it from the time difference between first two timesteps
    elif is_grid(xr_obj):
        timesteps = _get_timesteps(xr_obj)
        tolerance = np.diff(timesteps[0:2])[0]
    else:
        raise ValueError("Unrecognized GPM xarray object.")
    return tolerance


def _is_regular_timesteps(xr_obj, tolerance=None):
    """Return a boolean array indicating if the next regular timestep is present."""
    # Retrieve timesteps
    timesteps = _get_timesteps(xr_obj)
    # Infer tolerance if not specified
    tolerance = _infer_time_tolerance(xr_obj) if tolerance is None else tolerance
    #  # Identify if the next regular timestep is present
    bool_arr = np.diff(timesteps) <= tolerance
    # Add True to last position
    bool_arr = np.append(bool_arr, True)
    return bool_arr


def get_regular_time_slices(xr_obj, tolerance=None, min_size=1):
    """
    Return a list of slices ensuring timesteps to be regular.

    Output format: [slice(start,stop), slice(start,stop),...]

    Consecutive non-regular timesteps leads to slices of size 1.
    An input with less than 2 timesteps however returns an empty slice or a slice of size 1.

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM xarray object.
    tolerance : np.timedelta, optional
        The timedelta tolerance to define regular vs. non-regular timesteps.
        If None, it uses the first 2 timesteps to derive the tolerance timedelta.
        The default is None. 
    min_size : int
        Minimum size for a slice to be returned.
    
    Returns
    -------
    list_slices : list
        List of slice object to select regular time intervals.
    """
    # Retrieve timestep
    timesteps = _get_timesteps(xr_obj)
    n_timesteps = len(timesteps)

    # Define special behaviour if less than 2 timesteps
    # - If n_timesteps 0, slice(0, 0) returns empty array
    # - If n_timesteps 1, slice(0, 1) returns the single timestep
    # --> If less than 2 timesteps, assume regular timesteps
    if n_timesteps < 2:
        list_slices = [slice(0, n_timesteps)]
    # Otherwise
    else:
        # Get boolean array indicating if the next regular timestep is present
        is_regular = _is_regular_timesteps(xr_obj, tolerance=tolerance)

        # If non-regular timesteps are present, get the slices for each regular interval
        # - If consecutive non-regular timestep occurs, returns slices of size 1
        list_slices = get_contiguous_true_slices(
            is_regular, include_false=True, skip_consecutive_false=False
        )
    
    # Select only slices with at least min_size timesteps
    list_slices = filter_slices_by_size(list_slices, min_size=min_size)
    
    # Return list of slices with regular timesteps
    return list_slices


def get_nonregular_time_slices(xr_obj, tolerance=None):
    """
    Return a list of slices where there are supposedly missing timesteps.

    Output format: [slice(start,stop), slice(start,stop),...]

    The output slices have size 2.
    An input with less than 2 scans (along-track) returns an empty list.

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
        List of slice object to select intervals with non-regular timesteps.
    """
    # Retrieve timestep
    timesteps = _get_timesteps(xr_obj)
    n_timesteps = len(timesteps)

    # Define behaviour if less than 2 timesteps
    # --> Here we decide to return an empty list !
    if n_timesteps < 2:
        list_slices = []
        return list_slices

    # Get boolean array indicating if the next regular timestep is present
    is_regular = _is_regular_timesteps(xr_obj, tolerance=tolerance)

    # If non-regular timesteps are present, get the slices where timesteps non-regularity occur
    if not np.all(is_regular):
        indices_next_nonregular = np.argwhere(~is_regular).flatten()
        list_slices = [slice(i, i + 2) for i in indices_next_nonregular]
    else:
        list_slices = []
    return list_slices


def check_regular_timesteps(xr_obj, tolerance=None, verbose=True):
    """
    Check no missing timesteps for longer than 'tolerance' seconds.

    Note:
    - This sometimes occurs between orbit/grid granules
    - This sometimes occurs within a orbit granule

    Parameters
    ----------
    xr_obj : xr.Dataset or xr.DataArray
        xarray object.
    tolerance : np.timedelta, optional
        The timedelta tolerance to define regular vs. non-regular timesteps.
        The default is None.
        If GPM GRID object, it uses the first 2 timesteps to derive the tolerance timedelta.
        If GPM ORBIT object, it uses the gpm_api.utils.time.ORBIT_TIME_TOLERANCE
    verbose : bool
        If True, it prints the time interval when the non contiguous scans occurs.
        The default is True.

    Returns
    -------

    None.
    """
    list_discontinuous_slices = get_nonregular_time_slices(xr_obj, tolerance=tolerance)
    n_discontinuous = len(list_discontinuous_slices)
    if n_discontinuous > 0:
        # Retrieve discontinous timesteps interval
        timesteps = _get_timesteps(xr_obj)
        list_discontinuous = [timesteps[slc] for slc in list_discontinuous_slices]
        first_problematic_timestep = list_discontinuous[0][0]
        # Print non-regular timesteps
        if verbose:
            for start, stop in list_discontinuous:
                print(f"- Missing timesteps between {start} and {stop}")
        # Raise error and highlight first non-contiguous scan
        raise ValueError(
            f"There are {n_discontinuous} non-regular timesteps. The first occur at {first_problematic_timestep}."
        )


def has_regular_timesteps(xr_obj):
    """Return True if all timesteps are regular. False otherwise."""
    list_discontinuous_slices = get_nonregular_time_slices(xr_obj)
    n_discontinuous = len(list_discontinuous_slices)
    if n_discontinuous > 0:
        return False
    else:
        return True


####--------------------------------------------------------------------------.
########################
#### Regular scans  ####
########################


def _get_along_track_scan_distance(xr_obj):
    """Compute the distance between along_track centroids."""
    from pyproj import Geod

    # Select centroids coordinates in the middle of the cross_track scan
    middle_idx = int(xr_obj["cross_track"].shape[0] / 2)
    lons = xr_obj["lon"].isel(cross_track=middle_idx).data
    lats = xr_obj["lat"].isel(cross_track=middle_idx).data
    # Define between-centroids line coordinates
    start_lons = lons[:-1]
    start_lats = lats[:-1]
    end_lons = lons[1:]
    end_lats = lats[1:]
    # Compute distance
    geod = Geod(ellps="sphere")
    _, _, dist = geod.inv(start_lons, start_lats, end_lons, end_lats)
    return dist


def _is_contiguous_scan(xr_obj):
    """Return a boolean array indicating if the next scan is contiguous."""
    # Compute along track scan distance
    dist = _get_along_track_scan_distance(xr_obj)
    # Convert to km and round
    dist_km = np.round(dist / 1000, 0)

    ##  Option 1
    # Identify if the next scan is contiguous
    # - Use the smallest distance as reference
    # - Assumed to be non contiguous if separated by more than min_dist + half min_dist
    # - This fails if duplicated geolocation --> min_dist = 0
    min_dist = min(dist_km)
    bool_arr = dist_km < (min_dist + min_dist / 2)

    ### Option 2
    # # Identify if the next scan is contiguous
    # # - Use the smallest distance as reference
    # # - Assumed to be non contiguous if exceeds min_dist + min_dist/2
    # # - This fails when more discontiguous/repeated than contiguous
    # dist_unique, dist_counts = np.unique(dist_km, return_counts=True)
    # most_common_dist = dist_unique[np.argmax(dist_counts)]
    # # Identify if the next scan is contiguous
    # bool_arr = dist_km < (most_common_dist + most_common_dist/2)

    # Add True to last position
    bool_arr = np.append(bool_arr, True)
    return bool_arr


def get_contiguous_scan_slices(xr_obj, min_size=2):
    """
    Return a list of slices ensuring contiguous scans.

    Output format: [slice(start,stop), slice(start,stop),...]

    An input with less than 2 scans (along-track) returns an empty list.
    Consecutive non-contiguous scans are discarded and not included in the outputs.
    The minimum size of the output slices is 2.

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM xarray object.
    min_size : int
        Minimum size for a slice to be returned.

    Returns
    -------
    list_slices : list
        List of slice object to select contiguous scans.
    """
    check_is_orbit(xr_obj)
    # Get number of scans
    n_scans = xr_obj["along_track"].shape[0]
    # Define behaviour if less than 2 scan along track
    # - If n_scans 0, slice(0, 0) could return empty array
    # - If n_scans 1, slice(0, 1) could return the single scan
    # --> Here we decide to return an empty list !
    if n_scans < 2:
        list_slices = []
        return list_slices

    # Get boolean array indicating if the next scan is contiguous
    is_contiguous = _is_contiguous_scan(xr_obj)

    # If non-contiguous scans are present, get the slices with contiguous scans
    # - It discard consecutive non-contiguous scans
    list_slices = get_contiguous_true_slices(
        is_contiguous, include_false=True, skip_consecutive_false=True
    )

    # Select only slices with at least 2 scans
    list_slices = filter_slices_by_size(list_slices, min_size=min_size)

    # Return list of contiguous scan slices
    return list_slices


def get_discontiguous_scan_slices(xr_obj):
    """
    Return a list of slices where the scan discontinuity occurs.

    Output format: [slice(start,stop), slice(start,stop),...]

    The output slices have size 2.
    An input with less than 2 scans (along-track) returns an empty list.

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM xarray object.

    Returns
    -------
    list_slices : list
        List of slice object to select discontiguous scans.
    """
    check_is_orbit(xr_obj)

    # Get number of scans
    n_scans = xr_obj["along_track"].shape[0]

    # Define behaviour if less than 2 scan along track
    # - If n_scans 0, slice(0, 0) could return empty array
    # - If n_scans 1, slice(0, 1) could return the single scan
    # --> Here we decide to return an empty list !
    if n_scans < 2:
        list_slices = []
        return list_slices

    # Get boolean array indicating if the next scan is contiguous
    is_contiguous = _is_contiguous_scan(xr_obj)

    # If non-contiguous scans are present, get the slices when discontinuity occurs
    if not np.all(is_contiguous):
        indices_next_discontinuous = np.argwhere(~is_contiguous).flatten()
        list_slices = [slice(i, i + 2) for i in indices_next_discontinuous]
    else:
        list_slices = []
    return list_slices


def check_contiguous_scans(xr_obj, verbose=True):
    """
    Check no missing scans across the along_track direction.

    Note:
    - This sometimes occurs between orbit granules
    - This sometimes occurs within a orbit granule

    Parameters
    ----------
    xr_obj : xr.Dataset or xr.DataArray
        xarray object.
    verbose : bool
        If True, it prints the time interval when the non contiguous scans occurs

    Returns
    -------

    None.
    """
    list_discontinuous_slices = get_discontiguous_scan_slices(xr_obj)
    n_discontinuous = len(list_discontinuous_slices)
    if n_discontinuous > 0:
        # Retrieve discontinous timesteps interval
        timesteps = _get_timesteps(xr_obj)
        list_discontinuous = [timesteps[slc] for slc in list_discontinuous_slices]
        first_problematic_timestep = list_discontinuous[0][0]
        # Print non-contiguous scans
        if verbose:
            for start, stop in list_discontinuous:
                print(f"- Missing scans between {start} and {stop}")
        # Raise error and highlight first non-contiguous scan
        raise ValueError(
            f"There are {n_discontinuous} non-contiguous scans. The first occur at {first_problematic_timestep}."
        )


def has_contiguous_scans(xr_obj):
    """Return True if all scans are contiguous. False otherwise."""
    list_discontinuous_slices = get_discontiguous_scan_slices(xr_obj)
    n_discontinuous = len(list_discontinuous_slices)
    if n_discontinuous > 0:
        return False
    else:
        return True


####--------------------------------------------------------------------------.
##########################
#### Regular granules ####
##########################


def has_missing_granules(xr_obj):
    """Checks GPM object is composed of consecutive granules."""
    from gpm_api.utils.geospatial import is_orbit, is_grid

    if is_orbit(xr_obj):
        granule_ids = xr_obj["gpm_granule_id"].data
        if np.all(np.diff(granule_ids) <= 1):
            return True
        else:
            return False
    if is_grid(xr_obj):
        return has_regular_timesteps(xr_obj)
    else:
        raise ValueError("Unrecognized GPM xarray object.")


####---------------------------------------------------------------------------
#####################################
#### Check GPM object regularity ####
#####################################


def is_regular(xr_obj):
    """Checks the GPM object is regular.

    For GPM ORBITS, it checks that the scans are contiguous.
    For GPM GRID, it checks that the timesteps are regularly spaced.
    """
    from gpm_api.utils.geospatial import is_orbit, is_grid

    if is_orbit(xr_obj):
        return has_contiguous_scans(xr_obj)
    elif is_grid(xr_obj):
        return has_regular_timesteps(xr_obj)
    else:
        raise ValueError("Unrecognized GPM xarray object.")


def get_regular_slices(xr_obj, min_size=None):
    """
    Return a list of slices to select regular GPM objects.

    For GPM ORBITS, it returns slices to select contiguouse scans.
    For GPM GRID, it returns slices to select periods with regular timesteps.

    The output format is: [slice(start,stop), slice(start,stop),...]
    For more information, read the documentation of:
    - gpm_api.utils.checks.get_contiguous_scan_slices
    - gpm_api.utils.checks.get_regular_time_slices
    
    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM xarray object.
    min_size : int
        Minimum size for a slice to be returned.
        If None, default to 1 for GRID objects, 2 for ORBIT objects.

    Returns
    -------
    list_slices : list
        List of slice object to select regular portions.
    """
    from gpm_api.utils.geospatial import is_orbit, is_grid

    if is_orbit(xr_obj):
        min_size = 2 if min_size is None else min_size
        return get_contiguous_scan_slices(xr_obj, min_size=min_size)
    elif is_grid(xr_obj):
        min_size = 1 if min_size is None else min_size
        return get_regular_time_slices(xr_obj, min_size=min_size)
    else:
        raise ValueError("Unrecognized GPM xarray object.")


####--------------------------------------------------------------------------.
