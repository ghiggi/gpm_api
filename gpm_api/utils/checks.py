#!/usr/bin/env python3
"""
Created on Sat Dec 10 18:41:48 2022

@author: ghiggi
"""
import functools

import numpy as np
import pandas as pd

from gpm_api.checks import (
    check_is_orbit,
    is_grid,
    is_orbit,
)

# check_is_grid,
from gpm_api.utils.slices import (
    get_list_slices_from_bool_arr,
    list_slices_difference,
    list_slices_filter,
    list_slices_flatten,
    list_slices_intersection,
    list_slices_sort,
    list_slices_union,
)

ORBIT_TIME_TOLERANCE = np.timedelta64(3, "s")

# TODO: raise ValueError("Unrecognized GPM xarray object.") has decorator check?


####--------------------------------------------------------------------------.
##########################
#### Regular granules ####
##########################


def get_missing_granule_numbers(xr_obj):
    """Return ID numbers of missing granules."""
    granule_ids = np.unique(xr_obj["gpm_granule_id"].data)
    min_id = min(granule_ids)
    max_id = max(granule_ids)
    possible_ids = np.arange(min_id, max_id + 1)
    missing_ids = possible_ids[~np.isin(possible_ids, granule_ids)]
    return missing_ids


def _is_contiguous_granule(granule_ids):
    """Return a boolean array indicating if the next scan is not the same/next granule."""
    # Retrieve if next scan is in the same or next granule (True) or False.
    bool_arr = np.diff(granule_ids) <= 1

    # Add True to last position
    bool_arr = np.append(bool_arr, True)
    return bool_arr


def get_slices_contiguous_granules(xr_obj, min_size=2):
    """
    Return a list of slices ensuring contiguous granules.

    Output format: [slice(start,stop), slice(start,stop),...]

    The minimum size of the output slices is 2.

    Note: for GRID (i.e. IMERG) products, it checks for regular timesteps !
    Note: No granule_id is provided for GRID products

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM xarray object.
    min_size : int
        Minimum size for a slice to be returned.

    Returns
    -------
    list_slices : list
        List of slice object to select contiguous granules.
    """
    if is_grid(xr_obj):
        return get_slices_regular_time(xr_obj, tolerance=None, min_size=min_size)
    if is_orbit(xr_obj):
        # Get number of scans/timesteps
        n_scans = xr_obj["gpm_granule_id"].shape[0]
        # Define behaviour if less than 2 scans/timesteps
        # - If n_scans 0, slice(0, 0) could return empty array
        # - If n_scans 1, slice(0, 1) could return the single scan
        # --> Here we decide to return an empty list !
        if n_scans < min_size:
            list_slices = []
            return list_slices

        # Get boolean array indicating if the next scan/timesteps is same or next granule
        bool_arr = _is_contiguous_granule(xr_obj["gpm_granule_id"].data)

        # If granules are missing present, get the slices with non-missing granules
        list_slices = get_list_slices_from_bool_arr(
            bool_arr, include_false=True, skip_consecutive_false=True
        )

        # Select only slices with at least 2 scans
        list_slices = list_slices_filter(list_slices, min_size=min_size)

        # Return list of contiguous scan slices
        return list_slices
    else:
        raise ValueError("Unrecognized GPM xarray object.")


def check_missing_granules(xr_obj):
    """
    Check no missing granules in the GPM Dataset.

    Parameters
    ----------
    xr_obj : xr.Dataset or xr.DataArray
        xarray object.

    """
    missing_ids = get_missing_granule_numbers(xr_obj)
    n_missing = len(missing_ids)
    if n_missing > 0:
        msg = f"There are {n_missing} missing granules. Their IDs are {missing_ids}."
        raise ValueError(msg)


def check_contiguous_granules(xr_obj):
    return check_missing_granules(xr_obj)


def has_contiguous_granules(xr_obj):
    """Checks GPM object is composed of consecutive granules."""
    from gpm_api.checks import is_grid, is_orbit

    if is_orbit(xr_obj):
        return bool(np.all(_is_contiguous_granule(xr_obj["gpm_granule_id"].data)))
    if is_grid(xr_obj):
        return has_regular_time(xr_obj)
    else:
        raise ValueError("Unrecognized GPM xarray object.")


def has_missing_granules(xr_obj):
    """Checks GPM object has missing granules."""
    from gpm_api.checks import is_grid, is_orbit

    if is_orbit(xr_obj):
        return bool(np.any(~_is_contiguous_granule(xr_obj["gpm_granule_id"].data)))
    if is_grid(xr_obj):
        return ~has_regular_time(xr_obj)
    else:
        raise ValueError("Unrecognized GPM xarray object.")


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
    from gpm_api.checks import is_grid, is_orbit

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


def _is_regular_time(xr_obj, tolerance=None):
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


def get_slices_regular_time(xr_obj, tolerance=None, min_size=1):
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
        is_regular = _is_regular_time(xr_obj, tolerance=tolerance)

        # If non-regular timesteps are present, get the slices for each regular interval
        # - If consecutive non-regular timestep occurs, returns slices of size 1
        list_slices = get_list_slices_from_bool_arr(
            is_regular, include_false=True, skip_consecutive_false=False
        )

    # Select only slices with at least min_size timesteps
    list_slices = list_slices_filter(list_slices, min_size=min_size)

    # Return list of slices with regular timesteps
    return list_slices


def get_slices_non_regular_time(xr_obj, tolerance=None):
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
        If GPM ORBIT object, it uses the gpm_api.utils.time.ORBIT_TIME_TOLERANCE.
        It is discouraged to use this function for GPM ORBIT objects !

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
    is_regular = _is_regular_time(xr_obj, tolerance=tolerance)

    # If non-regular timesteps are present, get the slices where timesteps non-regularity occur
    if not np.all(is_regular):
        indices_next_nonregular = np.argwhere(~is_regular).flatten()
        list_slices = [slice(i, i + 2) for i in indices_next_nonregular]
    else:
        list_slices = []
    return list_slices


def check_regular_time(xr_obj, tolerance=None, verbose=True):
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
    list_discontinuous_slices = get_slices_non_regular_time(xr_obj, tolerance=tolerance)
    n_discontinuous = len(list_discontinuous_slices)
    if n_discontinuous > 0:
        # Retrieve discontinuous timesteps interval
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


def has_regular_time(xr_obj):
    """Return True if all timesteps are regular. False otherwise."""
    list_discontinuous_slices = get_slices_non_regular_time(xr_obj)
    n_discontinuous = len(list_discontinuous_slices)
    if n_discontinuous > 0:
        return False
    else:
        return True


####--------------------------------------------------------------------------.
########################
#### Regular scans  ####
########################
def _check_cross_track(function):
    """Check that the cross-track dimension is available.

    If not available, raise an error."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        # Write decorator function logic here
        xr_obj = args[0]
        if "cross_track" not in xr_obj.dims:
            raise ValueError(
                "cross-track dimension not available. Not possible to check for scan regularity."
            )
        # Call the function
        result = function(*args, **kwargs)
        # Return results
        return result

    return wrapper


@_check_cross_track
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


@_check_cross_track
def _is_contiguous_scans(xr_obj):
    """Return a boolean array indicating if the next scan is contiguous.

    The last element is set to True since it can not be verified.
    """
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


@_check_cross_track
def get_slices_contiguous_scans(xr_obj, min_size=2):
    """
    Return a list of slices ensuring contiguous scans (and granules).

    Output format: [slice(start,stop), slice(start,stop),...]

    It checks for contiguous scans only in the middle of the cross-track !
    If a scan geolocation is NaN, it will be considered non-contiguous.
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
    is_contiguous = _is_contiguous_scans(xr_obj)

    # If non-contiguous scans are present, get the slices with contiguous scans
    # - It discard consecutive non-contiguous scans
    list_slices = get_list_slices_from_bool_arr(
        is_contiguous, include_false=True, skip_consecutive_false=True
    )
    # Select only slices with at least 2 scans
    list_slices = list_slices_filter(list_slices, min_size=min_size)

    # Also retrieve the slices with non missing granule
    list_slices1 = get_slices_contiguous_granules(xr_obj)

    # Perform list_slices intersection
    list_slices = list_slices_intersection(list_slices, list_slices1)

    # Return list of contiguous scan slices
    return list_slices


@_check_cross_track
def get_slices_non_contiguous_scans(xr_obj):
    """
    Return a list of slices where the scans discontinuity occurs.

    Output format: [slice(start,stop), slice(start,stop),...]

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

    ### CODE to output slices with size 2.
    # # Get boolean array indicating if the next scan is contiguous
    # is_contiguous = _is_contiguous_scans(xr_obj)

    # # If non-contiguous scans are present, get the slices when discontinuity occurs
    # if not np.all(is_contiguous):
    #     indices_next_discontinuous = np.argwhere(~is_contiguous).flatten()
    #     list_slices = [slice(i, i + 2) for i in indices_next_discontinuous]
    # else:
    #     list_slices = []
    # return list_slices

    list_slices_valid = get_slices_contiguous_scans(xr_obj, min_size=2)
    list_slices_full = [slice(0, len(xr_obj["along_track"]))]
    list_slices = list_slices_difference(list_slices_full, list_slices_valid)
    return list_slices


@_check_cross_track
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
    list_discontinuous_slices = get_slices_non_contiguous_scans(xr_obj)
    n_discontinuous = len(list_discontinuous_slices)
    if n_discontinuous > 0:
        # Retrieve discontinuous timesteps interval
        timesteps = _get_timesteps(xr_obj)
        list_discontinuous = [
            (timesteps[slc][0], timesteps[slc][-1]) for slc in list_discontinuous_slices
        ]
        first_problematic_timestep = list_discontinuous[0][0]
        # Print non-contiguous scans
        if verbose:
            for start, stop in list_discontinuous:
                print(f"- Missing scans between {start} and {stop}")
        # Raise error and highlight first non-contiguous scan
        msg = f"There are {n_discontinuous} non-contiguous scans."
        msg += f"The first occur at {first_problematic_timestep}."
        raise ValueError(msg)


@_check_cross_track
def has_contiguous_scans(xr_obj):
    """Return True if all scans are contiguous. False otherwise."""
    list_discontinuous_slices = get_slices_non_contiguous_scans(xr_obj)
    n_discontinuous = len(list_discontinuous_slices)
    if n_discontinuous > 0:
        return False
    else:
        return True


####--------------------------------------------------------------------------.
#############################
#### Regular geolocation ####
#############################


def _is_non_valid_geolocation(xr_obj, x="lon"):
    """Return a boolean array indicating if the geolocation is invalid.

    True = Invalid, False = Valid
    """
    bool_arr = np.isnan(xr_obj[x])
    return bool_arr


def _is_valid_geolocation(xr_obj, x="lon"):
    """Return a boolean array indicating if the geolocation is valid.

    True = Valid, False = Invalid
    """
    bool_arr = ~np.isnan(xr_obj[x])
    return bool_arr


def get_slices_valid_geolocation(xr_obj, min_size=2):
    """Return a list of along-track slices ensuring valid geolocation.

    Output format: [slice(start,stop), slice(start,stop),...]

    The minimum size of the output slices is 2.

    If at a given cross-track index, there are always wrong geolocation,
    it discards such cross-track index(es) before identifying the along-track slices.

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM xarray object.
    min_size : int
        Minimum size for a slice to be returned.
        The default is 2.

    Returns
    -------
    list_slices : list
        List of slice object with valid geolocation.
    """
    # Check input
    if is_grid(xr_obj):
        raise ValueError("For GRID products, geolocation is expected to be valid.")
    if is_orbit(xr_obj):
        # - Get invalid coordinates
        invalid_coords = _is_non_valid_geolocation(xr_obj, x="lon")
        # - Identify cross-track index that along-track are always invalid
        idx_cross_track_not_all_invalid = np.where(~invalid_coords.all("along_track"))[0]
        # - If all invalid, return empty list
        if len(idx_cross_track_not_all_invalid) == 0:
            list_slices = []
            return list_slices
        # - Select only cross-track index that are not all invalid along-track
        invalid_coords = invalid_coords.isel(cross_track=idx_cross_track_not_all_invalid)
        # - Now identify scans across which there are still invalid coordinates
        invalid_scans = invalid_coords.any(dim="cross_track")
        valid_scans = ~invalid_scans
        # - Now identify valid along-track slices
        list_slices = get_list_slices_from_bool_arr(
            valid_scans, include_false=False, skip_consecutive_false=True
        )
        # Select only slices with at least 2 scans
        list_slices = list_slices_filter(list_slices, min_size=min_size)
        return list_slices
    return None


def get_slices_non_valid_geolocation(xr_obj):
    """Return a list of along-track slices with non-valid geolocation.

    Output format: [slice(start,stop), slice(start,stop),...]

    The minimum size of the output slices is 2.

    If at a given cross-track index, there are always wrong geolocation,
    it discards such cross-track index(es) before identifying the along-track slices.

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM xarray object.
    min_size : int
        Minimum size for a slice to be returned.
        The default is 1.

    Returns
    -------
    list_slices : list
        List of slice object with non-valid geolocation.
    """
    list_slices_valid = get_slices_valid_geolocation(xr_obj, min_size=1)
    list_slices_full = [slice(0, len(xr_obj["along_track"]))]
    list_slices = list_slices_difference(list_slices_full, list_slices_valid)
    return list_slices


def check_valid_geolocation(xr_obj, verbose=True):
    """
    Check no geolocation errors in the GPM Dataset.

    Parameters
    ----------
    xr_obj : xr.Dataset or xr.DataArray
        xarray object.

    """
    list_invalid_slices = get_slices_non_valid_geolocation(xr_obj)
    n_invalid_scan_slices = len(list_invalid_slices)
    if n_invalid_scan_slices > 0:
        # Retrieve timesteps interval with non valid geolocation
        timesteps = _get_timesteps(xr_obj)
        list_invalid = [(timesteps[slc][0], timesteps[slc][-1]) for slc in list_invalid_slices]
        first_problematic_timestep = list_invalid[0][0]
        # Print non-contiguous scans
        if verbose:
            for start, stop in list_invalid:
                print(f"- Missing scans between {start} and {stop}")
        # Raise error and highlight first non-contiguous scan
        msg = f"There are {n_invalid_scan_slices} swath portions with non-valid geolocation."
        msg += f"The first occur at {first_problematic_timestep}."
        raise ValueError(msg)
    return


def has_valid_geolocation(xr_obj):
    """Checks GPM object has valid geolocation."""
    if is_orbit(xr_obj):
        list_invalid_slices = get_slices_non_valid_geolocation(xr_obj)
        n_invalid_scan_slices = len(list_invalid_slices)
        return n_invalid_scan_slices == 0
    if is_grid(xr_obj):
        return True
    else:
        raise ValueError("Unrecognized GPM xarray object.")


def apply_on_valid_geolocation(function):
    """A decorator that apply the get_slices_<function> only on portions
    with valid geolocation."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        xr_obj = args[0]
        # Get slices with valid geolocation
        list_slices_valid = get_slices_valid_geolocation(xr_obj)
        # Loop over valid slices and retrieve the final slices
        list_slices = []
        new_args = list(args)
        for slc in list_slices_valid:
            # Retrieve slice offset
            start_offset = slc.start
            # Retrieve dataset subset
            subset_xr_obj = xr_obj.isel(along_track=slc)
            # Update args
            new_args[0] = subset_xr_obj
            # Apply function
            subset_slices = function(*args, **kwargs)
            # Add start offset to subset slices
            if len(subset_slices) > 0:
                subset_slices = [
                    slice(slc.start + start_offset, slc.stop + start_offset)
                    for slc in subset_slices
                ]
                list_slices.append(subset_slices)
        # Flatten the list
        list_slices = list_slices_flatten(list_slices)
        # Return list of slices
        return list_slices

    return wrapper


####--------------------------------------------------------------------------.
############################
#### Non-wobbling orbit ####
############################


def _replace_0_values(x):
    """Replace 0 values with previous left non-zero occurring values.

    If the array start with 0, it take the first non-zero occurring values
    """
    # Check inputs
    x = np.array(x)
    dtype = x.dtype
    if np.all(x == 0):
        raise ValueError("It's flat swath orbit.")
    # Set 0 to NaN
    x = x.astype(float)
    x[x == 0] = np.nan
    # Infill from left values, and then from right (if x start with 0)
    x = pd.Series(x).fillna(method="ffill").fillna(method="bfill").to_numpy()
    # Reset original dtype
    x = x.astype(dtype)
    return x


def _get_non_wobbling_lats(lats, threshold=100):
    from gpm_api.utils.slices import list_slices_filter, list_slices_simplify

    # Get direction (1 ascending , -1 descending)
    directions = np.sign(np.diff(lats))
    directions = np.append(directions[0], directions)  # include startpoint
    directions = _replace_0_values(directions)

    # Identify index where next element change
    idxs_change = np.where(np.diff(directions) != 0)[0]

    # Retrieve valid slices
    indices = np.unique(np.concatenate(([0], idxs_change, [len(lats) - 1])))
    orbit_slices = [slice(indices[i], indices[i + 1] + 1) for i in range(len(indices) - 1)]
    list_slices = list_slices_filter(orbit_slices, min_size=threshold)
    list_slices = list_slices_simplify(list_slices)
    return list_slices


@apply_on_valid_geolocation
def get_slices_non_wobbling_swath(xr_obj, threshold=100):
    """Return the along-track slices along which the swath is not wobbling.

    For wobbling, we define the occurrence of changes in latitude directions
    in less than `threshold` scans.
    The function extract the along-track boundary on both swath sides and
    identify where the change in orbit direction occurs.
    """

    from gpm_api.utils.slices import list_slices_intersection

    # Assume lats, lons having shape (y, x) with x=along_track direction
    swath_def = xr_obj.gpm_api.pyresample_area
    lats_side0 = swath_def.lats[0, :]
    lats_side2 = swath_def.lats[-1, :]
    # Get valid slices
    list_slices1 = _get_non_wobbling_lats(lats_side0, threshold=100)
    list_slices2 = _get_non_wobbling_lats(lats_side2, threshold=100)
    list_slices = list_slices_intersection(list_slices1, list_slices2)
    return list_slices


@apply_on_valid_geolocation
def get_slices_wobbling_swath(xr_obj, threshold=100):
    """Return the along-track slices along which the swath is wobbling.

    For wobbling, we define the occurrence of changes in latitude directions
    in less than `threshold` scans.
    The function extract the along-track boundary on both swath sides and
    identify where the change in orbit direction occurs.
    """
    # TODO: this has not been well checked...likely need +1 somewhere ...
    list_slices1 = get_slices_non_wobbling_swath(xr_obj, threshold=threshold)
    list_slices_full = [slice(0, len(xr_obj["along_track"]))]
    list_slices = list_slices_difference(list_slices_full, list_slices1)
    return list_slices


####---------------------------------------------------------------------------
#####################################
#### Check GPM object regularity ####
#####################################


def is_regular(xr_obj):
    """Checks the GPM object is regular.

    For GPM ORBITS, it checks that the scans are contiguous.
    For GPM GRID, it checks that the timesteps are regularly spaced.
    """
    from gpm_api.checks import is_grid, is_orbit

    if is_orbit(xr_obj):
        return has_contiguous_scans(xr_obj)
    elif is_grid(xr_obj):
        return has_regular_time(xr_obj)
    else:
        raise ValueError("Unrecognized GPM xarray object.")


def get_slices_regular(xr_obj, min_size=None):
    """
    Return a list of slices to select regular GPM objects.

    For GPM ORBITS, it returns slices to select contiguouse scans with valid geolocation.
    For GPM GRID, it returns slices to select periods with regular timesteps.

    The output format is: [slice(start,stop), slice(start,stop),...]
    For more information, read the documentation of:
    - gpm_api.utils.checks.get_slices_contiguous_scans
    - gpm_api.utils.checks.get_slices_regular_time

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
    from gpm_api.checks import is_grid, is_orbit

    if is_orbit(xr_obj):
        min_size = 2 if min_size is None else min_size
        # Get swath portions where there are not missing scans (and granules)
        list_slices_contiguous = get_slices_contiguous_scans(xr_obj, min_size=min_size)
        # Get swath portions where there are valid geolocation
        list_slices_geolocation = get_slices_valid_geolocation(xr_obj, min_size=min_size)
        # Find swath portions meeting all the requirements
        list_slices = list_slices_intersection(list_slices_geolocation, list_slices_contiguous)

        return list_slices
    elif is_grid(xr_obj):
        min_size = 1 if min_size is None else min_size
        return get_slices_regular_time(xr_obj, min_size=min_size)
    else:
        raise ValueError("Unrecognized GPM xarray object.")


####--------------------------------------------------------------------------.
#### Get slices from GPM object variable values


def _check_criteria(criteria):
    if criteria not in ["all", "any"]:
        raise ValueError("Invalid value for criteria parameter. Must be 'all' or 'any'.")


def _get_slices_variable_equal_value(da, value, dim=None, criteria="all"):
    """Return the slices with contiguous `value` in the `dim` dimension."""
    # Check dim
    if isinstance(dim, str):
        dim = [dim]
    dim = set(dim)
    # Retrieve dims
    dims = set(da.dims)
    # Identify regions where the value occurs
    da_bool = da == value
    # Collapse other dimension than dim
    dims_apply_over = list(dims.difference(dim))
    bool_arr = (
        da_bool.all(dim=dims_apply_over).data
        if criteria == "all"
        else da_bool.all(dim=dims_apply_over).data
    )
    # Get list of slices with contiguous value
    list_slices = get_list_slices_from_bool_arr(
        bool_arr, include_false=False, skip_consecutive_false=True
    )
    return list_slices


def get_slices_var_equals(da, dim, values, union=True, criteria="all"):
    """Return a list of slices along the `dim` dimension where `values` occurs.

    The function is applied recursively to each value in `values`.
    If the DataArray has additional dimensions, the "criteria" parameter is
    used to determine whether all values within each slice index must be equal
    to value (if set to "all") or if at least one value within the slice index
    must be equal to value (if set to "any").

    If `values` are a list of values:
    - if union=True, it return slices corresponding to the sequence of consecutive values.
    - if union=False, it return slices for each value in values.

    If union=False [0,0, 1, 1] with values=[0,1] will return [slice(0,2), slice(2,4)]
    If union=True [0,0, 1, 1] with values=[0,1] will return [slice(0,4)]

    `union` matters when multiple values are specified
    `criteria` matters when the DataArray has multiple dimensions.
    """
    _check_criteria(criteria)
    # Get data array and dimensions
    da = da.compute()
    # Retrieve the slices where the value(s) occur
    if isinstance(values, (float, int)):
        return _get_slices_variable_equal_value(da=da, value=values, dim=dim, criteria=criteria)
    else:
        list_of_list_slices = [
            _get_slices_variable_equal_value(da=da, value=value, dim=dim, criteria=criteria)
            for value in values
        ]
        list_slices = (
            list_slices_union(*list_of_list_slices)
            if union
            else list_slices_sort(*list_of_list_slices)
        )
    return list_slices


def get_slices_var_between(da, dim, vmin=-np.inf, vmax=np.inf, criteria="all"):
    """Return a list of slices along the `dim` dimension where values are between the interval.

    If the DataArray has additional dimensions, the "criteria" parameter is
    used to determine whether all values within each slice index must be between
    the interval (if set to "all") or if at least one value within the slice index
    must be between the interval (if set to "any").

    """
    _check_criteria(criteria)
    # Check dim
    if isinstance(dim, str):
        dim = [dim]
    dim = set(dim)
    # Get data array and dimensions
    da = da.compute()
    # Retrieve dims
    dims = set(da.dims)
    # Identify regions where the value are between the thresholds
    da_bool = np.logical_and(da >= vmin, da <= vmax)
    # Collapse other dimension than dim
    dims_apply_over = list(dims.difference(dim))
    bool_arr = (
        da_bool.all(dim=dims_apply_over).data
        if criteria == "all"
        else da_bool.any(dim=dims_apply_over).data
    )
    # Get list of slices with contiguous value
    list_slices = get_list_slices_from_bool_arr(
        bool_arr, include_false=False, skip_consecutive_false=True
    )
    return list_slices
