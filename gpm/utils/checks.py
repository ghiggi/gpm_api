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
"""This module contains utilities to check GPM-API Dataset coordinates."""
import functools

import numpy as np
import pandas as pd

from gpm.checks import is_grid, is_orbit
from gpm.utils.decorators import (
    check_has_along_track_dimension,
    check_has_cross_track_dimension,
    check_is_gpm_object,
    check_is_orbit,
)
from gpm.utils.slices import (
    get_list_slices_from_bool_arr,
    list_slices_difference,
    list_slices_filter,
    list_slices_flatten,
    list_slices_intersection,
    list_slices_simplify,
    list_slices_sort,
    list_slices_union,
)

ORBIT_TIME_TOLERANCE = np.timedelta64(5, "s")

# TODO: maybe infer from time? <3 s for DPR, up to 5s per old PMW

####--------------------------------------------------------------------------.
##########################
#### Regular granules ####
##########################


def get_missing_granule_numbers(xr_obj):
    """Return ID numbers of missing granules.

    It assumes xr_obj is a GPM ORBIT object.
    """
    granule_ids = np.unique(xr_obj["gpm_granule_id"].data)
    min_id = min(granule_ids)
    max_id = max(granule_ids)
    possible_ids = np.arange(min_id, max_id + 1)
    return possible_ids[~np.isin(possible_ids, granule_ids)]


def _is_contiguous_granule(granule_ids):
    """Return a boolean array indicating if the next scan is not the same/next granule."""
    # Retrieve if next scan is in the same or next granule (True) or False.
    bool_arr = np.diff(granule_ids) <= 1

    # Add True to last position
    return np.append(bool_arr, True)


@check_is_gpm_object
def get_slices_contiguous_granules(xr_obj, min_size=2):
    """Return a list of slices ensuring contiguous granules.

    The minimum size of the output slices is 2.

    Note: for GRID (i.e. IMERG) products, it checks for regular timesteps !
    Note: No granule_id is provided for GRID products.

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
        Output format: [slice(start,stop), slice(start,stop),...]

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
            return []

        # Get boolean array indicating if the next scan/timesteps is same or next granule
        bool_arr = _is_contiguous_granule(xr_obj["gpm_granule_id"].data)

        # If granules are missing present, get the slices with non-missing granules
        list_slices = get_list_slices_from_bool_arr(
            bool_arr,
            include_false=True,
            skip_consecutive_false=True,
        )

        # Select only slices with at least 2 scans
        return list_slices_filter(list_slices, min_size=min_size)
    return None

    # Return list of contiguous scan slices


def check_missing_granules(xr_obj):
    """Check no missing granules in the GPM Dataset.

    It assumes xr_obj is a GPM ORBIT object.

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
    """Check no missing granules in the GPM Dataset.

    It assumes xr_obj is a GPM ORBIT object.

    Parameters
    ----------
    xr_obj : xr.Dataset or xr.DataArray
        xarray object.

    """
    return check_missing_granules(xr_obj)


@check_is_gpm_object
def has_contiguous_granules(xr_obj):
    """Checks GPM object is composed of consecutive granules.

    For ORBIT objects, it checks the gpm_granule_id.
    For GRID objects, it checks timesteps regularity.
    """
    # ORBIT
    if is_orbit(xr_obj):
        return bool(np.all(_is_contiguous_granule(xr_obj["gpm_granule_id"].data)))
    # if is_grid(xr_obj):
    return has_regular_time(xr_obj)


@check_is_gpm_object
def has_missing_granules(xr_obj):
    """Checks GPM object has missing granules.

    For ORBIT objects, it checks the gpm_granule_id.
    For GRID objects, it checks timesteps regularity.
    """
    # ORBIT
    if is_orbit(xr_obj):
        return bool(np.any(~_is_contiguous_granule(xr_obj["gpm_granule_id"].data)))
    # if is_grid(xr_obj):
    return not has_regular_time(xr_obj)


####--------------------------------------------------------------------------.
############################
#### Regular timesteps  ####
############################


def _get_timesteps(xr_obj):
    """Get timesteps with second precision from xarray object."""
    timesteps = xr_obj["time"].to_numpy()
    return timesteps.astype("M8[s]")


@check_is_gpm_object
def _infer_time_tolerance(xr_obj):
    """Infer time interval tolerance between timesteps."""
    # For GPM ORBIT objects, use the ORBIT_TIME_TOLERANCE
    if is_orbit(xr_obj):
        tolerance = ORBIT_TIME_TOLERANCE
    # For GPM GRID objects, infer it from the time difference between first two timesteps
    elif is_grid(xr_obj):
        timesteps = _get_timesteps(xr_obj)
        tolerance = np.diff(timesteps[0:2])[0]
    return tolerance


def _is_regular_time(xr_obj, tolerance=None):
    """Return a boolean array indicating if the next regular timestep is present."""
    # Retrieve timesteps
    timesteps = _get_timesteps(xr_obj)
    # Infer tolerance if not specified
    tolerance = _infer_time_tolerance(xr_obj) if tolerance is None else tolerance
    # Identify if the next regular timestep is present
    bool_arr = np.diff(timesteps) <= tolerance
    # Add True to last position
    return np.append(bool_arr, True)


def get_slices_regular_time(xr_obj, tolerance=None, min_size=1):
    """Return a list of slices ensuring timesteps to be regular.

    Output format: [slice(start,stop), slice(start,stop),...]

    Consecutive non-regular timesteps leads to slices of size 1.
    An xarray object with a single timestep leads to a slice of size 1.
    If min_size=1 (the default), such slices are returned.

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM xarray object.
    tolerance : np.timedelta, optional
        The timedelta tolerance to define regular vs. non-regular timesteps.
        The default is ``None``.
        If GPM GRID object, it uses the first 2 timesteps to derive the tolerance timedelta.
        If GPM ORBIT object, it uses the ORBIT_TIME_TOLERANCE.
    min_size : int
        Minimum size for a slice to be returned.

    Returns
    -------
    list_slices : list
        List of slice object to select regular time intervals.
        Output format: [slice(start,stop), slice(start,stop),...]

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
            is_regular,
            include_false=True,
            skip_consecutive_false=False,
        )

    # Select only slices with at least min_size timesteps
    return list_slices_filter(list_slices, min_size=min_size)

    # Return list of slices with regular timesteps


def get_slices_non_regular_time(xr_obj, tolerance=None):
    """Return a list of slices where there are supposedly missing timesteps.

    The output slices have size 2.
    An input with less than 2 scans (along-track) returns an empty list.

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM xarray object.
    tolerance : np.timedelta, optional
        The timedelta tolerance to define regular vs. non-regular timesteps.
        The default is ``None``.
        If GPM GRID object, it uses the first 2 timesteps to derive the tolerance timedelta.
        If GPM ORBIT object, it uses the ORBIT_TIME_TOLERANCE.
        It is discouraged to use this function for GPM ORBIT objects !

    Returns
    -------
    list_slices : list
        List of slice object to select intervals with non-regular timesteps.
        Output format: [slice(start,stop), slice(start,stop),...]

    """
    # Retrieve timestep
    timesteps = _get_timesteps(xr_obj)
    n_timesteps = len(timesteps)

    # Define behaviour if less than 2 timesteps
    # --> Here we decide to return an empty list !
    if n_timesteps < 2:
        return []

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
    """Check no missing timesteps for longer than 'tolerance' seconds.

    Note:
    - This sometimes occurs between orbit/grid granules
    - This sometimes occurs within a orbit granule

    Parameters
    ----------
    xr_obj : xr.Dataset or xr.DataArray
        xarray object.
    tolerance : np.timedelta, optional
        The timedelta tolerance to define regular vs. non-regular timesteps.
        The default is ``None``.
        If GPM GRID object, it uses the first 2 timesteps to derive the tolerance timedelta.
        If GPM ORBIT object, it uses the ORBIT_TIME_TOLERANCE
    verbose : bool
        If True, it prints the time interval when the non contiguous scans occurs.
        The default is ``True``.

    """
    list_discontinuous_slices = get_slices_non_regular_time(xr_obj, tolerance=tolerance)
    n_discontinuous = len(list_discontinuous_slices)
    if n_discontinuous > 0:
        # Retrieve discontinuous timesteps interval
        timesteps = _get_timesteps(xr_obj)
        list_discontinuous = [(timesteps[slc.start], timesteps[slc.stop - 1]) for slc in list_discontinuous_slices]
        first_problematic_timestep = list_discontinuous[0][0]
        # Print non-regular timesteps
        if verbose:
            for start, stop in list_discontinuous:
                print(f"- Missing timesteps between {start} and {stop}")
        # Raise error and highlight first non-contiguous scan
        raise ValueError(
            f"There are {n_discontinuous} non-regular timesteps. The first occur at {first_problematic_timestep}.",
        )


def has_regular_time(xr_obj):
    """Return True if all timesteps are regular. False otherwise."""
    list_discontinuous_slices = get_slices_non_regular_time(xr_obj)
    n_discontinuous = len(list_discontinuous_slices)
    if n_discontinuous > 0:
        return False
    return True


####--------------------------------------------------------------------------.
########################
#### Regular scans  ####
########################


def _select_lons_lats_centroids(xr_obj):
    if "cross_track" not in xr_obj.dims:
        lons = xr_obj["lon"].to_numpy()
        lats = xr_obj["lat"].to_numpy()
    else:
        # Select centroids coordinates in the middle of the cross_track scan
        middle_idx = int(xr_obj["cross_track"].shape[0] / 2)
        lons = xr_obj["lon"].isel(cross_track=middle_idx).to_numpy()
        lats = xr_obj["lat"].isel(cross_track=middle_idx).to_numpy()
    return lons, lats


def _get_along_track_scan_distance(xr_obj):
    """Compute the distance between along_track centroids."""
    from pyproj import Geod

    # Select centroids coordinates
    # - If no cross-track, take the availbles lat/lon 1D array
    # - If cross-track dimension is present, takes the coordinates in the swath middle.
    lons, lats = _select_lons_lats_centroids(xr_obj)

    # Define between-centroids line coordinates
    start_lons = lons[:-1]
    start_lats = lats[:-1]
    end_lons = lons[1:]
    end_lats = lats[1:]

    # Compute distance
    geod = Geod(ellps="sphere")
    _, _, dist = geod.inv(start_lons, start_lats, end_lons, end_lats)
    return dist


def _is_contiguous_scans(xr_obj):
    """Return a boolean array indicating if the next scan is contiguous.

    It assumes at least 3 scans are provided.

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
    min_dist = np.nanmin(dist_km)
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
    return np.append(bool_arr, True)


@check_is_orbit
@check_has_along_track_dimension
def get_slices_contiguous_scans(xr_obj, min_size=2, min_n_scans=3):
    """Return a list of slices ensuring contiguous scans (and granules).

    It checks for contiguous scans only in the middle of the cross-track !
    If a scan geolocation is NaN, it will be considered non-contiguous.

    An input with less than 3 scans (along-track) returns an empty list, since
    scan contiguity can't be verified.
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
        Output format: [slice(start,stop), slice(start,stop),...]

    """
    # Get number of scans
    n_scans = xr_obj["along_track"].shape[0]

    # Define behaviour if less than 2/3 scan along track
    # --> Contiguity can't be verified without at least 3 slices !
    # --> But for visualization purpose, if only 2 scans available, we want to plot it (and consider it contiguous)
    # --> Here we decide to return an empty list !
    if n_scans < min_n_scans:
        return []

    # Get boolean array indicating if the next scan is contiguous
    is_contiguous = _is_contiguous_scans(xr_obj)

    # If non-contiguous scans are present, get the slices with contiguous scans
    # - It discard consecutive non-contiguous scans
    list_slices = get_list_slices_from_bool_arr(
        is_contiguous,
        include_false=True,
        skip_consecutive_false=True,
    )

    # Select only slices with at least 2 scans
    list_slices = list_slices_filter(list_slices, min_size=min_size)

    # Also retrieve the slices with non missing granule
    list_slices1 = get_slices_contiguous_granules(xr_obj)

    # Perform list_slices intersection
    return list_slices_intersection(list_slices, list_slices1)

    # Return list of contiguous scan slices


@check_is_orbit
@check_has_along_track_dimension
def get_slices_non_contiguous_scans(xr_obj):
    """Return a list of slices where the scans discontinuity occurs.

    An input with less than 2 scans (along-track) returns an empty list.

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM xarray object.

    Returns
    -------
    list_slices : list
        List of slice object to select discontiguous scans.
        Output format: [slice(start,stop), slice(start,stop),...]

    """
    # Get number of scans
    n_scans = xr_obj["along_track"].shape[0]

    # Define behaviour if less than 3 scan along track
    # --> Contiguity can't be verified without at least 3 slices !
    # --> Here we decide to return an empty list !
    if n_scans < 3:
        return []

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
    return list_slices_difference(list_slices_full, list_slices_valid)


@check_has_cross_track_dimension
def check_contiguous_scans(xr_obj, verbose=True):
    """Check no missing scans across the along_track direction.

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
        list_discontinuous = [(timesteps[slc.start], timesteps[slc.stop - 1]) for slc in list_discontinuous_slices]
        first_problematic_timestep = list_discontinuous[0][0]
        # Print non-contiguous scans
        if verbose:
            for start, stop in list_discontinuous:
                print(f"- Missing scans between {start} and {stop}")
        # Raise error and highlight first non-contiguous scan
        msg = f"There are {n_discontinuous} non-contiguous scans. "
        msg += f"The first occur at {first_problematic_timestep}. "
        msg += "Use get_slices_contiguous_scans() to retrieve a list of contiguous slices."
        raise ValueError(msg)


@check_has_cross_track_dimension
def has_contiguous_scans(xr_obj):
    """Return True if all scans are contiguous. False otherwise."""
    list_discontinuous_slices = get_slices_non_contiguous_scans(xr_obj)
    n_discontinuous = len(list_discontinuous_slices)
    if n_discontinuous > 0:
        return False
    return True


####--------------------------------------------------------------------------.
#############################
#### Regular geolocation ####
#############################


def _is_non_valid_geolocation(xr_obj, x="lon"):
    """Return a boolean array indicating if the geolocation is invalid.

    True = Invalid, False = Valid
    """
    return np.isnan(xr_obj[x])


def _is_valid_geolocation(xr_obj, x="lon"):
    """Return a boolean array indicating if the geolocation is valid.

    True = Valid, False = Invalid
    """
    return ~np.isnan(xr_obj[x])


@check_is_orbit
def get_slices_valid_geolocation(xr_obj, min_size=2):
    """Return a list of GPM ORBIT along-track slices with valid geolocation.

    The minimum size of the output slices is 2.

    If at a given cross-track index, there are always wrong geolocation,
    it discards such cross-track index(es) before identifying the along-track slices.

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM ORBIT xarray object.
    min_size : int
        Minimum size for a slice to be returned.
        The default is 2.

    Returns
    -------
    list_slices : list
        List of slice object with valid geolocation.
        Output format: [slice(start,stop), slice(start,stop),...]

    """
    # - Get invalid coordinates
    invalid_lon_coords = _is_non_valid_geolocation(xr_obj, x="lon")
    invalid_lat_coords = _is_non_valid_geolocation(xr_obj, x="lat")
    invalid_coords = np.logical_or(invalid_lon_coords, invalid_lat_coords)

    # - Identify cross-track index that along-track are always invalid
    idx_cross_track_not_all_invalid = np.where(~invalid_coords.all("along_track"))[0]
    # - If all invalid, return empty list
    if len(idx_cross_track_not_all_invalid) == 0:
        return []
    # - Select only cross-track index that are not all invalid along-track
    invalid_coords = invalid_coords.isel(cross_track=idx_cross_track_not_all_invalid)
    # - Now identify scans across which there are still invalid coordinates
    invalid_scans = invalid_coords.any(dim="cross_track")
    valid_scans = ~invalid_scans
    # - Now identify valid along-track slices
    list_slices = get_list_slices_from_bool_arr(
        valid_scans,
        include_false=False,
        skip_consecutive_false=True,
    )
    # Select only slices with at least 2 scans
    return list_slices_filter(list_slices, min_size=min_size)


def get_slices_non_valid_geolocation(xr_obj):
    """Return a list of GPM ORBIT along-track slices with non-valid geolocation.

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
        Output format: [slice(start,stop), slice(start,stop),...]

    """
    list_slices_valid = get_slices_valid_geolocation(xr_obj, min_size=1)
    list_slices_full = [slice(0, len(xr_obj["along_track"]))]
    return list_slices_difference(list_slices_full, list_slices_valid)


def check_valid_geolocation(xr_obj, verbose=True):
    """Check no geolocation errors in the GPM Dataset.

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


@check_is_gpm_object
def has_valid_geolocation(xr_obj):
    """Checks GPM object has valid geolocation."""
    if is_orbit(xr_obj):
        list_invalid_slices = get_slices_non_valid_geolocation(xr_obj)
        n_invalid_scan_slices = len(list_invalid_slices)
        return n_invalid_scan_slices == 0
    if is_grid(xr_obj):
        return True
    return None


def apply_on_valid_geolocation(function):
    """Decorator appliying the input function on valid geolocation GPM ORBIT slices."""

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
                subset_slices = [slice(slc.start + start_offset, slc.stop + start_offset) for slc in subset_slices]
                list_slices.append(subset_slices)
        # Flatten the list
        return list_slices_flatten(list_slices)
        # Return list of slices

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
        raise ValueError("It's a flat swath orbit.")
    # Set 0 to NaN
    x = x.astype(float)
    x[x == 0] = np.nan
    # Infill from left values, and then from right (if x start with 0)
    x = pd.Series(x).ffill().bfill().to_numpy()
    # Reset original dtype
    return x.astype(dtype)


def _get_non_wobbling_lats(lats, threshold=100):
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
    return list_slices_simplify(list_slices)


@apply_on_valid_geolocation
def get_slices_non_wobbling_swath(xr_obj, threshold=100):
    """Return the GPM ORBIT along-track slices along which the swath is not wobbling.

    For wobbling, we define the occurrence of changes in latitude directions
    in less than `threshold` scans.
    The function extract the along-track boundary on both swath sides and
    identify where the change in orbit direction occurs.
    """
    xr_obj = xr_obj.transpose("cross_track", "along_track", ...)
    lats = xr_obj["lat"].to_numpy()
    lats_side0 = lats[0, :]
    lats_side2 = lats[-1, :]
    # Get valid slices
    list_slices1 = _get_non_wobbling_lats(lats_side0, threshold=threshold)
    list_slices2 = _get_non_wobbling_lats(lats_side2, threshold=threshold)
    return list_slices_intersection(list_slices1, list_slices2)


@apply_on_valid_geolocation
def get_slices_wobbling_swath(xr_obj, threshold=100):
    """Return the GPM ORBIT along-track slices along which the swath is wobbling.

    For wobbling, we define the occurrence of changes in latitude directions
    in less than `threshold` scans.
    The function extract the along-track boundary on both swath sides and
    identify where the change in orbit direction occurs.
    """
    # TODO: this has not been well checked...likely need +1 somewhere ...
    list_slices1 = get_slices_non_wobbling_swath(xr_obj, threshold=threshold)
    list_slices_full = [slice(0, len(xr_obj["along_track"]))]
    return list_slices_difference(list_slices_full, list_slices1)


####---------------------------------------------------------------------------
#####################################
#### Check GPM object regularity ####
#####################################


@check_is_gpm_object
def is_regular(xr_obj):
    """Checks the GPM object is regular.

    For GPM ORBITS, it checks that the scans are contiguous.
    For GPM GRID, it checks that the timesteps are regularly spaced.
    """
    if is_orbit(xr_obj):
        return has_contiguous_scans(xr_obj)
    if is_grid(xr_obj):
        return has_regular_time(xr_obj)
    return None


@check_is_gpm_object
def get_slices_regular(xr_obj, min_size=None, min_n_scans=3):
    """Return a list of slices to select regular GPM objects.

    For GPM ORBITS, it returns slices to select contiguous scans with valid geolocation.
    For GPM GRID, it returns slices to select periods with regular timesteps.

    For more information, read the documentation of:
    - gpm.utils.checks.get_slices_contiguous_scans
    - gpm.utils.checks.get_slices_regular_time

    Parameters
    ----------
    xr_obj : (xr.Dataset, xr.DataArray)
        GPM xarray object.
    min_size : int
        Minimum size for a slice to be returned.
        If ``None``, default to 1 for GRID objects, 2 for ORBIT objects.
    min_n_scans : int
        Minimum number of scans to be able to check for scan contiguity.
        For visualization purpose, this value might want to be set to 2.
        This parameter applies only to ORBIT objects.

    Returns
    -------
    list_slices : list
        List of slice object to select regular portions.
        Output format: [slice(start,stop), slice(start,stop),...]

    """
    if is_orbit(xr_obj):
        min_size = 2 if min_size is None else min_size
        # Get swath portions where there are not missing scans (and granules)
        list_slices_contiguous = get_slices_contiguous_scans(
            xr_obj,
            min_size=min_size,
            min_n_scans=min_n_scans,
        )
        # Get swath portions where there are valid geolocation
        list_slices_geolocation = get_slices_valid_geolocation(xr_obj, min_size=min_size)
        # Find swath portions meeting all the requirements
        return list_slices_intersection(list_slices_geolocation, list_slices_contiguous)

    if is_grid(xr_obj):
        min_size = 1 if min_size is None else min_size
        return get_slices_regular_time(xr_obj, min_size=min_size)
    return None


####--------------------------------------------------------------------------.
#### Get slices from GPM object variable values


def _check_criteria(criteria):
    if criteria not in ["all", "any"]:
        raise ValueError("Invalid value for criteria parameter. Must be 'all' or 'any'.")
    return criteria


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
    bool_arr = da_bool.all(dim=dims_apply_over).data if criteria == "all" else da_bool.any(dim=dims_apply_over).data
    # Get list of slices with contiguous value
    return get_list_slices_from_bool_arr(bool_arr, include_false=False, skip_consecutive_false=True)


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
    criteria = _check_criteria(criteria)
    # Get data array and dimensions
    da = da.compute()
    # Retrieve the slices where the value(s) occur
    if isinstance(values, (float, int)):
        return _get_slices_variable_equal_value(da=da, value=values, dim=dim, criteria=criteria)
    # If multiple values, apply recursively
    list_of_list_slices = [
        _get_slices_variable_equal_value(da=da, value=value, dim=dim, criteria=criteria) for value in values
    ]
    # Return list_slices
    return list_slices_union(*list_of_list_slices) if union else list_slices_sort(*list_of_list_slices)


def get_slices_var_between(da, dim, vmin=-np.inf, vmax=np.inf, criteria="all"):
    """Return a list of slices along the `dim` dimension where values are between the interval.

    If the DataArray has additional dimensions, the "criteria" parameter is
    used to determine whether all values within each slice index must be between
    the interval (if set to "all") or if at least one value within the slice index
    must be between the interval (if set to "any").

    """
    criteria = _check_criteria(criteria)
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
    bool_arr = da_bool.all(dim=dims_apply_over).data if criteria == "all" else da_bool.any(dim=dims_apply_over).data
    # Get list of slices with contiguous value
    return get_list_slices_from_bool_arr(bool_arr, include_false=False, skip_consecutive_false=True)
