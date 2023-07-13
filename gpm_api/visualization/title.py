#!/usr/bin/env python3
"""
Created on Sat Dec 10 18:50:02 2022

@author: ghiggi
"""
import numpy as np


def get_time_str(timesteps, time_idx=None, resolution="m", timezone="UTC"):
    """Return the time string from a single timestep or array of timesteps.

    If an array, it takes the timesteps in the middle of the array.
    """

    if timesteps.size == 1:
        timestep = timesteps
    else:
        if time_idx is None:
            timestep = timesteps[int(len(timesteps) / 2)]
        else:
            timestep = timesteps[time_idx]
    # Get time string with custom unit and timezone
    time_str = np.datetime_as_string(timestep, unit=resolution, timezone=timezone)
    time_str = time_str.replace("T", " ").replace("Z", "")
    # Return time string
    return time_str


def get_dataset_title(
    ds,
    add_timestep=True,
    time_idx=None,
    resolution="m",
    timezone="UTC",
):
    """
    Generate the GPM Dataset title.

    Parameters
    ----------
    ds : xr.Dataset
        GPM xarray Dataset.
    add_timestep : bool, optional
        Whether to add time information to the title. The default is True.
        For GRID objects (like IMERG), the timestep is added only if
        the DataArray has 1 timestep.
    time_idx : int, optional
        Index of timestep to select, instead of selecting the middle.
        The default is None.
    resolution : str, optional
        The maximum temporal resolution to display.
        The default is "m" (for minutes).
    timezone : str, optional
        The timezone of the time to display.
        The default is "UTC" (as the reference timezone of the dataset).

    Returns
    -------
    title_str : str
        Title of the Dataset.

    """
    from gpm_api.checks import is_orbit

    # Get GPM product name
    product = ds.attrs.get("gpm_api_product", "")
    title_str = product

    # Make title in Capital Case
    title_str = " ".join([word[0].upper() + word[1:] for word in title_str.split(" ")])

    # Add time
    if add_timestep and (is_orbit(ds) or ds["time"].size == 1):
        time_str = get_time_str(
            timesteps=ds["time"].data,
            time_idx=time_idx,
            resolution=resolution,
            timezone=timezone,
        )
        title_str = title_str + " (" + time_str + ")"
    return title_str


def get_dataarray_title(
    da,
    prefix_product=True,
    add_timestep=True,
    time_idx=None,
    resolution="m",
    timezone="UTC",
):
    """
    Generate the GPM xarray DataArray title.

    Parameters
    ----------
    da : xr.DataArray
        GPM xarray DataArray.
    prefix_product : bool, optional
        Whether to add the GPM product as prefix.
        The default is True.
    add_timestep : bool, optional
        Whether to add time information to the title. The default is True.
        For GRID objects (like IMERG), the timestep is added only if
        the DataArray has 1 timestep.
    time_idx : int, optional
        Index of timestep to select, instead of selecting the middle.
        The default is None.
    resolution : str, optional
        The maximum temporal resolution to display.
        The default is "m" (for minutes).
    timezone : str, optional
        The timezone of the time to display.
        The default is "UTC" (as the reference timezone of the dataset).

    Returns
    -------
    title_str : str
        Title of the DataArray.

    """
    from gpm_api.checks import is_orbit

    # Get variable name
    variable = da.name
    # Get product name
    product = da.attrs.get("gpm_api_product", "")

    # Create title string
    title_str = product + " " + variable if prefix_product else da.name

    # Make title in Capital Case
    title_str = " ".join([word[0].upper() + word[1:] for word in title_str.split(" ")])

    # Add time
    if add_timestep and (is_orbit(da) or da["time"].size == 1):
        time_str = get_time_str(
            timesteps=da["time"].data,
            time_idx=time_idx,
            resolution=resolution,
            timezone=timezone,
        )
        title_str = title_str + " (" + time_str + ")"
    return title_str
