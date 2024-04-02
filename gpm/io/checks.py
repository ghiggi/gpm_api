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
"""This module contains functions to check the GPM-API arguments."""
import datetime
import os
import subprocess

import numpy as np


def check_base_dir(base_dir):
    """Check base directory path.

    If base_dir ends with "GPM" directory, it removes it from the base_dir path.
    If base_dir does not end with "GPM", the GPM directory will be created.

    Parameters
    ----------
    base_dir : str
        Base directory where the GPM directory is located.

    Returns
    -------
    base_dir: str
        Base directory where the GPM directory is located.

    """
    base_dir = str(base_dir)  # deal with PathLib path
    # Check base_dir does not end with /
    if base_dir[-1] == os.path.sep:
        base_dir = base_dir[0:-1]
    # Retrieve last folder name
    dir_name = os.path.basename(base_dir)
    # If ends with GPM, take the parent directory path
    if dir_name == "GPM":
        base_dir = os.path.dirname(base_dir)
    return base_dir


def check_filepaths(filepaths):
    """Ensure filepaths is a list of string."""
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    if not isinstance(filepaths, list):
        raise TypeError("Expecting a list of filepaths.")
    return filepaths


def check_variables(variables):
    """Ensure variables is a numpy array of string."""
    if not isinstance(variables, (str, list, np.ndarray, type(None))):
        raise TypeError("'variables' must be a either a str, list, np.ndarray or None.")
    if variables is None:
        return None
    if isinstance(variables, str):
        variables = [variables]
    elif isinstance(variables, list):
        variables = np.array(variables)
    return variables


def check_groups(groups):
    """Ensure groups is a numpy array of string."""
    if not isinstance(groups, (str, list, np.ndarray, type(None))):
        raise TypeError("'groups' must be a either a str, list, np.ndarray or None.")
    if isinstance(groups, str):
        groups = [groups]
    elif isinstance(groups, list):
        groups = np.array(groups)
    return groups


def check_storage(storage):
    """Check storage."""
    if not isinstance(storage, str):
        raise TypeError("'storage' must be a string.")
    valid_storages = ["GES_DISC", "PPS", "LOCAL"]
    if storage.upper() not in valid_storages:
        raise ValueError(f"{storage} is an invalid 'storage'. Valid values are {valid_storages}.")
    return storage.upper()


def check_remote_storage(storage):
    """Check storage is remote."""
    if not isinstance(storage, str):
        raise TypeError("'storage' must be a string.")
    valid_storages = ["GES_DISC", "PPS"]
    if storage.upper() not in valid_storages:
        raise ValueError(
            f"'{storage}' is an invalid remote 'storage'. Valid values are {valid_storages}.",
        )
    return storage.upper()


def check_transfer_tool_availability(transfer_tool):
    """Check availability of a transfer_tool. Return True if available."""
    try:
        subprocess.run(
            [transfer_tool, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


CURL_IS_AVAILABLE = check_transfer_tool_availability("curl")
WGET_IS_AVAILABLE = check_transfer_tool_availability("wget")


def check_transfer_tool(transfer_tool):
    """Check the transfer tool."""
    valid_transfer_tools = ["CURL", "WGET"]

    if transfer_tool.upper() not in valid_transfer_tools:
        raise ValueError(
            f"'{transfer_tool}' is an invalid 'transfer_tool'. Valid values are {valid_transfer_tools}.",
        )

    # Check WGET or CURL is installed
    transfer_tool = transfer_tool.upper()
    if transfer_tool == "CURL" and not CURL_IS_AVAILABLE:
        raise ValueError("CURL is not installed on your machine !")
    if transfer_tool == "WGET" and not WGET_IS_AVAILABLE:
        raise ValueError("WGET is not installed on your machine !")
    return transfer_tool


def check_product(product, product_type):
    """Check product validity."""
    from gpm.io.products import available_products

    if not isinstance(product, str):
        raise ValueError("'Ask for a single product at time.'product' must be a single string.")
    if product not in available_products(product_types=product_type):
        raise ValueError("Please provide a valid GPM product --> gpm.available_products().")
    return product


def check_product_version(version, product):
    """Check valid version for the specified product."""
    from gpm.io.products import available_product_versions, get_last_product_version

    if version is None:
        version = get_last_product_version(product)
    version = check_version(version)
    # Check valid version for such product
    valid_versions = available_product_versions(product)
    if version not in valid_versions:
        raise ValueError(f"Valid versions for product '{product}' are {valid_versions}.")
    return version


def check_product_validity(product, product_type=None):
    """Check product validity for the specified product_type."""
    from gpm.io.products import available_products  # circular otherwise

    if product not in available_products(product_types=product_type):
        if product_type is None:
            raise ValueError(
                f"The '{product}' product is not available. See gpm.available_products().",
            )
        raise ValueError(
            f"The '{product}' product is not available as '{product_type}' product_type.",
        )
    return product


def check_time(time):
    """Check time validity.

    It returns a datetime.datetime object to seconds precision.

    Parameters
    ----------
    time : (datetime.datetime, datetime.date, np.datetime64, str)
        Time object.
        Accepted types: ``datetime.datetime``, ``datetime.date``, ``np.datetime64`` or ``str``.
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.

    Returns
    -------
    time : datetime.datetime

    """
    if not isinstance(time, (datetime.datetime, datetime.date, np.datetime64, np.ndarray, str)):
        raise TypeError(
            "Specify time with datetime.datetime objects or a " "string of format 'YYYY-MM-DD hh:mm:ss'.",
        )

    # If numpy array with datetime64 (and size=1)
    if isinstance(time, np.ndarray):
        if np.issubdtype(time.dtype, np.datetime64):
            if time.size == 1:
                time = time[0].astype("datetime64[s]").tolist()
            else:
                raise ValueError("Expecting a single timestep!")
        else:
            raise ValueError("The numpy array does not have a np.datetime64 dtype!")

    # If np.datetime64, convert to datetime.datetime
    if isinstance(time, np.datetime64):
        time = time.astype("datetime64[s]").tolist()
    # If datetime.date, convert to datetime.datetime
    if not isinstance(time, (datetime.datetime, str)):
        time = datetime.datetime(time.year, time.month, time.day, 0, 0, 0)
    if isinstance(time, str):
        try:
            time = datetime.datetime.fromisoformat(time)
        except ValueError:
            raise ValueError("The time string must have format 'YYYY-MM-DD hh:mm:ss'")

    # If datetime object carries timezone that is not UTC, raise error
    if time.tzinfo is not None:
        if str(time.tzinfo) != "UTC":
            raise ValueError("The datetime object must be in UTC timezone if timezone is given.")
        # If UTC, strip timezone information
        time = time.replace(tzinfo=None)
    return time


def check_date(date):
    """Check is a datetime.date object."""
    if date is None:
        raise ValueError("date cannot be None")
    # Use check_time to convert to datetime.datetime
    datetime_obj = check_time(date)
    return datetime_obj.date()


def check_start_end_time(start_time, end_time):
    """Check start_time and end_time value validity."""
    start_time = check_time(start_time)
    end_time = check_time(end_time)

    # Check start_time and end_time are chronological
    if start_time > end_time:
        raise ValueError("Provide 'start_time' occurring before of 'end_time'.")
    # Check start_time and end_time are in the past
    if start_time > datetime.datetime.utcnow():
        raise ValueError("Provide a 'start_time' occurring in the past.")
    if end_time > datetime.datetime.utcnow():
        raise ValueError("Provide a 'end_time' occurring in the past.")
    return (start_time, end_time)


def check_valid_time_request(start_time, end_time, product):
    """Check validity of the time request."""
    from gpm.io.products import get_product_end_time, get_product_start_time

    product_start_time = get_product_start_time(product)
    product_end_time = get_product_end_time(product)
    if start_time < product_start_time:
        raise ValueError(f"{product} production started the {product_start_time}.")
    if end_time > product_end_time:
        raise ValueError(f"{product} production ended the {get_product_end_time}.")
    return start_time, end_time


def check_scan_mode(scan_mode, product, version):
    """Checks scan_mode validity."""
    from gpm.io.products import available_scan_modes

    scan_modes = available_scan_modes(product, version)
    # Infer scan mode if not specified
    if scan_mode is None:
        scan_mode = scan_modes[0]
        if len(scan_modes) > 1:
            print(f"'scan_mode' has not been specified. Default to {scan_mode}.")
    # Check that a single scan mode is specified
    if scan_mode is not None and not isinstance(scan_mode, str):
        raise ValueError("Specify a single 'scan_mode'.")
    # Check that a valid scan mode is specified
    if scan_mode is not None and scan_mode not in scan_modes:
        raise ValueError(f"For {product} product, valid 'scan_modes' are {scan_modes}.")
    return scan_mode


#### Single arguments


def check_product_type(product_type):
    """Check product_type validity."""
    from gpm.io.products import get_available_product_types  # circular otherwise

    if not isinstance(product_type, str):
        raise ValueError("Please specify the 'product_type' as a string.")
    valid_values = get_available_product_types()
    if product_type not in valid_values:
        raise ValueError("Please specify the 'product_type' as 'RS' or 'NRT'.")
    return product_type


def check_product_category(product_category):
    """Check product_category validity."""
    from gpm.io.products import get_available_product_categories  # circular otherwise

    if not isinstance(product_category, str):
        raise ValueError("Please specify the 'product_category' as a string.")
    valid_values = get_available_product_categories()  #  ['CMB', 'IMERG', 'PMW', 'RADAR']
    if product_category not in valid_values:
        raise ValueError(
            f"'{product_category}' is an invalid 'product_category'. Valid values are {valid_values}.",
        )
    return product_category


def check_product_level(product_level):
    """Check product_level validity."""
    from gpm.io.products import get_available_product_levels  # circular otherwise

    if not isinstance(product_level, str):
        raise ValueError("Please specify the 'product_level' as a string.")
    valid_values = get_available_product_levels(full=False)
    if product_level not in valid_values:
        raise ValueError(
            f"'{product_level}' is an invalid 'product_level'. Currently accepted values are {valid_values}.",
        )
    return product_level


def check_version(version):
    """Check version validity."""
    from gpm.io.products import get_available_versions  # circular otherwise

    if not isinstance(version, int):
        raise ValueError("Please specify the GPM 'version' with an integer between 5 and 7.")
    valid_values = get_available_versions()
    if version not in valid_values:
        raise ValueError("GPM-API currently supports only GPM versions 5, 6 and 7.")
    return version


def check_full_product_level(full_product_level):
    """Check full_product_level validity."""
    from gpm.io.products import get_available_product_levels  # circular otherwise

    if not isinstance(full_product_level, str):
        raise ValueError("Please specify the full_product_level as a string.")
    valid_values = get_available_product_levels(full=True)
    if full_product_level not in valid_values:
        raise ValueError(
            f"'{full_product_level}' is an invalid 'full_product_level'. Currently accepted values are {valid_values}.",
        )
    return full_product_level


def check_sensor(sensor):
    """Check sensor validity."""
    from gpm.io.products import get_available_sensors

    if not isinstance(sensor, str):
        raise ValueError("Please specify the 'sensor' as a string.")

    valid_sensors = get_available_sensors()
    if sensor not in valid_sensors:
        raise ValueError(
            f"'{sensor}' is not an available 'sensor'. Available sensors are {valid_sensors}.",
        )
    return sensor


def check_satellite(satellite):
    """Check satellite validity."""
    from gpm.io.products import get_available_satellites

    if not isinstance(satellite, str):
        raise ValueError("Please specify the 'satellite' as a string.")

    valid_satellites = get_available_satellites()
    if satellite not in valid_satellites:
        raise ValueError(
            f"'{satellite}' is not an available 'satellite'. Available satellite are {valid_satellites}.",
        )
    return satellite


#### List arguments
def check_sensors(sensors):
    """Check sensors list validity."""
    if isinstance(sensors, str):
        sensors = [sensors]
    if sensors is not None:
        sensors = [check_sensor(sensor) for sensor in sensors]
    return sensors


def check_satellites(satellites):
    """Check satellites list validity."""
    if isinstance(satellites, str):
        satellites = [satellites]
    if satellites is not None:
        satellites = [check_satellite(satellite) for satellite in satellites]
    return satellites


def check_full_product_levels(full_product_levels):
    """Check full product levels list validity."""
    if isinstance(full_product_levels, str):
        full_product_levels = [full_product_levels]
    if full_product_levels is not None:
        full_product_levels = [
            check_full_product_level(full_product_level) for full_product_level in full_product_levels
        ]
    return full_product_levels


def check_product_levels(product_levels):
    """Check product levels list validity."""
    if isinstance(product_levels, str):
        product_levels = [product_levels]
    if product_levels is not None:
        product_levels = [check_product_level(product_level) for product_level in product_levels]
    return product_levels


def check_product_categories(product_categories):
    """Check product category list validity."""
    if isinstance(product_categories, str):
        product_categories = [product_categories]
    if product_categories is not None:
        product_categories = [check_product_category(product_category) for product_category in product_categories]
    return product_categories


def check_product_types(product_types):
    """Check product types list validity."""
    if isinstance(product_types, str):
        product_types = [product_types]
    if product_types is not None:
        product_types = [check_product_type(product_type) for product_type in product_types]
    return product_types


def check_versions(versions):
    """Check versions list validity."""
    if isinstance(versions, (int, str)):
        versions = [versions]
    if versions is not None:
        versions = [check_version(version) for version in versions]
    return versions
