#!/usr/bin/env python3
"""
Created on Sun Aug 14 20:02:18 2022
@author: ghiggi
"""
import datetime
import os

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
    if isinstance(filepaths, str):
        filepaths = [filepaths]
    if not isinstance(filepaths, list):
        raise TypeError("Expecting a list of filepaths.")
    return filepaths


def check_variables(variables):
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
    valid_storages = ["ges_disc", "pps", "local"]
    if storage.lower() not in valid_storages:
        raise ValueError(f"{storage} is an invalid storage. Valid values are {valid_storages}.")
    return storage.lower()


def check_remote_storage(storage):
    """Check storage is remote."""
    if not isinstance(storage, str):
        raise TypeError("'storage' must be a string.")
    valid_storages = ["ges_disc", "pps"]
    if storage.lower() not in valid_storages:
        raise ValueError(
            f"{storage} is an invalid remote storage. Valid values are {valid_storages}."
        )
    return storage.lower()


def check_version(version):
    if not isinstance(version, int):
        raise ValueError("Please specify the GPM version with an integer between 5 and 7.")
    if version not in [4, 5, 6, 7]:
        raise ValueError("Download/Reading have been implemented only for GPM versions 5, 6 and 7.")


def check_product_version(version, product):
    from gpm_api.io.products import available_versions, get_last_product_version

    if version is None:
        version = get_last_product_version(product)
    check_version(version)
    # Check valid version for such product
    valid_versions = available_versions(product)
    if version not in valid_versions:
        raise ValueError(f"Valid versions for product {product} are {valid_versions}.")
    return version


def check_product(product, product_type):
    from gpm_api.io.products import available_products

    if not isinstance(product, str):
        raise ValueError("'Ask for a single product at time.'product' must be a single string.")
    if product not in available_products(product_type=product_type):
        raise ValueError("Please provide a valid GPM product --> gpm_api.available_products().")


def check_product_type(product_type):
    if not isinstance(product_type, str):
        raise ValueError("Please specify the product_type as a string..")
    if product_type not in ["RS", "NRT"]:
        raise ValueError("Please specify the product_type as 'RS' or 'NRT'.")


def check_product_category(product_category):
    if not isinstance(product_category, str):
        raise ValueError("Please specify the product_category as a string.")
    valid_values = ["RADAR", "PMW", "CMB", "IMERG"]
    if product_category not in valid_values:
        raise ValueError(
            f"{product_category} is an invalid product_category. Valid values are {valid_values}."
        )


def check_product_level(product_level):
    if not isinstance(product_level, str):
        raise ValueError("Please specify the product_level as a string.")
    valid_values = ["1A", "1B", "1C", "2A", "2B"]
    if product_level not in valid_values:
        raise ValueError(
            f"{product_level} is an invalid product_level. Currently accepted values are {valid_values}."
        )


def check_product_validity(product, product_type=None):
    """Check product validity."""
    from gpm_api.io.products import available_products  # circular otherwise

    if product not in available_products(product_type=product_type):
        if product_type is None:
            raise ValueError(
                f"The {product} product is not available. See gpm_api.available_products()."
            )
        else:
            raise ValueError(
                f"The {product} product is not available as {product_type} product_type."
            )


def check_time(time):
    """Check time validity.

    It returns a datetime.datetime object to seconds precision.

    Parameters
    ----------
    time : (datetime.datetime, datetime.date, np.datetime64, str)
        Time object.
        Accepted types: datetime.datetime, datetime.date, np.datetime64, str
        If string type, it expects the isoformat 'YYYY-MM-DD hh:mm:ss'.

    Returns
    -------
    time : datetime.datetime
        datetime.datetime object

    """
    if not isinstance(time, (datetime.datetime, datetime.date, np.datetime64, np.ndarray, str)):
        raise TypeError(
            "Specify time with datetime.datetime objects or a "
            "string of format 'YYYY-MM-DD hh:mm:ss'."
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
        else:
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
        raise ValueError("Provide start_time occurring before of end_time.")
    # Check start_time and end_time are in the past
    if start_time > datetime.datetime.utcnow():
        raise ValueError("Provide a start_time occurring in the past.")
    if end_time > datetime.datetime.utcnow():
        raise ValueError("Provide a end_time occurring in the past.")
    return (start_time, end_time)


def check_valid_time_request(start_time, end_time, product):
    """Check validity of the time request."""
    from gpm_api.io.products import get_product_end_time, get_product_start_time

    product_start_time = get_product_start_time(product)
    product_end_time = get_product_end_time(product)
    if start_time < product_start_time:
        raise ValueError(f"{product} production started the {product_start_time}.")
    if end_time > product_end_time:
        raise ValueError(f"{product} production ended the {get_product_end_time}.")


def check_scan_mode(scan_mode, product, version):
    """Checks the validity of scan_mode."""
    # -------------------------------------------------------------------------.
    # Get valid scan modes
    from gpm_api.io.products import available_scan_modes

    scan_modes = available_scan_modes(product, version)

    # Infer scan mode if not specified
    if scan_mode is None:
        scan_mode = scan_modes[0]
        if len(scan_modes) > 1:
            print(f"'scan_mode' has not been specified. Default to {scan_mode}.")

    # -------------------------------------------------------------------------.
    # Check that a single scan mode is specified
    if scan_mode is not None and not isinstance(scan_mode, str):
        raise ValueError("Specify a single 'scan_mode'.")

    # -------------------------------------------------------------------------.
    # Check that a valid scan mode is specified
    if scan_mode is not None and scan_mode not in scan_modes:
        raise ValueError(f"For {product} product, valid scan_modes are {scan_modes}.")

    # -------------------------------------------------------------------------.
    return scan_mode
