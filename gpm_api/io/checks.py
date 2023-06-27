#!/usr/bin/env python3
"""
Created on Sun Aug 14 20:02:18 2022
@author: ghiggi
"""
import datetime
import os

import numpy as np


def is_not_empty(x):
    return bool(x)


def is_empty(x):
    return not x


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
    if base_dir[-1] == "/":
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


def check_version(version):
    if not isinstance(version, int):
        raise ValueError("Please specify the GPM version with an integer between 5 and 7.")
    if version not in [5, 6, 7]:
        raise ValueError("Download/Reading have been implemented only for GPM versions 5, 6 and 7.")


def check_product(product, product_type):
    from gpm_api.io.products import available_products

    if not isinstance(product, str):
        raise ValueError("'Ask for a single product at time.'product' must be a single string.")
    if product not in available_products(product_type=product_type):
        raise ValueError("Please provide a valid GPM product --> gpm_api.available_products().")


def check_product_type(product_type):
    if not isinstance(product_type, str):
        raise ValueError("Please specify the product_type as 'RS' or 'NRT'.")
    if product_type not in ["RS", "NRT"]:
        raise ValueError("Please specify the product_type as 'RS' or 'NRT'.")


def check_time(time):
    """Check time validity.

    It returns a datetime.datetime object.

    Parameters
    ----------
    time : (datetime.datetime, datetime.date, np.datetime64, str)
        Time object.
        Accepted types: datetime.datetime, datetime.date, np.datetime64, str
        If string type, it expects the isoformat 'YYYY-MM-DD hh:mm:ss'.

    Returns
    -------
    time : datetime.datetime
        datetime.datetime object.

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
                time = time.astype("datetime64[s]").tolist()
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
    return time


def check_date(date):
    if not isinstance(date, (datetime.date, datetime.datetime)):
        raise ValueError("date must be a datetime object")
    if isinstance(date, datetime.datetime):
        date = date.date()
    return date


def check_start_end_time(start_time, end_time):
    start_time = check_time(start_time)
    end_time = check_time(end_time)
    # Check start_time and end_time are chronological
    if start_time > end_time:
        raise ValueError("Provide start_time occuring before of end_time.")
    # Check start_time and end_time are in the past
    if start_time > datetime.datetime.utcnow():
        raise ValueError("Provide a start_time occuring in the past.")
    if end_time > datetime.datetime.utcnow():
        raise ValueError("Provide a end_time occuring in the past.")
    return (start_time, end_time)


def check_scan_mode(scan_mode, product, version):
    """Checks the validity of scan_mode."""
    # -------------------------------------------------------------------------.
    # Get valid scan modes
    from gpm_api.io.scan_modes import available_scan_modes

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


def check_bbox(bbox):
    """
    Check correctnes of bounding box.
    bbox format: [lon_0, lon_1, lat_0, lat_1]
    bbox should be provided with longitude between -180 and 180, and latitude
    between -90 and 90.
    """
    if bbox is None:
        return bbox
    # If bbox provided
    if not (isinstance(bbox, list) and len(bbox) == 4):
        raise ValueError("Provide valid bbox [lon_0, lon_1, lat_0, lat_1]")
    if bbox[2] > 90 or bbox[2] < -90 or bbox[3] > 90 or bbox[3] < -90:
        raise ValueError("Latitude is defined between -90 and 90")
    # Try to be sure that longitude is specified between -180 and 180
    if bbox[0] > 180 or bbox[1] > 180:
        print("bbox should be provided with longitude between -180 and 180")
        bbox[0] = bbox[0] - 180
        bbox[1] = bbox[1] - 180
    return bbox
