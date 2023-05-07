#!/usr/bin/env python3
"""
Created on Thu Mar  9 11:46:07 2023

@author: ghiggi
"""
import os
from typing import Dict

import yaml


def _read_yaml_file(fpath):
    """Read a YAML file into dictionary."""
    with open(fpath) as f:
        dictionary = yaml.safe_load(f)
    return dictionary


def _write_yaml_file(dictionary, fpath, sort_keys=False):
    """Write dictionary to YAML file."""
    with open(fpath, "w") as f:
        yaml.dump(dictionary, f, sort_keys=sort_keys)
    return


def define_gpm_api_configs(gpm_username: str, gpm_password: str, gpm_base_dir: str):
    """
    Defines the GPM-API configuration file with the given credentials and base directory.

    Parameters
    ----------
    gpm_username : str
        The username for the NASA GPM PPS account.
    gpm_password : str
        The password for the NASA GPM PPS account.
    gpm_base_dir : str
        The base directory where GPM data are stored.

    Notes
    -----
    This function writes a YAML file to the user's home directory at ~/.config_gpm_api.yml
    with the given GPM-API credentials and base directory. The configuration file can be
    used for authentication when making GPM-API requests.

    """
    config_dict = {}
    config_dict["gpm_username"] = gpm_username
    config_dict["gpm_password"] = gpm_password
    config_dict["gpm_base_dir"] = gpm_base_dir

    # Retrieve user home directory
    home_directory = os.path.expanduser("~")

    # Define path to .config_gpm_api.yaml file
    fpath = os.path.join(home_directory, ".config_gpm_api.yml")

    # Write the config file
    _write_yaml_file(config_dict, fpath, sort_keys=False)

    print("The GPM-API config file has been written successfully!")
    return


def read_gpm_api_configs() -> Dict[str, str]:
    """
    Reads the GPM-API configuration file and returns a dictionary with the configuration settings.

    Returns
    -------
    dict
        A dictionary containing the configuration settings for the GPM-API, including the
        username, password, and GPM base directory.

    Raises
    ------
    ValueError
        If the configuration file has not been defined yet. Use `gpm_api.define_configs()` to
        specify the configuration file path and settings.

    Notes
    -----
    This function reads the YAML configuration file located at ~/.config_gpm_api.yml, which
    should contain the GPM-API credentials and base directory specified by `gpm_api.define_configs()`.
    """
    # Retrieve user home directory
    home_directory = os.path.expanduser("~")
    # Define path where .config_gpm_api.yaml file should be located
    fpath = os.path.join(home_directory, ".config_gpm_api.yml")
    if not os.path.exists(fpath):
        raise ValueError(
            "The GPM-API config file has not been specified. Use gpm_api.define_configs to specify it !"
        )
    # Read the GPM-API config file
    config_dict = _read_yaml_file(fpath)
    return config_dict


####--------------------------------------------------------------------------.
def _get_config_key(key, value=None):
    """Return the config key if `value` is None."""
    if value is None:
        value = read_gpm_api_configs()[key]
    return value


def get_gpm_base_dir(gpm_base_dir=None):
    """Return the GPM base directory."""
    return _get_config_key(key="gpm_base_dir", value=gpm_base_dir)


def get_gpm_username(gpm_username=None):
    """Return the GPM-API PPS username."""
    return _get_config_key(key="gpm_username", value=gpm_username)


def get_gpm_password(gpm_password=None):
    """Return the GPM-API PPS password."""
    return _get_config_key(key="gpm_password", value=gpm_password)
