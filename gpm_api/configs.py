#!/usr/bin/env python3
"""
Created on Thu Mar  9 11:46:07 2023

@author: ghiggi
"""
import os
import platform
import shutil
from subprocess import Popen
from typing import Dict

import yaml


####--------------------------------------------------------------------------.
def set_ges_disc_authentification(username, password):
    """Create authentication files for access to the GES DISC Data Archive.

    Follow the additional steps detailed at https://disc.gsfc.nasa.gov/earthdata-login
    to enable access to the GES DISC HTTPS Data Archive.

    The code snippet is taken from
    https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Generate%20Earthdata%20Prerequisite%20Files

    Parameters
    ----------
    username : str
        EarthData login username.
    password : TYPE
        EarthData login password.

    Returns
    -------
    None.

    """
    # TODO
    # - Write earthdata login and password to gpm_api config yaml
    urs = "urs.earthdata.nasa.gov"  # Earthdata URL to call for authentication
    home_dir_path = os.path.expanduser("~") + os.sep

    with open(home_dir_path + ".netrc", "w") as file:
        file.write(f"machine {urs} login {username} password {password}")
        file.close()
    with open(home_dir_path + ".urs_cookies", "w") as file:
        file.write("")
        file.close()
    with open(home_dir_path + ".dodsrc", "w") as file:
        file.write(f"HTTP.COOKIEJAR={home_dir_path}.urs_cookies\n")
        file.write(f"HTTP.NETRC={home_dir_path}.netrc")
        file.close()

    print("Saved .netrc, .urs_cookies, and .dodsrc to:", home_dir_path)

    # Set appropriate permissions for Linux/macOS
    if platform.system() != "Windows":
        Popen("chmod og-rw ~/.netrc", shell=True)
    else:
        # Copy dodsrc to working directory in Windows
        shutil.copy2(home_dir_path + ".dodsrc", os.getcwd())
        print("Copied .dodsrc to:", os.getcwd())


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


def define_gpm_api_configs(
    gpm_base_dir: str,
    username_pps: str,
    password_pps: str,
    username_earthdata=None,
    password_earthdata=None,
):
    """
    Defines the GPM-API configuration file with the given credentials and base directory.

    Parameters
    ----------
    gpm_base_dir : str
        The base directory where GPM data are stored.
    username_pps : str, optional
        The username for the NASA GPM PPS account.
    password_pps : str, optional
        The password for the NASA GPM PPS account.
    username_earthdata : str, optional
        The username for the NASA EarthData account.
    password_earthdata : str, optional
        The password for the NASA EarthData account.

    Notes
    -----
    This function writes a YAML file to the user's home directory at ~/.config_gpm_api.yml
    with the given GPM-API credentials and base directory. The configuration file can be
    used for authentication when making GPM-API requests.

    """
    config_dict = {}
    config_dict["gpm_base_dir"] = gpm_base_dir

    # Define PPS authentication parameters
    if isinstance(username_pps, str) and isinstance(password_pps, str):
        config_dict["username_pps"] = username_pps
        config_dict["password_pps"] = password_pps

    # Define EarthData authentication files and parameters
    if isinstance(username_earthdata, str) and isinstance(password_earthdata, str):
        config_dict["username_earthdata"] = username_earthdata
        config_dict["password_earthdata"] = password_earthdata
        set_ges_disc_authentification(username_earthdata, password_earthdata)

    # Retrieve user home directory
    home_directory = os.path.expanduser("~")

    # Define path to .config_gpm_api.yaml file
    fpath = os.path.join(home_directory, ".config_gpm_api.yml")

    # Write the GPM-API config file
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
def _get_config_key(key):
    """Return the config key if `value` is None."""
    value = read_gpm_api_configs().get(key, None)
    if value is None:
        raise ValueError(f"The GPM-API {key} parameter has not been defined ! ")
    return value


def get_gpm_base_dir():
    """Return the GPM base directory."""
    return _get_config_key(key="gpm_base_dir")


def get_pps_username():
    """Return the GPM-API PPS username."""
    return _get_config_key(key="username_pps")


def get_pps_password():
    """Return the GPM-API PPS password."""
    return _get_config_key(key="password_pps")


def get_earthdata_username():
    """Return the GPM-API EarthData username."""
    return _get_config_key(key="username_earthdata")


def get_earthdata_password():
    """Return the GPM-API EarthData password."""
    return _get_config_key(key="password_earthdata")
