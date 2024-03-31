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
"""GPM-API configurations settings."""
import os
import platform
from subprocess import Popen
from typing import Optional

from gpm.utils.yaml import read_yaml, write_yaml


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


def _define_config_filepath():
    """Define the config YAML file path."""
    # Retrieve user home directory
    home_directory = os.path.expanduser("~")
    # Define path where .config_gpm.yaml file should be located
    return os.path.join(home_directory, ".config_gpm.yaml")


def define_configs(
    base_dir: Optional[str] = None,
    username_pps: Optional[str] = None,
    password_pps: Optional[str] = None,
    username_earthdata: Optional[str] = None,
    password_earthdata: Optional[str] = None,
):
    """Defines the GPM-API configuration file with the given credentials and base directory.

    Parameters
    ----------
    base_dir : str
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
    This function writes a YAML file to the user's home directory at ~/.config_gpm.yaml
    with the given GPM-API credentials and base directory. The configuration file can be
    used for authentication when making GPM-API requests.

    """
    # Define path to .config_gpm.yaml file
    filepath = _define_config_filepath()

    # If the config exists, read it and update it ;)
    if os.path.exists(filepath):
        config_dict = read_yaml(filepath)
        action_msg = "updated"
    else:
        config_dict = {}
        action_msg = "written"

    # Add GPM-API Base directory
    if base_dir is not None:
        config_dict["base_dir"] = str(base_dir)  # deal with Pathlib

    # Define PPS authentication parameters
    if isinstance(username_pps, str) and isinstance(password_pps, str):
        config_dict["username_pps"] = username_pps
        config_dict["password_pps"] = password_pps

    # Define EarthData authentication files and parameters
    if isinstance(username_earthdata, str) and isinstance(password_earthdata, str):
        config_dict["username_earthdata"] = username_earthdata
        config_dict["password_earthdata"] = password_earthdata
        if not os.environ.get("PYTEST_CURRENT_TEST"):  # run this only if not executing tests !
            set_ges_disc_authentification(username_earthdata, password_earthdata)

    # Write the GPM-API config file
    write_yaml(config_dict, filepath, sort_keys=False)

    print(f"The GPM-API config file has been {action_msg} successfully!")


def read_configs() -> dict[str, str]:
    """Reads the GPM-API configuration file and returns a dictionary with the configuration settings.

    Returns
    -------
    dict
        A dictionary containing the configuration settings for the GPM-API, including the
        username, password, and GPM base directory.

    Raises
    ------
    ValueError
        If the configuration file has not been defined yet. Use `gpm.define_configs()` to
        specify the configuration file path and settings.

    Notes
    -----
    This function reads the YAML configuration file located at ~/.config_gpm.yaml, which
    should contain the GPM-API credentials and base directory specified by `gpm.define_configs()`.

    """
    # Define path to .config_gpm.yaml file
    filepath = _define_config_filepath()
    # Check it exists
    if not os.path.exists(filepath):
        raise ValueError(
            "The GPM-API config file has not been specified. Use gpm.define_configs to specify it !",
        )
    # Read the GPM-API config file
    return read_yaml(filepath)


####--------------------------------------------------------------------------.
def _get_config_key(key):
    """Return the config key if `value` is not None."""
    import gpm

    value = gpm.config.get(key, None)
    if value is None:
        raise ValueError(f"The '{key}' is not specified in the GPM-API configuration file.")
    return value


def get_base_dir(base_dir=None):
    """Return the GPM base directory."""
    import gpm

    if base_dir is None:
        base_dir = gpm.config.get("base_dir")
    if base_dir is None:
        raise ValueError("The 'base_dir' is not specified in the GPM-API configuration file.")
    return str(base_dir)  # convert Path to str


def get_username_pps():
    """Return the GPM-API PPS username."""
    return _get_config_key(key="username_pps")


def get_password_pps():
    """Return the GPM-API PPS password."""
    return _get_config_key(key="password_pps")


def get_username_earthdata():
    """Return the GPM-API EarthData username."""
    return _get_config_key(key="username_earthdata")


def get_password_earthdata():
    """Return the GPM-API EarthData password."""
    return _get_config_key(key="password_earthdata")
