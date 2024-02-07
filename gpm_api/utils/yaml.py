#!/usr/bin/env python3
"""
Created on Wed Jul 12 13:30:50 2023

@author: ghiggi
"""
import yaml


def read_yaml(filepath: str) -> dict:
    """Read a YAML file into a dictionary.

    Parameters
    ----------
    filepath : str
        Input YAML file path.

    Returns
    -------
    dict
        Dictionary with the attributes read from the YAML file.
    """
    with open(filepath) as f:
        dictionary = yaml.safe_load(f)
    return dictionary


def write_yaml(dictionary, filepath, sort_keys=False):
    """Write a dictionary into a YAML file.

    Parameters
    ----------
    dictionary : dict
        Dictionary to write into a YAML file.
    """
    with open(filepath, "w") as f:
        yaml.dump(dictionary, f, sort_keys=sort_keys)
    return None
