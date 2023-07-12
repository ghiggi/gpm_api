#!/usr/bin/env python3
"""
Created on Wed Jul 12 13:30:50 2023

@author: ghiggi
"""
import yaml


def read_yaml_file(fpath):
    """Read a YAML file into dictionary."""
    with open(fpath) as f:
        dictionary = yaml.safe_load(f)
        # dictionary = yaml.load(infile, Loader=yaml.FullLoader)
    return dictionary


def write_yaml_file(dictionary, fpath, sort_keys=False):
    """Write dictionary to YAML file."""
    with open(fpath, "w") as f:
        yaml.safe_dump(dictionary, f, sort_keys=sort_keys)
    return None
