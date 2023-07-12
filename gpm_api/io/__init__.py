#!/usr/bin/env python3
"""
Created on Mon Aug 15 00:17:07 2022

@author: ghiggi
"""

import os

from gpm_api import _root_path
from gpm_api.utils.yaml import read_yaml_file

# Current GPM version
GPM_VERSION = 7


def get_info_dict():
    fpath = os.path.join(_root_path, "gpm_api", "etc", "product_def.yml")
    return read_yaml_file(fpath)
