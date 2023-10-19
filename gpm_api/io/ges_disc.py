#!/usr/bin/env python3
"""
Created on Mon Oct  9 12:44:42 2023

@author: ghiggi
"""
import os
import re
import subprocess

from gpm_api.io.products import get_info_dict, is_trmm_product

###---------------------------------------------------------------------------.


def get_ges_disc_dir_key(product):
    info_dict = get_info_dict()[product]
    dir_pattern = info_dict["ges_disc_dir"]
    return dir_pattern


def _get_gesc_disc_product_level_dirname(product):
    dir_pattern = get_ges_disc_dir_key(product)
    if isinstance(dir_pattern, str):
        return dir_pattern.split("/")[0]
    else:
        return None


def _get_gesc_disc_product_name(product):
    dir_pattern = get_ges_disc_dir_key(product)
    if isinstance(dir_pattern, str):
        return dir_pattern.split("/")[1]
    else:
        return None


###---------------------------------------------------------------------------.


def get_ges_disc_base_url(product):
    # TRMM
    if is_trmm_product(product):
        ges_disc_base_url = "https://disc2.gesdisc.eosdis.nasa.gov/data/"

    # GPM
    else:
        ges_disc_base_url = "https://gpm1.gesdisc.eosdis.nasa.gov/data"
        ges_disc_base_url = "https://gpm2.gesdisc.eosdis.nasa.gov/data"
    return ges_disc_base_url


def get_ges_disc_product_path(product, version):
    base_url = get_ges_disc_base_url(product)
    dir_pattern = get_ges_disc_dir_key(product)
    if isinstance(dir_pattern, str):
        dir_pattern = f"{dir_pattern}.0{version}"
    url = os.path.join(base_url, dir_pattern)
    return url


def _get_ges_disc_url_content(url):
    cmd = f"wget -O - {url}"
    args = cmd.split()
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = process.communicate()[0].decode()
    # Check if server is available
    if stdout == "":
        raise ValueError(
            "The GES DISC data archive is currently unavailable. Sorry for the inconvenience."
        )
    return stdout


def _get_href_value(input_string):
    """Infer href value."""
    match = re.search(r'<a\s+href="([^"]+)"', input_string)
    # Check if a match was found and extract the value
    if match:
        href_value = match.group(1)
    else:
        href_value = ""
    # Exclude .xml files and doc directory
    if ".xml" in href_value or "doc/" in href_value:
        href_value = ""
    return href_value


def _get_gesc_disc_list_path(url):
    wget_output = _get_ges_disc_url_content(url)
    list_content = [_get_href_value(s) for s in wget_output.split("alt=")[4:]]
    list_content = [s for s in list_content if s != ""]
    if len(list_content) == 0:
        dirname = os.path.basename(url)
        raise ValueError(f"The GES DISC {dirname} directory is empty.")
    list_path = [os.path.join(url, s) for s in list_content]
    return list_path
