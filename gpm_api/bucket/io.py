#!/usr/bin/env python3
"""
Created on Wed Aug  2 16:10:28 2023

@author: ghiggi
"""
import glob
import os
import fnmatch


def get_parquet_fpaths(
    bucket_base_dir,
    year="*",
    month="*",
    day="*",
):
    """Search for bin bucket parquet file paths in the bucket_base_dir."""
    glob_pattern = os.path.join(bucket_base_dir, year, month, day, "*", "*=*", "*=*", "*.parquet")
    fpaths = glob.glob(glob_pattern)
    return fpaths


def group_fpaths_by_bin(fpaths):
    """Group bin bucket parquet file paths by geographic bin."""
    from collections import defaultdict  # more efficient than plain dict

    grouped_fpaths = defaultdict(list)
    for fpath in fpaths:
        bin_name = "|".join(fpath.split(os.path.sep)[-3:-1])
        grouped_fpaths[bin_name].append(fpath)
    return grouped_fpaths
