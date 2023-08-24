#!/usr/bin/env python3
"""
Created on Wed Aug  2 16:10:28 2023

@author: ghiggi
"""
import os


def _retrieve_list_bin_dir_path(base_dir):
    """Retrieve a dictionary with the list of filepaths for each bucket bin."""
    list_bin_dir_path = []
    with os.scandir(base_dir) as base_it:
        for lonbin_entry in base_it:
            if lonbin_entry.is_dir():
                lonbin_name = lonbin_entry.name
                lonbin_path = os.path.join(base_dir, lonbin_name)
                with os.scandir(lonbin_path) as latbin_it:
                    for latbin_entry in latbin_it:
                        if latbin_entry.is_dir():
                            latbin_name = latbin_entry.name
                            latbin_path = os.path.join(lonbin_path, latbin_name)
                            list_bin_dir_path.append(latbin_path)
    return list_bin_dir_path


def _get_parquet_file_list(bin_dir_path):
    """Retrieve (key, list_file_path) tuple."""
    # Retrieve file path list
    with os.scandir(bin_dir_path) as file_it:
        file_list = [
            file_entry.path
            for file_entry in file_it
            if (file_entry.is_file() and file_entry.name.endswith(".parquet"))
        ]

    # Define dictionary key
    sep = os.path.sep
    key = sep.join(bin_dir_path.split(os.path.sep)[-2:])
    return key, file_list


def _get_fpaths_by_bin_parallel(list_bin_dir_path):
    """Retrieve a dictionary with the list of filepaths for each bucket bin."""
    import concurrent

    results = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_dict = {
            executor.submit(_get_parquet_file_list, bin_dir_path): bin_dir_path
            for bin_dir_path in list_bin_dir_path
        }
        for future in concurrent.futures.as_completed(future_dict):
            bin_dir_path = future_dict[future]
            try:
                key, file_list = future.result()
                results[key] = file_list
            except Exception as e:
                print(f"Error while listing Parquet file paths for bin {bin_dir_path}: {e}")
    return results


def get_fpaths_by_bin(base_dir, parallel=True):
    """Retrieve a dictionary with the list of filepaths for each bucket bin."""
    list_bin_dir_path = _retrieve_list_bin_dir_path(base_dir)
    list_bin_dir_path = list_bin_dir_path
    if parallel:
        results = _get_fpaths_by_bin_parallel(list_bin_dir_path)
    else:
        results = {}
        for bin_dir_path in list_bin_dir_path:
            key, file_list = _get_parquet_file_list(bin_dir_path)
            results[key] = file_list
    return results
