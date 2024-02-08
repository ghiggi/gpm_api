#!/usr/bin/env python3
"""
Created on Thu Oct 26 18:09:38 2023

@author: ghiggi
"""


def flatten_list(nested_list):
    """Flatten a nested list into a single-level list."""

    if isinstance(nested_list, list) and len(nested_list) == 0:
        return nested_list
    # If list is already flat, return as is to avoid flattening to chars
    if isinstance(nested_list, list) and not isinstance(nested_list[0], list):
        return nested_list
    return (
        [item for sublist in nested_list for item in sublist]
        if isinstance(nested_list, list)
        else [nested_list]
    )
