#!/usr/bin/env python3
"""
Created on Thu Oct 26 18:09:38 2023

@author: ghiggi
"""

# TODO: flatten_list was previously defined in pps.py, download.py, groups_variable.py
# -->


# download.py
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


# groups_variable.py
# def flatten_list(nested_list):
#     flat_list = []
#     for item in nested_list:
#         if isinstance(item, list):
#             flat_list.extend(flatten_list(item))
#         else:
#             flat_list.append(item)
#     return flat_list

### PPS.PY
# def flatten_list(nested_list):

#     """Flatten a nested list into a single-level list."""
#     return (
#         [item for sublist in nested_list for item in sublist]
#         if isinstance(nested_list, list)
#         else [nested_list]
#     )
