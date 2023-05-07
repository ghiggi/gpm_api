#!/usr/bin/env python3
"""
Created on Tue Jul 21 19:54:34 2020

@author: ghiggi
"""
import ast

import h5py
import numpy

# -----------------------------------------------------------------------------.
from gpm_api.utils.utils_string import (
    str_collapse,
    str_detect,
    str_isfloat,
    str_isinteger,
    str_islist,
    str_remove_empty,
    str_replace,
)


# -----------------------------------------------------------------------------.
def initialize_dict_with(keys):
    # dict(zip(keys, [None]*len(keys)))
    # {key: None for key in keys}
    return dict.fromkeys(keys)


def numpy_numeric_format():
    return (
        float,  # numpy.float deprecated since 1.20 and remove in 1.24
        numpy.float32,
        numpy.float64,
        numpy.integer,
        numpy.int16,
        numpy.int32,
    )


def parse_attr_string(s):
    # If multiple stuffs between brackets [ ], convert to list
    if isinstance(s, str) and str_islist(s):
        s = ast.literal_eval(s)
    # If still a comma in a string --> Convert into a list
    if isinstance(s, str) and str_detect(s, ","):
        s = s.split(",")
    # If the character can be a number, convert it
    if isinstance(s, str) and str_isinteger(s):
        s = int(float(s))  # prior float because '0.0000' otherwise crash
    elif isinstance(s, str) and str_isfloat(s):
        s = float(s)
    else:
        s = s
    return s


def parse_HDF5_GPM_attributes(x, parser=parse_attr_string):
    """
    Parse attributes of hdf objects
    parser: function parsing strings
    """
    # TODO: customize missing ...
    # Initialize dictionary with hdf keys
    attr_dict = initialize_dict_with(x.attrs.keys())
    # Assign values to dictionary
    for item in list(x.attrs.keys()):
        # Extract hdf attributes
        attr = x.attrs[item]
        # If attr number
        if isinstance(attr, numpy_numeric_format()):
            attr_dict[item] = attr
        # If attr is a string
        elif isinstance(attr, str):
            attr_dict[item] = parser(attr)
        # If a compressed string [with lot of attributes separeted by \n ...
        elif isinstance(attr, numpy.bytes_):
            # Decode
            attr_str = attr.decode("UTF-8", errors="ignore").split("\n")  # Always create a list
            # Clean the string
            attr_str = str_replace(attr_str, ";", "")
            attr_str = str_replace(attr_str, "\t", "")
            # attr_str = str_replace(attr_str,"=$","='-9999'")
            # Return a sub-dictionary if multiple arguments
            if isinstance(attr_str, list) and len(attr_str) > 1:
                # Remove empty list element
                attr_str = str_remove_empty(attr_str)
                # If = not present, collapse the string
                if not all(str_detect(attr_str, "=")):
                    attr_dict[item] = str_collapse(attr_str)
                    continue
                # If = is present in each list element --> Return a subdictionary
                else:
                    tmp_dict = dict(
                        (k.strip(), v.strip()) for k, v in (s.split("=", 1) for s in attr_str)
                    )
                    # Process dictionary values
                    for k, v in tmp_dict.items():
                        tmp_dict[k] = parser(v)
                    # Attach dictionary
                    attr_dict[item] = tmp_dict
            # Return as string otherwise
            else:
                attr_dict[item] = parser(attr_str[0])
        elif attr is None:
            attr_dict[item] = None
        else:
            continue
            # print("Not able to parse attribute:", item)
            # print(attr)
            # breakpoint()
    # -------------------------------------
    return attr_dict


# -----------------------------------------------------------------------------.
def print_hdf5(hdf, sep="\t", dataset_attrs=True, group_attrs=False):
    """
    Print the structure of HDF5 file.

    Parameters
    ----------
    hdf : TYPE
        DESCRIPTION.
    sep : str, optional
        How to separate the printed text. The default is '\t'.
    dataset_attrs : boolean, optional
        Print datasets attributes if True. The default is True.
    group_attrs : boolean, optional
        Print group attributes if True. The default is False.

    Returns
    -------
    None.

    """
    if isinstance(hdf, (h5py.Group, h5py.File)):
        # Option to print group attributes
        if group_attrs:
            # Now retrieve group (or global) attributes
            attr_keys = list(hdf.attrs.keys())
            if len(attr_keys) >= 1:
                for attr_key in attr_keys:
                    tmp_attr = hdf.attrs[attr_key]
                    # Decode if compressed
                    if isinstance(tmp_attr, numpy.bytes_):
                        tmp_attr = tmp_attr.decode("UTF-8")
                        tmp_attr = "\n" + tmp_attr
                        tmp_attr = str_replace(tmp_attr, "\n", str("\n" + sep + "\t" + "\t" + "\t"))
                        print(sep + "\t", "-->", attr_key, ":", tmp_attr)
                    else:
                        print(sep + "\t", "-->", attr_key, ":", tmp_attr)
        # Print keys of sub-group and sub-dataset
        for key in hdf:
            print(sep, "-", key, ":", hdf[key])
            print_hdf5(
                hdf[key],
                sep=sep + "\t",
                dataset_attrs=dataset_attrs,
                group_attrs=group_attrs,
            )
    elif isinstance(hdf, h5py.Dataset):
        # Option to print dataset attributes
        if dataset_attrs:
            for attr_key in hdf.attrs:
                print(sep + "\t", "-", attr_key, ":", hdf.attrs[attr_key])


def h5dump(filepath, group="/", dataset_attrs=True, group_attrs=True):
    """
    Print HDF5 file metadata (and then close HDF5 file).

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    group : str, optional
        Specify a HDF5 group. The default is '/' (root group).
    dataset_attrs : boolean, optional
        Print datasets attributes if True. The default is True.
    group_attrs : boolean, optional
        Print group attributes if True. The default is True.

    Returns
    -------
    None.

    """
    with h5py.File(filepath, "r") as hdf:
        print_hdf5(hdf[group], dataset_attrs=dataset_attrs, group_attrs=group_attrs)


# -----------------------------------------------------------------------------.
# def print_hdf5_keys(x):
#     if (isinstance(x, h5py.highlevel.Group)):
#         print(x.name,":", list(x.keys()))
#         if hasattr(x, 'keys'):
#             for item in x.keys():
#                 print_hdf5_keys(x[item])
#         else: # not sure occurs ...
#             print(x.name,":", [])
#
# def print_hdf5_datasets(x):
#     if (isinstance(x, h5py.highlevel.Group)):
#         if hasattr(x, 'keys'):
#             for item in x.keys():
#                 print_hdf5_datasets(x[item])
#     elif (isinstance(x, h5py._hl.dataset.Dataset)):
#         print(x.name)
#     else:
#         print("What is going on? Check type(x) in debugger")
#
# def print_hdf5_shape(x):
#     if (isinstance(x, h5py.highlevel.Group) or isinstance(x, h5py._hl.dataset.Dataset)):
#         if hasattr(x, 'shape'):
#             print(x.name,":", x.shape)
#         if hasattr(x, 'keys'):
#             for item in x.keys():
#                 print_hdf5_shape(x[item])

# -----------------------------------------------------------------------------.
def hdf5_objects_names(hdf):
    l_objs = []
    hdf.visit(l_objs.append)
    return l_objs


def hdf5_groups_names(hdf):
    # Does not include the one you pass
    l_objs = hdf5_objects_names(hdf)
    return [obj for obj in l_objs if isinstance(hdf[obj], (h5py.Group, h5py.File))]


def hdf5_datasets_names(hdf):
    l_objs = hdf5_objects_names(hdf)
    return [obj for obj in l_objs if isinstance(hdf[obj], h5py.Dataset)]


def hdf5_objects(hdf):
    objects_names = hdf5_objects_names(hdf)
    return {object_name: hdf[object_name] for object_name in objects_names}


def hdf5_groups(hdf):
    # Does not include the one you pass
    groups_names = hdf5_groups_names(hdf)
    return {group_name: hdf[group_name] for group_name in groups_names}


def hdf5_datasets(hdf):
    datasets_names = hdf5_datasets_names(hdf)
    return {dataset_name: hdf[dataset_name] for dataset_name in datasets_names}


# Shape
def hdf5_datasets_shape(hdf):
    datasets_names = hdf5_datasets_names(hdf)
    return {dataset_name: hdf[dataset_name].shape for dataset_name in datasets_names}


# Dtype
def hdf5_datasets_dtype(hdf):
    datasets_names = hdf5_datasets_names(hdf)
    return {dataset_name: hdf[dataset_name].dtype for dataset_name in datasets_names}


# Attributes
def hdf5_objects_attrs(hdf, parser=parse_attr_string):
    dict_hdf = hdf5_objects(hdf)
    return {k: parse_HDF5_GPM_attributes(v, parser=parser) for k, v in dict_hdf.items()}


def hdf5_groups_attrs(hdf, parser=parse_attr_string):
    dict_hdf = hdf5_groups(hdf)
    return {k: parse_HDF5_GPM_attributes(v, parser=parser) for k, v in dict_hdf.items()}


def hdf5_datasets_attrs(hdf, parser=parse_attr_string):
    dict_hdf = hdf5_datasets(hdf)
    return {k: parse_HDF5_GPM_attributes(v, parser=parser) for k, v in dict_hdf.items()}


def hdf5_file_attrs(hdf, parser=parse_attr_string):
    return parse_HDF5_GPM_attributes(hdf, parser=parser)
