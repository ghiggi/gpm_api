#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 16:10:52 2022

@author: ghiggi
"""
import os
import yaml
import difflib


def get_country_extent_dictionary():
    # TODO: improve relative path
    file_dir = os.path.realpath(__file__)
    base_dir = "/" + os.path.join(*file_dir.split("/")[0:-2])
    # Define file with extents dictionary
    countries_extent_fpath = os.path.join(base_dir, "etc/country_extent.yaml")
    # Read the data from the YAML file
    with open(countries_extent_fpath, "r") as infile:
        countries_extent_dict = yaml.load(infile, Loader=yaml.FullLoader)
    return countries_extent_dict


def get_country_extent(name):
    # TODO: we could create a dictionary class which
    # - optionally is key unsensitive for get method !!!
    # - provide suggestions ...
    # --> Could be reused also when searching for GPM products, ...
    # -------------------------------------------------------------------------.
    # Check country format
    if not isinstance(name, str):
        raise TypeError("Please provide the country name as a string.")
    # Get country extent dictionary
    countries_extent_dict = get_country_extent_dictionary()
    # Create same dictionary with lower case keys
    countries_lower_extent_dict = {
        s.lower(): v for s, v in countries_extent_dict.items()
    }
    # Get list of valid countries
    valid_countries = list(countries_extent_dict.keys())
    valid_countries_lower = list(countries_lower_extent_dict)
    if name.lower() in valid_countries_lower:
        extent = countries_lower_extent_dict[name.lower()]
        # TODO:
        # - add 0.5° degree buffer
        # - ensure extent is correct
        return extent
    else:
        possible_match = difflib.get_close_matches(
            name, valid_countries, n=1, cutoff=0.6
        )
        if len(possible_match) == 0:
            raise ValueError("Provide a valid country name.")
        else:
            possible_match = possible_match[0]
            raise ValueError(
                f"No matching country. Maybe are you looking for '{possible_match}'?"
            )
