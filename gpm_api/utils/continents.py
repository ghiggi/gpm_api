#!/usr/bin/env python3
"""
Created on Mon Jan 16 18:04:41 2023

@author: ghiggi
"""
import difflib

CONTINENT_EXTENT_DICT = {
    "Africa": [-18.042, 51.292, -40.833, 37.092],
    "Antarctica": [-180, 180, -90, -60],
    "Arctic": [-180, 180, 60, 90],
    "Asia": [35, 180, 5, 80],
    "Australia": [105, 180, -55, 12],
    "Oceania": [105, 180, -55, 12],
    "Europe": [-30, 40, 34, 72],
    "North America": [-180, -52, 5, 83],
    "South America": [-85, -30, -60, 15],
}


def get_continent_extent(name):
    # TODO: we could create a dictionary class which
    # - optionally is key unsensitive for get method !!!
    # - provide suggestions ...
    # --> Could be reused also when searching for GPM products, ...
    # -------------------------------------------------------------------------.
    # Check country format
    if not isinstance(name, str):
        raise TypeError("Please provide the continent name as a string.")

    # Create same dictionary with lower case keys
    continent_lower_extent_dict = {s.lower(): v for s, v in CONTINENT_EXTENT_DICT.items()}
    # Get list of valid continents
    valid_continent = list(CONTINENT_EXTENT_DICT.keys())
    valid_continent_lower = list(continent_lower_extent_dict)
    if name.lower() in valid_continent_lower:
        extent = continent_lower_extent_dict[name.lower()]
        # TODO:
        # - add 0.5° degree buffer
        # - ensure extent is correct
        return extent
    else:
        possible_match = difflib.get_close_matches(name, valid_continent, n=1, cutoff=0.6)
        if len(possible_match) == 0:
            raise ValueError("Provide a valid continent name.")
        else:
            possible_match = possible_match[0]
            raise ValueError(
                f"No matching continent. Maybe are you looking for '{possible_match}'?"
            )
