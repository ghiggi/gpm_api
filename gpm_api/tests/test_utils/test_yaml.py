#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 18:14:38 2024

@author: ghiggi
"""
import os
import yaml
from gpm_api.utils.yaml import read_yaml, write_yaml

# Sample data for testing
sample_data = dictionary = {
    "key1": "value1",
    "key2": 2,
    "key3": 3.0,
    "key4": ["value4"],
    "key5": [5],
    "key6": None,
    "key7": "",
}


def test_read_yaml(tmp_path):
    """Test read_yaml."""

    # Create a temporary file path
    tmp_filepath = tmp_path / "test.yaml"

    with tmp_filepath.open("w") as f:
        yaml.safe_dump(sample_data, f)

    # Read the data back
    read_data = read_yaml(tmp_filepath)

    # Check if the read data matches the original data
    assert read_data == sample_data, "Data read from YAML does not match expected data"

    # Cleanup
    os.remove(tmp_filepath)


def test_write_yaml(tmp_path):
    "Test write_yaml." ""
    # Create a temporary file path
    tmp_filepath = tmp_path / "test.yaml"

    # Write data to the file
    write_yaml(sample_data, tmp_filepath)

    # Read the data back
    with tmp_filepath.open() as f:
        read_data = yaml.safe_load(f)

    # Check if the read data matches the original data
    assert read_data == sample_data, "Data read from YAML does not match written data"

    # Cleanup
    os.remove(tmp_filepath)
