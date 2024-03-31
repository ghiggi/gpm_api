# -----------------------------------------------------------------------------.
# MIT License

# Copyright (c) 2024 GPM-API developers
#
# This file is part of GPM-API.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -----------------------------------------------------------------------------.
"""This module test the YAML reader and writer utilities."""

import os

import yaml

from gpm.utils.yaml import read_yaml, write_yaml

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
