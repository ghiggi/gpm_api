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
"""This module defines a YAML file reader and writer."""
import yaml


class NoAliasDumper(yaml.SafeDumper):
    """YAML Safe Dumper class avoiding use of aliases."""

    def ignore_aliases(self, data):  # noqa ARG002
        """Ignore aliases."""
        return True


def read_yaml(filepath: str) -> dict:
    """Read a YAML file into a dictionary.

    Parameters
    ----------
    filepath : str
        Input YAML file path.

    Returns
    -------
    dict
        Dictionary with the attributes read from the YAML file.

    """
    with open(filepath) as f:
        return yaml.safe_load(f)


def write_yaml(dictionary, filepath, sort_keys=False):
    """Write a dictionary into a YAML file.

    Parameters
    ----------
    dictionary : dict
        Dictionary to write into a YAML file.

    """
    with open(filepath, "w") as f:
        yaml.dump(dictionary, f, sort_keys=sort_keys, Dumper=NoAliasDumper)
