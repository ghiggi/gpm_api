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
"""This module provides PMW utilities."""
import functools
import os
import re
from functools import total_ordering

import numpy as np
import xarray as xr

from gpm.utils.list import flatten_list
from gpm.utils.yaml import read_yaml


@functools.cache
def get_pmw_frequency_dict():
    """Get PMW info dictionary."""
    from gpm import _root_path

    filepath = os.path.join(_root_path, "gpm", "etc", "pmw", "frequencies.yaml")
    return read_yaml(filepath)


@functools.cache
def get_pmw_frequency(sensor, scan_mode):
    """Get product info dictionary."""
    pmw_dict = get_pmw_frequency_dict()
    return pmw_dict[sensor][scan_mode]


def available_pmw_frequencies(sensor):
    from gpm.io.products import available_sensors

    # Check valid sensor
    available_pmw_sensors = available_sensors(product_categories="PMW")
    if sensor not in available_pmw_sensors:
        raise ValueError(f"{sensor} is not a valid PMW sensor. Valid sensors: {available_pmw_sensors}. ")

    # Retrieve dictionary with frequencies for each scan mode
    pmw_dict = get_pmw_frequency_dict()
    sensor_dict = pmw_dict[sensor]

    # Retrieve frequencies list (across all scan modes)
    frequencies = list(sensor_dict.values())
    frequencies = flatten_list(frequencies)
    frequencies = np.unique(frequencies).tolist()

    return frequencies


####--------------------------------------------------------------------------------.
def strip_trailing_zero_decimals(num: float) -> str:
    """Strip trailing zeros from a float, e.g. 183.0 -> '183'."""
    s = f"{num}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")  # remove trailing zeros and '.'
    return s


@total_ordering
class PMWFrequency:
    """
    Class to represent a Passive Microwave frequency channel.

    Attributes
    ----------
    center_frequency : float or str
        The (nominal) center frequency in GHz.
    polarization : str
        Polarization code, e.g. 'V', 'H', 'QV', 'QH'.
    offset : float or None
        Offset from the center frequency in GHz (e.g., 3 for 183±3 GHz). None if not applicable.
    """

    # Map for flipping polarization
    _OPPOSITE_POLARIZATION = {
        "": "",
        "V": "H",
        "H": "V",
        "QV": "QH",
        "QH": "QV",
    }

    def __init__(self, center_frequency: float, polarization: str = "", offset=None):

        # Check valid polarization
        if polarization not in self._OPPOSITE_POLARIZATION:
            valid_polarization = ", ".join(self._OPPOSITE_POLARIZATION.keys())
            raise ValueError(f"Invalid polarization '{polarization}'. Must be one of: {valid_polarization}.")

        self.polarization = polarization
        self.center_frequency = float(center_frequency)
        self.offset = float(offset) if offset is not None else None

    @property
    def center_frequency_str(self):
        """Return center frequency string."""
        return strip_trailing_zero_decimals(self.center_frequency)

    @property
    def offset_str(self):
        """Return center frequency offset string."""
        if self.offset is not None and abs(self.offset) > 1e-9:
            return strip_trailing_zero_decimals(self.offset)
        return ""

    @classmethod
    def from_string(cls, string: str) -> "PMWFrequency":
        """
        Create a PMWFrequency object from a string like '10.65V', '18.7H', or '183V3'.

        Pattern:
            1. Numeric frequency (integer or float)
            2. Polarization ('V', 'H', 'QV', 'QH')
            3. Optional numeric offset, e.g. '3'

        Examples
        --------
            '10.65V'   -> center_frequency=10.65, polarization='V', offset=None
            '183V3'    -> center_frequency=183, polarization='V', offset=3
            '183.31QH7.5' -> center_frequency=183.31, polarization='QH', offset=7.5
        """
        if not isinstance(string, (str, np.str_)):
            raise TypeError("Expecting a string.")

        # Regex breakdown:
        # ^(\d+(?:\.\d+)?)     -> group 1: one or more digits, optional decimal part
        # (V|H|QV|QH)         -> group 2: exactly 'V', 'H', 'QV', or 'QH'
        # (\d+(?:\.\d+)?)?$   -> group 3: optional one or more digits, optional decimal part
        pattern = r"^(\d+(?:\.\d+)?)(V|H|QV|QH)(\d+(?:\.\d+)?)?$"
        match = re.match(pattern, string)
        if not match:
            raise ValueError(f"String '{string}' is not a valid PMW channel string.")

        freq_str, pol_str, offset_str = match.groups()
        freq = float(freq_str)
        offset = float(offset_str) if offset_str is not None else None
        return cls(center_frequency=freq, polarization=pol_str, offset=offset)

    def title(self) -> str:
        """Return a nicely formatted string representation of the frequency channel."""
        freq_str = self.center_frequency_str
        offset_str = self.offset_str
        if offset_str != "":
            return f"{freq_str} ± {offset_str} GHz ({self.polarization})"
        return f"{freq_str} GHz ({self.polarization})"

    def __repr__(self) -> str:
        """Return a concise string representation."""
        freq_str = self.center_frequency_str
        offset_str = self.offset_str
        if offset_str != "":
            title = f"{freq_str} ± {offset_str} GHz ({self.polarization})"
        else:
            title = f"{freq_str} GHz ({self.polarization})"
        return f"<PMWFrequency: {title}>"

    def to_string(self) -> str:
        """
        Recreate the original acronym string from the PMWFrequency object.

        Examples
        --------
            - "10.65V"
            - "183V3"
            - "89QV7.5"
        """
        freq_str = self.center_frequency_str
        offset_str = self.offset_str
        if offset_str != "":
            return f"{freq_str}{self.polarization}{offset_str}"
        return f"{freq_str}{self.polarization}"

    def __hash__(self) -> int:
        """
        Return a hash value based on the string returned by `to_string()`.

        This ensures that two PMWFrequency objects which produce the same
        string representation share the same hash.
        """
        return hash(self.to_string())

    def opposite_polarization(self):
        """Return a new PMWFrequency object with flipped polarization (V <-> H, QV <-> QH).

        Examples
        --------
            10.65V -> 10.65H
            183QH3 -> 183QV3
        """
        if self.polarization in self._OPPOSITE_POLARIZATION:
            new_pol = self._OPPOSITE_POLARIZATION[self.polarization]
            return PMWFrequency(self.center_frequency, new_pol, self.offset)
        return None

    def has_same_center_frequency(self, other: "PMWFrequency", tol: float = 1e-6) -> bool:
        """Return True if the center frequency is the same as `other` within a specified tolerance."""
        return abs(self.center_frequency - other.center_frequency) < tol

    @property
    def wavelength(self) -> float:
        """Returns the channel wavelength in meters.

        The wavelength is computed as c / (f * 1e9),
        where c ~ 3e8 m/s and f is the center_frequency in GHz.
        """
        c_m_s = 3.0e8  # Speed of light in m/s
        return c_m_s / (self.center_frequency * 1e9)

    def __eq__(self, other: object) -> bool:
        """Define equality between two PMWFrequency objects.

        It performs:
          - approximate comparison (1e-9) for center_frequency
          - approximate comparison (1e-9) for offset (treat None as 0)
          - exact match for polarization.
        """
        if not isinstance(other, PMWFrequency):
            return NotImplemented

        tol = 1e-9
        offset_self = 0.0 if self.offset is None else self.offset
        offset_other = 0.0 if other.offset is None else other.offset

        same_freq = abs(self.center_frequency - other.center_frequency) < tol
        same_off = abs(offset_self - offset_other) < tol
        same_pol = self.polarization == other.polarization
        return same_freq and same_off and same_pol

    def __lt__(self, other: object) -> bool:
        """Return True if self is less than other by comparing center frequency, offset, and polarization.

        Polarization V or QV takes precedence over H
        """
        if not isinstance(other, PMWFrequency):
            return NotImplemented
        offset_self = 0.0 if self.offset is None else self.offset
        offset_other = 0.0 if other.offset is None else other.offset

        # Compare by center frequency first.
        if self.center_frequency != other.center_frequency:
            return self.center_frequency < other.center_frequency

        # If center frequencies are equal, compare offsets (treat None as 0.0).
        offset_self = 0.0 if self.offset is None else self.offset
        offset_other = 0.0 if other.offset is None else other.offset
        if offset_self != offset_other:
            return offset_self < offset_other

        # If offsets are also equal, compare polarization.
        # Custom order: V and QV are considered lower than H and QH
        polarization_order = {"": 0, "V": 1, "QV": 2, "H": 3, "QH": 4}
        return polarization_order.get(self.polarization) < polarization_order.get(other.polarization)


def find_polarization_pairs(pmw_frequencies):
    """Identify polariazion pairs of PMWFrequency objects.

    The PMWFrequency objects must share the same center frequency
    but differ in polarization (e.g., vertical vs. horizontal).

    This function iterates through each PMWFrequency in the input list and
    attempts to match it with another PMWFrequency that:
      1. Has the same center frequency.
      2. Has the opposite polarization (e.g., V vs. H or QV vs. QH).

    Once a valid pair is found, it is stored in a dictionary keyed by the
    shared center frequency. For consistent ordering of pairs, any item with
    vertical polarization (e.g., "V", "QV") is placed first in the tuple,
    followed by the corresponding horizontal polarization ("H", "QH").

    Parameters
    ----------
    pmw_frequencies : list of PMWFrequency
        A list of PMWFrequency objects to be examined for pairs.

    Returns
    -------
    dict
        A dictionary where keys are center frequencies (float or int),
        and values are 2-tuples of PMWFrequency objects in (vertical, horizontal) order.
        If no match is found for a given frequency, that frequency is not included
        in the dictionary.
    """
    dict_pairs = {}
    for freq in pmw_frequencies:
        for other_freq in pmw_frequencies:
            if (
                freq != other_freq
                and freq.has_same_center_frequency(other_freq)
                and freq.opposite_polarization() == other_freq
            ):
                center_frequency_str = freq.center_frequency_str
                if center_frequency_str not in dict_pairs:
                    # Enforce V and H order
                    if freq.polarization in ["V", "QV"]:
                        dict_pairs[center_frequency_str] = (freq, other_freq)
                    else:
                        dict_pairs[center_frequency_str] = (other_freq, freq)
    return dict_pairs


def get_rgb_composites_receipts(sensor):
    """Return the available RGB composites receipts for a PMW sensor.

    The receipts are derived from original Stephen Joe Munchak code available at
    https://github.com/jmunchak/plot_GPM_RGB/blob/main/Plot%20GPM%20RGB%20imagery.ipynb
    """
    # TODO: Create a YAML RGB composite per sensor
    dict_receipts = {
        # -------------------------------------------
        #### GMI
        "GMI": {
            "10 + 18 GHz": {
                "channels": {
                    "R": "19V",
                    "G": "19H",
                    "B": "10H",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
            "19 + 23 GHz": {
                "channels": {
                    "R": "23V",
                    "G": "19V",
                    "B": "19H",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
            "36 + 89 GHz": {
                "channels": {
                    "R": "89V",
                    "G": "89H",
                    "B": "37H",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
            "165 + 183 GHz": {
                "channels": {
                    "R": "165V",
                    "G": "183V7",
                    "B": "183V3",
                },
                "vmin": 50,
                "vmax": 300,
                "vmax_dynamic": True,
                "vmin_dynamic": True,
            },
            "36 + 88 + 165 GHz": {
                "channels": {
                    "R": "165V",
                    "G": "89V",
                    "B": "37H",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
        },
        # -------------------------------------------
        #### TMI
        "TMI": {
            "10 + 18 GHz": {
                "channels": {
                    "R": "19V",
                    "G": "19H",
                    "B": "10H",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
            "19 + 21 GHz": {
                "channels": {
                    "R": "21V",
                    "G": "19V",
                    "B": "19H",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
            "36 + 89 GHz": {
                "channels": {
                    "R": "89V",
                    "G": "89H",
                    "B": "37H",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
        },
        # -------------------------------------------
        #### SSMIS
        "SSMIS": {
            "19 + 22 GHz": {
                "channels": {
                    "R": "22V",
                    "G": "19V",
                    "B": "19H",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
            "37 + 91 GHz": {
                "channels": {
                    "R": "91V",
                    "G": "91H",
                    "B": "37H",
                },
                "vmin": 65,
                "vmax": 300,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
            "183 GHz": {
                "channels": {
                    "R": "183H7",
                    "G": "183H3",
                    "B": "183H1",
                },
                "vmin": 100,
                "vmax": 300,
                "vmax_dynamic": True,
                "vmin_dynamic": True,
            },
        },
        # -------------------------------------------
        #### SSMI
        "SSMI": {
            "19 + 22 GHz": {
                "channels": {
                    "R": "22V",
                    "G": "19V",
                    "B": "19H",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
            "37 + 85 GHz": {
                "channels": {
                    "R": "85V",
                    "G": "85H",
                    "B": "37H",
                },
                "vmin": 65,
                "vmax": 300,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
        },
        # -------------------------------------------
        #### AMSRE
        "AMSRE": {
            "10 + 18 GHz": {
                "channels": {
                    "R": "18.7V",
                    "G": "18.7H",
                    "B": "10.65H",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
            "19 + 23 GHz": {
                "channels": {
                    "R": "23.8V",
                    "G": "18.7V",
                    "B": "18.7H",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
            "37 + 89 GHz": {
                "channels": {
                    "R": "89V",
                    "G": "89H",
                    "B": "36.5H",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
        },
        # -------------------------------------------
        #### AMSR2
        "AMSR2": {
            "10 + 18 GHz": {
                "channels": {
                    "R": "18.7V",
                    "G": "18.7H",
                    "B": "10.65H",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
            "18 + 23 GHz": {
                "channels": {
                    "R": "23.8V",
                    "G": "18.7V",
                    "B": "18.7H",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
            "37 + 89 GHz": {
                "channels": {
                    "R": "89V",
                    "G": "89H",
                    "B": "36.5H",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
        },
        # -------------------------------------------
        #### ATMS
        "ATMS": {
            "31 + 88 + 165 GHz": {
                "channels": {
                    "R": "165.5QH",
                    "G": "88.2QV",
                    "B": "31.4QV",
                },
                "vmin": 65,
                "vmax": 320,
                "vmax_dynamic": False,
                "vmin_dynamic": False,
            },
            "165 + 183 GHz": {
                "channels": {
                    "R": "165.5QH",
                    "G": "183.31QH7",
                    "B": "183.31QH3",
                },
                "vmin": 50,
                "vmax": 300,
                "vmax_dynamic": True,
                "vmin_dynamic": True,
            },
            "183 GHz": {
                "channels": {
                    "R": "183.31QH7",
                    "G": "183.31QH3",
                    "B": "183.31QH1",
                },
                "vmin": 100,
                "vmax": 300,
                "vmax_dynamic": True,
                "vmin_dynamic": True,
            },
        },
        # -------------------------------------------
        #### AMSUB
        "AMSUB": {
            "150 + 183 GHz": {
                "channels": {
                    "R": "150V0.9",
                    "G": "183.31V7",
                    "B": "183.31V1",
                },
                "vmin": 100,
                "vmax": 290,
                "vmax_dynamic": True,
                "vmin_dynamic": True,
            },
            "183 GHz": {
                "channels": {
                    "R": "183.31V7",
                    "G": "183.31V3",
                    "B": "183.31V1",
                },
                "vmin": 100,
                "vmax": 300,
                "vmax_dynamic": True,
                "vmin_dynamic": True,
            },
        },
        # -------------------------------------------
        #### MHS
        "MHS": {
            "157 + 183 GHz": {
                "channels": {
                    "R": "157V",
                    "G": "190.31V",
                    "B": "183.31H1",
                },
                "vmin": 100,
                "vmax": 290,
                "vmax_dynamic": True,
                "vmin_dynamic": True,
            },
            "183 GHz": {
                "channels": {
                    "R": "190.31V",
                    "G": "183.31H3",
                    "B": "183.31H1",
                },
                "vmin": 100,
                "vmax": 300,
                "vmax_dynamic": True,
                "vmin_dynamic": True,
            },
        },
        # -------------------------------------------
        #### SAPHIR
        "SAPHIR": {
            "183 GHz": {
                "channels": {
                    "R": "183.31H6.8",
                    "G": "183.31H2.8",
                    "B": "183.31H1.1",
                },
                "vmin": 100,
                "vmax": 300,
                "vmax_dynamic": True,
                "vmin_dynamic": True,
            },
        },
        # -------------------------------------------
    }

    # ----------------------------------------
    # TROPICS Water vapour composite
    # - 10, 9, 8 (vmax=290, vmin=140)
    # - 186.51 184.41 118.58 GHz

    # TROPICS Window channel composite
    # - 0, 1, 11 (vmax=300, vmin=150 for 0-1, (min=125,vmax=300 for 11)
    # - 91.655 114.50 190.31 GHz

    return dict_receipts[sensor]


def create_rgb_composite(da, receipt):
    """Create a PMW RGB composite given a receipt and a brightness temperature DataArray."""
    # Retrieve information
    channels = list(receipt["channels"].values())
    vmax = receipt["vmax"]
    vmin = receipt["vmin"]
    vmax_dynamic = receipt["vmax_dynamic"]
    vmin_dynamic = receipt["vmin_dynamic"]
    # Retrieve channels
    da_channels = da.sel(pmw_frequency=channels)
    # Redefine vmin and vmax
    if vmax_dynamic:
        vmax = np.minimum(vmax, da_channels.max())  # .item() not compatible with dask
    if vmin_dynamic:
        vmin = np.maximum(vmin, da_channels.min())  # .item() not compatible with dask
    # Define dynamic range
    vrange = vmax - vmin
    # Retrieve normalized arrays
    list_da = [(vmax - da_channels.sel(pmw_frequency=c)) / vrange for c in channels]
    # Concatenate and clip values
    da_rgb = xr.concat(list_da, dim="rgb")
    da_rgb = da_rgb.assign_coords({"rgb": ["r", "g", "b"]})
    da_rgb = da_rgb.clip(0, 1)
    return da_rgb
