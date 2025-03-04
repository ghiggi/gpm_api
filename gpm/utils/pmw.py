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

import gpm
from gpm.io.checks import check_sensor
from gpm.utils.list import flatten_list
from gpm.utils.xarray import (
    get_default_variable,
    get_xarray_variable,
)
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

    def has_same_polarization(self, other: "PMWFrequency") -> bool:
        """Return True if the polarization is the same as `other`."""
        return self.polarization == other.polarization

    def has_same_offset(self, other: "PMWFrequency") -> bool:
        """Return True if offset is the same as `other`."""
        if self.offset is None and other.offset is None:
            return True
        if self.offset is None and other.offset is not None:
            return False
        if self.offset is not None and other.offset is None:
            return False
        return self.offset == other.offset

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

    - Has the same center frequency.
    - Has the opposite polarization (e.g., V vs. H or QV vs. QH).

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


PCT_COEFFICIENTS = {  # (tolerance, coef)
    PMWFrequency(center_frequency=89): (5, 0.7),  # 91 of SSMIS, 85 of SSMI
    PMWFrequency(center_frequency=36): (3, 1.15),  # 36.5, 37
    PMWFrequency(center_frequency=19): (1, 1.40),  # 18.7 OK  . AVOID 21, 22, 23.8 ?
    PMWFrequency(center_frequency=10): (3, 1.50),
}


def get_pct_coefficient(center_frequency):
    for freq, (tol, coef) in PCT_COEFFICIENTS.items():
        if freq.has_same_center_frequency(PMWFrequency(center_frequency=center_frequency), tol=tol):
            return coef
    raise ValueError(f"PCT coefficients are not available for the {center_frequency} GHz frequency.")


def normalize_channel(arr, vmin, vmax, vmin_dynamic, vmax_dynamic, invert):
    """Normalize array between 0 and 1 with optional inversion and dynamic min/max adjustment.

    Parameters
    ----------
    arr : array-like
        Input array to be normalized.
    vmin : float
        Minimum value for normalization.
    vmax : float
        Maximum value for normalization.
    vmin_dynamic : bool
        If True, update vmin to the maximum of the provided vmin and the array minimum.
    vmax_dynamic : bool
        If True, update vmax to the minimum of the provided vmax and the array maximum.
    invert : bool
        If True, invert the normalized values (i.e., compute 1 - normalized array).

    Returns
    -------
    norm_arr : array-like
        Normalized array with values clipped between 0 and 1.
    """
    # Define dynamic vmin and vmax if asked
    if vmax_dynamic:
        vmax = np.minimum(vmax, arr.max())  # .item() not compatible with dask
    if vmin_dynamic:
        vmin = np.maximum(vmin, arr.min())  # .item() not compatible with dask
    # Define range
    vrange = vmax - vmin
    # Retrieve normalized arrays
    norm_arr = (arr - vmin) / vrange
    norm_arr = norm_arr.clip(0, 1)
    # Invert if asked
    if invert:
        norm_arr = 1 - norm_arr
    return norm_arr


def get_brightness_temperature(xr_obj, variable):
    """Retrieve the brightness temperature data array from an xarray object.

    Parameters
    ----------
    xr_obj : xarray.Dataset or xarray.DataArray
        Input xarray object containing brightness temperature data.
    variable : str or None
        Variable name to extract. If None, a default variable is determined based on possible options.

    Returns
    -------
    dataarray : xarray.DataArray
        DataArray of the brightness temperature.
    """
    if variable is None and isinstance(xr_obj, xr.Dataset):
        variable = get_default_variable(xr_obj, possible_variables=["Tb", "Tc"])
    return get_xarray_variable(xr_obj, variable=variable)


def get_frequencies_polarized_pairs(ds):
    """Get center frequency of polarized pairs."""
    # Retrieve available frequencies
    pmw_frequencies = [PMWFrequency.from_string(freq) for freq in ds["pmw_frequency"].data]
    # Retrieve polarized frequencies couples
    dict_polarization_pairs = find_polarization_pairs(pmw_frequencies)
    return list(dict_polarization_pairs)


def get_available_pct_features(ds):
    """Get list of available PCT features."""
    return [f"PCT_{center_freq}" for center_freq in get_frequencies_polarized_pairs(ds) if float(center_freq) < 95]


def get_available_pd_features(ds):
    """Get list of available PCT features."""
    return [f"PD_{center_freq}" for center_freq in get_frequencies_polarized_pairs(ds)]


def get_available_pr_features(ds):
    """Get list of available PR features."""
    return [f"PR_{center_freq}" for center_freq in get_frequencies_polarized_pairs(ds)]


def get_pmw_channel(xr_obj, name, variable=None):
    """Extract a specific PMW (Passive Microwave) channel from an xarray object.

    Parameters
    ----------
    xr_obj : xarray.Dataset or xarray.DataArray
        PMW L1B or L1C product containing the brightness temperature variable.
    name : str or PMWFrequency
        PMW channel name or PMWFrequency object representing the desired frequency.
    variable : str, optional
        Variable name to extract from the xarray object. If None, the default variable is used.

    Returns
    -------
    dataarray : xarray.DataArray
        DataArray corresponding to the selected PMW channel.
    """
    # Check name type
    if not isinstance(name, (str, PMWFrequency)):
        raise TypeError("Parameter 'name' must be either a string or a PMWFrequency object.")
    # Ensure to get a brightness temperature dataarray
    da = get_brightness_temperature(xr_obj, variable=variable)
    # Retrieve matching frequency
    pmw_frequencies_str = da["pmw_frequency"].data
    pmw_frequency = PMWFrequency.from_string(name) if not isinstance(name, (PMWFrequency)) else name
    pmw_frequencies = [PMWFrequency.from_string(s) for s in pmw_frequencies_str]
    matched_frequency = find_closely_matching_frequency(pmw_frequency, pmw_frequencies, center_frequency_tol=2)
    if matched_frequency is None:
        raise ValueError(
            f"Requested PMW frequency '{pmw_frequency.to_string()}' is not available. "
            f"Available frequencies: {list(pmw_frequencies_str)}.",
        )

    # freq_str = name.to_string() if isinstance(name, PMWFrequency) else name
    # if freq_str not in pmw_frequencies_str:
    #     raise ValueError(
    #         f"Requested PMW frequency '{freq_str}' is not available. "
    #         f"Available frequencies: {list(pmw_frequencies_str)}.",
    #     )
    return da.sel(pmw_frequency=matched_frequency.to_string()).squeeze()


def find_closely_matching_frequency(pmw_frequency, pmw_frequencies, center_frequency_tol):
    """Find the closely matching frequency within a set of frequencies."""
    for freq in pmw_frequencies:
        if (
            pmw_frequency.has_same_center_frequency(freq, tol=center_frequency_tol)
            and pmw_frequency.has_same_polarization(freq)
            and pmw_frequency.has_same_offset(freq)
        ):
            return freq
    return None


def find_closely_matching_center_frequency(center_frequency, center_frequencies):
    """Find the closely matching center frequency within a set of frequencies."""
    center_frequency = PMWFrequency(center_frequency=center_frequency)
    center_frequencies = [PMWFrequency(center_frequency=freq) for freq in center_frequencies]
    for freq in center_frequencies:
        if center_frequency.has_same_center_frequency(freq, tol=2):
            return freq.center_frequency_str
    return None


def _get_polarized_pairs_datarray(ds, name, feature_prefix):
    """Return the polarized pair of V and H DataArray."""
    # Check valid feature name
    if not name.startswith(f"{feature_prefix}_"):
        raise ValueError(f"Specify {feature_prefix} name as {feature_prefix}_<center_freq>")
    # Retrieve center frequency
    center_frequency_str = name.replace(f"{feature_prefix}_", "")
    # Retrieve available frequencies
    pmw_frequencies = [PMWFrequency.from_string(freq) for freq in ds["pmw_frequency"].data]
    # Retrieve available polarized frequencies couples
    dict_polarization_pairs = find_polarization_pairs(pmw_frequencies)
    # If feature not available, raise error
    key = find_closely_matching_center_frequency(
        center_frequency=center_frequency_str,
        center_frequencies=list(dict_polarization_pairs),
    )
    if key is None:
        valid_pct = [f"{feature_prefix}_{c}" for c in list(dict_polarization_pairs)]
        raise ValueError(f"{name} is unavailable. Available {feature_prefix} are {valid_pct}.")
    # Retrieve polarization pair
    freq_v, freq_h = dict_polarization_pairs[key]
    da_v = get_pmw_channel(ds, name=freq_v)
    da_h = get_pmw_channel(ds, name=freq_h)
    return da_v, da_h


def get_pct(ds, name):
    """Compute a PCT (Polarization Corrected Temperature) from a PMW L1C product.

    Parameters
    ----------
    ds : xarray.Dataset
        PMW L1C product containing the brightness temperature variable.
    name : str
        Name of the PCT feature in the format 'PCT_<center_freq>'.

    Returns
    -------
    dataarray : xarray.DataArray
        DataArray representing the computed PCT.

    """
    # Retrieve V and H Tb DataArrays
    da_v, da_h = _get_polarized_pairs_datarray(ds, name=name, feature_prefix="PCT")
    # Retrieve PCTs coefficient
    center_frequency_str = name.replace("PCT_", "")
    coef = get_pct_coefficient(center_frequency_str)
    # Compute PCT
    da_pct = (1 + coef) * da_v - coef * da_h
    return da_pct.squeeze()


def get_pd(ds, name):
    """Compute a PD (Polarization Difference) from a PMW L1C product.

    Parameters
    ----------
    ds : xarray.Dataset
        PMW L1C product containing the brightness temperature variable.
    name : str
        Name of the PD feature in the format 'PD_<center_freq>'.

    Returns
    -------
    dataarray : xarray.DataArray
        DataArray representing the computed PD.
    """
    # Retrieve V and H Tb DataArrays
    da_v, da_h = _get_polarized_pairs_datarray(ds, name=name, feature_prefix="PD")
    # Compute PD
    da_pd = da_v - da_h
    return da_pd.squeeze()


def get_pr(ds, name):
    """Compute a PR (Polarization Ratio) from a PMW L1C product.

    Parameters
    ----------
    ds : xarray.Dataset
        PMW L1C product containing the brightness temperature variable.
    name : str
        Name of the PR feature in the format 'PR_<center_freq>'.

    Returns
    -------
    dataarray : xarray.DataArray
        DataArray representing the computed PR.
    """
    # Retrieve V and H Tb DataArrays
    da_v, da_h = _get_polarized_pairs_datarray(ds, name=name, feature_prefix="PR")
    # Compute PD
    da_pr = da_v / da_h
    return da_pr.squeeze()


def get_pmw_feature(ds, name):
    """
    Retrieve or compute a PMW feature from an L1C PMW product.

    This function can handle:
      1. Simple PMW feature names such as "PCT_37", "PD_37", or direct channel names.
      2. Complex expressions using PMW feature names combined with basic arithmetic
         operations.

    Parameters
    ----------
    ds : xarray.Dataset
        PMW L1C product containing the brightness temperature variable.
    name : str
        A single PMW feature name (e.g. "PCT_37", "PD_37", "37V", ...)
        or a string expression combining multiple PMW feature names via
        mathematical operators (+, -, *, /) and parentheses
        (e.g. ```"(PCT_37 + PCT_19)/(PCT_37 - PCT_19)" ```).

    Returns
    -------
    xarray.DataArray
        DataArray corresponding to the requested or computed PMW feature.
    """
    # Check if the input is a simple feature name or a compound expression
    if is_simple_feature_name(name):
        # Handle direct feature retrieval
        return _get_simple_feature(ds, name)
    # Handle expression
    return _evaluate_feature_expression(ds, name)


def is_simple_feature_name(name):
    """Determine if `name` is a single PMW feature or a more complex expression.

    A 'simple' feature name is assumed to not contain any arithmetic
    symbols (+, -, *, /) nor parentheses.

    """
    # If we see +, -, *, /, or parentheses, treat as an expression
    if any(symbol in name for symbol in "+-*/()"):  # noqa: SIM103
        return False
    return True


def _get_simple_feature(ds, name):
    """Internal helper to retrieve a simple PMW feature.

    Simple PMW features are PCT, PD, or a PMW channel.
    """
    # Check known prefixes
    if name.startswith("PCT"):
        return get_pct(ds, name)
    if name.startswith("PD"):
        return get_pd(ds, name)
    # Fall back to channel retrieval
    return get_pmw_channel(ds, name)


def _is_number(string):
    """Check if a string can be interpreted as a float/int."""
    try:
        float(string)
        return True
    except ValueError:
        return False


ALLOWED_OPERATORS = {"+", "-", "*", "/", "**"}
PARENS = {"(", ")"}


def _evaluate_feature_expression(ds, expression):
    """Safely evaluate an expression containing multiple PMW features.

    Parameters
    ----------
    ds : xarray.Dataset
        PMW L1C product.
    expression : str
        A string expression, for example:
        - "PCT_37 - PCT_19"
        - "(PCT_37 + PCT_19)/(PCT_37 - PCT_19)"
        - "PCT_37**2 + 273.15"

    Returns
    -------
    xr.DataArray
        DataArray resulting from the expression evaluation.
    """
    # A regex that will capture:
    #   - ** (exponent)
    #   - single-character operators + - * / ( )
    #   - sequences of alphanumeric + underscore + optional dot (e.g., "PCT_37", "37V", "273.15")
    token_pattern = re.compile(r"(\*\*|[+\-*/()]|[A-Za-z0-9_\.]+)")

    # 1. Tokenize the expression
    tokens = token_pattern.findall(expression)
    if not tokens:
        raise ValueError(f"Invalid or empty expression: {expression}")

    feature_dict = {}
    var_counter = 0
    new_tokens = []

    for token in tokens:
        # Check parentheses
        if token in PARENS:
            new_tokens.append(token)
            continue

        # Check operators
        if token in ALLOWED_OPERATORS:
            new_tokens.append(token)
            continue

        # Check if token is numeric
        if _is_number(token):
            new_tokens.append(token)
            continue

        # Otherwise, assume it's a PMW feature name
        # (e.g. "PCT_37", "PD_19", or direct channel "37V")
        var_key = f"_var{var_counter}"
        feature_dict[var_key] = _get_simple_feature(ds, token)
        new_tokens.append(var_key)
        var_counter += 1

    # 2. Rebuild expression with strictly allowed tokens or variable placeholders
    safe_expression = " ".join(new_tokens)

    # 3. Evaluate the expression with xarray DataArrays in a restricted namespace
    safe_namespace = {
        "__builtins__": None,  # Remove builtins to reduce security concerns
    }
    try:
        result = eval(safe_expression, safe_namespace, feature_dict)
    except Exception as exc:
        raise ValueError(
            f"Failed to evaluate expression '{expression}'. " f"Check syntax and feature names. Original error: {exc}",
        )

    # 4. Verify result is an xarray DataArray
    if not isinstance(result, xr.DataArray):
        raise TypeError(
            f"Expression did not result in an xr.DataArray. Actual type: {type(result)}",
        )

    return result


def get_pmw_rgb_receipts(sensor):
    """Return the RGB composite receipts available for a PMW sensor."""
    sensor = check_sensor(sensor)
    filepath = os.path.join(gpm._root_path, "gpm", "etc", "pmw", "composites", f"{sensor}.yaml")
    return read_yaml(filepath)


def create_rgb_composite(ds, receipt):
    """Generate an RGB composite from a 1C PMW dataset using the provided receipt.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing PMW data.
    receipt : dict
        Dictionary containing configuration for each channel ('R', 'G', 'B') and optional global normalization.
        Each channel configuration should include the following keys:
            - 'name': The PMW feature name.
            - 'vmin': Minimum value for normalization.
            - 'vmax': Maximum value for normalization.
            - 'vmin_dynamic': Boolean flag for dynamic minimum adjustment.
            - 'vmax_dynamic': Boolean flag for dynamic maximum adjustment.
            - 'invert': Boolean flag to invert the channel.
        Optionally, the dictionary may include a 'global_normalization' key (bool).
        If True, the vmin and vmax of channels with vmin_dynamic and vmax_dynamic equal False are
        updated with the minimum and maximum values across all such channels.
        The specified vmin and vmax are used to bound the update of vmin and vmax.

    Returns
    -------
    dataarray : xarray.DataArray
        An RGB composite DataArray with a coordinate 'rgb' corresponding to channels ['r', 'g', 'b'].
    """
    receipt = receipt.copy()
    # Retrieve input channels
    dict_channels = {c: get_pmw_feature(ds, receipt[c]["name"]) for c in ["R", "G", "B"]}
    # Check if perform global normalization
    if receipt.get("global_normalization", False):
        # Compute global minimum and maximum across channels to not dynamically normalize
        global_min = min(da.min() for key, da in dict_channels.items() if not receipt[key]["vmin_dynamic"])
        global_max = max(da.max() for key, da in dict_channels.items() if not receipt[key]["vmax_dynamic"])
        # Update each channel's normalization parameters
        for key in ["R", "G", "B"]:
            if not receipt[key]["vmin_dynamic"]:
                if receipt[key]["vmin"] is None:
                    receipt[key]["vmin"] = global_min
                else:
                    receipt[key]["vmin"] = np.maximum(receipt[key]["vmin"], global_min)
            if not receipt[key]["vmax_dynamic"]:
                if receipt[key]["vmax"] is None:
                    receipt[key]["vmax"] = global_max
                else:
                    receipt[key]["vmax"] = np.minimum(receipt[key]["vmax"], global_max)

    # Retrieve normalized channels
    dict_rgb = {
        c: normalize_channel(
            da_channel,
            vmin=receipt[c]["vmin"],
            vmax=receipt[c]["vmax"],
            vmin_dynamic=receipt[c]["vmin_dynamic"],
            vmax_dynamic=receipt[c]["vmax_dynamic"],
            invert=receipt[c]["invert"],
        )
        for c, da_channel in dict_channels.items()
    }

    # Concatenate channels
    da_rgb = xr.concat(dict_rgb.values(), coords="minimal", dim="rgb", compat="override")
    da_rgb = da_rgb.assign_coords({"rgb": ["r", "g", "b"]})
    return da_rgb
