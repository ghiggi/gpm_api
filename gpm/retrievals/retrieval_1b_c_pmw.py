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
"""This module contains GPM PMW 1B and 1C products community-based retrievals."""
import numpy as np
import xarray as xr

from gpm.utils.pmw import (
    PMWFrequency,
    create_rgb_composite,
    find_polarization_pairs,
    get_rgb_composites_receipts,
)
from gpm.utils.xarray import (
    get_default_variable,
    get_xarray_variable,
)

PCT_COEFFIENTS = {  # (tolerance, coef)
    PMWFrequency(center_frequency=89): (5, 0.7),  # 91 of SSMIS, 85 of SSMI
    PMWFrequency(center_frequency=36): (3, 1.15),  # 36.5, 37
    PMWFrequency(center_frequency=19): (1, 1.40),  # 18.7 OK  . AVOID 21, 22, 23.8 ?
    PMWFrequency(center_frequency=10): (3, 1.50),
}


def _retrieve_pct_coeff_dict(dict_pairs):
    dict_coeff = {}
    for f in dict_pairs:
        for freq, (tol, coef) in PCT_COEFFIENTS.items():
            if freq.has_same_center_frequency(PMWFrequency(center_frequency=f), tol=tol):
                dict_coeff[f] = coef
    return dict_coeff


def retrieve_rgb_composites(ds, variable=None):
    """Retrieve the available PMW RGB composites."""
    # Retrieve sensor
    sensor = ds.attrs["InstrumentName"]
    # Retrieve defined RGB receipts
    receipts = get_rgb_composites_receipts(sensor)
    # Retrieve DataArray with brightness temperatures
    if variable is None:
        variable = get_default_variable(ds, possible_variables=["Tb", "Tc"])
    da = get_xarray_variable(ds, variable=variable)
    # Keep only receipts for which PMW channels are available
    receipts = {
        name: receipt
        for name, receipt in receipts.items()
        if np.all(np.isin(list(receipt["channels"].values()), da["pmw_frequency"].data))
    }
    # Compute RGB composites
    list_ds_rgb = [
        create_rgb_composite(da, receipt=receipt).to_dataset(name=name) for name, receipt in receipts.items()
    ]
    ds_rgb = xr.merge(
        list_ds_rgb,
        compat="override",
    )  # deal with incompatible coords across scanModes (i.e. sunLocalTime)
    return ds_rgb


def retrieve_polarization_corrected_temperature(ds, variable=None):
    """Retrieve PMW Polarization-Corrected Temperature (PCT).

    Coefficients are taken from Cecil et al., 2018.

    References
    ----------
    Cecil, D. J., and T. Chronis, 2018.
    Polarization-Corrected Temperatures for 10-, 19-, 37-, and 89-GHz Passive Microwave Frequencies.
    J. Appl. Meteor. Climatol., 57, 2249-2265, https://doi.org/10.1175/JAMC-D-18-0022.1.
    """
    # Retrieve DataArray with brightness temperatures
    if variable is None:
        variable = get_default_variable(ds, possible_variables=["Tb", "Tc"])
    da = get_xarray_variable(ds, variable=variable)

    # Retrieve available frequencies
    pmw_frequencies = [PMWFrequency.from_string(freq) for freq in da["pmw_frequency"].data]

    # Retrieve polarized frequencies couples
    dict_polarization_pairs = find_polarization_pairs(pmw_frequencies)

    # If no combo, raise error
    if len(dict_polarization_pairs) == 0:
        pmw_frequencies = [freq.title() for freq in pmw_frequencies]
        raise ValueError(f"Impossible to compute polarized corrected temperature with channels: {pmw_frequencies}.")

    # Unstack channels
    ds_t = da.gpm.unstack_dimension(dim="pmw_frequency", prefix="", suffix="")

    # Retrieve PCTs coefficients
    dict_coeff = _retrieve_pct_coeff_dict(dict_polarization_pairs)

    # Compute PCTs
    dict_pct = {}
    for center_frequency_str, coef in dict_coeff.items():
        freq_v, freq_h = dict_polarization_pairs[center_frequency_str]
        pct_name = f"PCT_{center_frequency_str}"
        dict_pct[pct_name] = (1 + coef) * ds_t[f"{variable}{freq_v.to_string()}"] - coef * ds_t[
            f"{variable}{freq_h.to_string()}"
        ]

    # Create dataset
    ds_pct = xr.Dataset(dict_pct)
    return ds_pct


def retrieve_polarization_difference(ds, variable=None):
    """Retrieve PMW Channels Polarized Difference (PD)."""
    # Retrieve DataArray with brightness temperatures
    if variable is None:
        variable = get_default_variable(ds, possible_variables=["Tb", "Tc"])
    da = get_xarray_variable(ds, variable=variable)

    # Retrieve available frequencies
    pmw_frequencies = [PMWFrequency.from_string(freq) for freq in da["pmw_frequency"].data]

    # Retrieve polarized frequencies couples
    dict_polarization_pairs = find_polarization_pairs(pmw_frequencies)

    # If no combo, raise error
    if len(dict_polarization_pairs) == 0:
        pmw_frequencies = [freq.title() for freq in pmw_frequencies]
        raise ValueError(f"Impossible to compute polarized difference with channels: {pmw_frequencies}. No pairs.")

    # Compute PDs
    ds_t = da.gpm.unstack_dimension(dim="pmw_frequency", prefix="", suffix="")
    dict_pd = {}
    for center_frequency_str, (freq_v, freq_h) in dict_polarization_pairs.items():
        pd_name = f"PD_{center_frequency_str}"
        dict_pd[pd_name] = ds_t[f"{variable}{freq_v.to_string()}"] - ds_t[f"{variable}{freq_h.to_string()}"]

    # Create dataset
    ds_pd = xr.Dataset(dict_pd)
    return ds_pd


def retrieve_polarization_ratio(ds, variable=None):
    """Retrieve PMW Channels Polarization Ratio (PR)."""
    # Retrieve DataArray with brightness temperatures
    if variable is None:
        variable = get_default_variable(ds, possible_variables=["Tb", "Tc"])
    da = get_xarray_variable(ds, variable=variable)

    # Retrieve available frequencies
    pmw_frequencies = [PMWFrequency.from_string(freq) for freq in da["pmw_frequency"].data]

    # Retrieve polarized frequencies couples
    dict_polarization_pairs = find_polarization_pairs(pmw_frequencies)

    # If no combo, raise error
    if len(dict_polarization_pairs) == 0:
        pmw_frequencies = [freq.title() for freq in pmw_frequencies]
        raise ValueError(f"Impossible to compute polarization ratio with channels: {pmw_frequencies}. No pairs.")

    # Compute PRs
    ds_t = da.gpm.unstack_dimension(dim="pmw_frequency", prefix="", suffix="")
    dict_pr = {}
    for center_frequency_str, (freq_v, freq_h) in dict_polarization_pairs.items():
        pr_name = f"PR_{center_frequency_str}"
        dict_pr[pr_name] = ds_t[f"{variable}{freq_v.to_string()}"] / ds_t[f"{variable}{freq_h.to_string()}"]

    # Create dataset
    ds_pr = xr.Dataset(dict_pr)
    return ds_pr


#### ALIAS
retrieve_PCT = retrieve_polarization_corrected_temperature
retrieve_PD = retrieve_polarization_difference
retrieve_PR = retrieve_polarization_ratio


#### PESCA


def _np_pesca_classification(Tc_23V, Tc_37V, Tc_89V, t2m, theta, sensor):  # noqa: PLR0911
    """
    Classify a single pixel based on PESCA algorithm.

    Parameters
    ----------
    - Tc_23V: Brightness temperature at 23V
    - Tc_37V: Brightness temperature at 37V
    - Tc_89V: Brightness temperature at 89V
    - t2m: 2-meter temperature
    - theta: viewing angle

    Returns
    -------
    - Snow class (int):
      0 = No snow
      1 = Deep Dry Snow
      2 = Polar Winter Snow
      3 = Perennial Snow
      4 = Thin Snow
    """
    # Define thresholds
    th1 = 280  # Threshold to launch the PESCA classification
    th2 = 1.01  # Threshold for RLF
    th3 = 257 - t2m
    if sensor == "GMI":
        th4 = (495 - t2m) / 250  # Threshold for Perennial Snow
        th5 = 5  # SI threshold for Thin Snow
    elif sensor == "ATMS":
        th4 = (465 - t2m) / 225  # Threshold for Perennial Snow
        th5 = 3 / np.cos(np.deg2rad(theta))  # SI threshold for Thin Snow
    else:
        return -1

    # Exclude pixels with T2m > 280 K
    if t2m > th1:
        return 0  # No snow

    # Compute derived metrics
    RLF = Tc_23V / Tc_37V  # ratio low frequency
    SI = Tc_23V - Tc_89V  # scattering index

    # Test 2: RLF > th2
    if th2 < RLF:
        # Test 3: Differentiate Deep Dry Snow and Polar Winter Snow
        if th3 < SI:
            return 1  # Deep Dry Snow
        return 2  # Polar Winter Snow

    # Test 4: Perennial Snow
    if Tc_23V / t2m < th4:
        return 3  # Perennial Snow

    # Test 5: Thin Snow
    if th5 < SI:
        return 4  # Thin Snow

    # Default: No snow
    return 0


def retrieve_PESCA(ds, t2m="t2m"):
    """Retrieve PESCA snow-classification."""
    # Retrieve sensor
    sensor = ds.attrs["InstrumentName"]

    # Retrieve viewing angle
    da_theta = get_xarray_variable(ds, variable="incidenceAngle")  # TODO retrieve viewing angle

    # Retrieve surface temperature
    da_t2m = get_xarray_variable(ds, variable=t2m)

    # Unstack Tb
    ds_tb = ds["Tc"].gpm.unstack_dimension(dim="pmw_frequency", suffix="_")

    # Retrieve Tb channels and incidence angle
    if sensor == "GMI":
        da_t23 = ds_tb["Tc_23V"]
        da_t37 = ds_tb["Tc_37V"]
        da_t89 = ds_tb["Tc_89V"]
    elif sensor == "ATMS":
        da_t23 = ds_tb["Tc_23.8QV"]
        da_t37 = ds_tb["Tc_31.4QV"]
        da_t89 = ds_tb["Tc_88.2QV"]
    else:
        raise NotImplementedError("PESCA not yet implemented for {sensor} sensor.")

    # Apply the function to the dataset using apply_ufunc
    kwargs = {"sensor": sensor}
    da_pesca = xr.apply_ufunc(
        _np_pesca_classification,
        da_t23,  # Tc_23V
        da_t37,  # Tc_37V
        da_t89,  # Tc_89V
        da_t2m,  # T2m
        da_theta,
        kwargs=kwargs,
        vectorize=True,  # Allow the function to apply element-wise
        dask="parallelized",  # Enable parallel computation with Dask
        output_dtypes=[np.float32],  # Specify output data type
    )

    # Set attributes for the classification output
    da_pesca.name = "PESCA"
    da_pesca.attrs["description"] = "Snow classification based on PESCA algorithm"
    dict_pesca_classes = {
        0: "No snow",
        1: "Deep Dry Snow",
        2: "Polar Winter Snow",
        3: "Perennial Snow",
        4: "Thin Snow",
    }
    da_pesca.attrs["flag_values"] = list(dict_pesca_classes)
    da_pesca.attrs["flag_meanings"] = list(dict_pesca_classes.values())
    return da_pesca


####----------------------------------------------------------------------------------------.
