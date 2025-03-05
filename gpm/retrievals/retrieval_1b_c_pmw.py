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
import pandas as pd
import xarray as xr

from gpm.checks import check_is_spatial_2d
from gpm.utils.decorators import check_software_availability
from gpm.utils.pmw import (
    PMWFrequency,
    create_rgb_composite,
    get_available_pct_features,
    get_available_pd_features,
    get_available_pr_features,
    get_pct,
    get_pd,
    get_pmw_rgb_receipts,
    get_pr,
)
from gpm.utils.xarray import (
    get_default_variable,
    get_xarray_variable,
)


def retrieve_rgb_composites(ds):
    """Retrieve the available PMW RGB composites."""
    # Retrieve sensor
    if "InstrumentName" in ds.attrs:
        sensor = ds.attrs["InstrumentName"]
    elif "instrument" in ds.attrs:  # compatibility with TC PRIMED
        sensor = ds.attrs["instrument"]
    else:
        raise ValueError("Impossible to determine instrument name.")

    # Retrieve defined RGB receipts
    receipts = get_pmw_rgb_receipts(sensor)

    # Compute RGB composites
    list_ds_rgb = []
    for name, receipt in receipts.items():
        try:
            da_rgb = create_rgb_composite(ds, receipt=receipt).to_dataset(name=name)
            list_ds_rgb.append(da_rgb)
        except Exception as e:
            print(f"RGB Composite {name} not available: {e!s}")

    ds_rgb = xr.merge(
        list_ds_rgb,
        compat="override",
    )  # deal with incompatible coords across scanModes (i.e. sunLocalTime)
    return ds_rgb


def retrieve_polarization_corrected_temperature(xr_obj, variable=None):
    """Retrieve PMW Polarization-Corrected Temperature (PCT).

    Coefficients are taken from Cecil et al., 2018.

    References
    ----------
    Cecil, D. J., and T. Chronis, 2018.
    Polarization-Corrected Temperatures for 10-, 19-, 37-, and 89-GHz Passive Microwave Frequencies.
    J. Appl. Meteor. Climatol., 57, 2249-2265, https://doi.org/10.1175/JAMC-D-18-0022.1.
    """
    # Retrieve DataArray with brightness temperatures
    if variable is None and isinstance(xr_obj, xr.Dataset):
        variable = get_default_variable(xr_obj, possible_variables=["Tb", "Tc"])
    da = get_xarray_variable(xr_obj, variable=variable)

    # If no combo, raise error
    pct_features = get_available_pct_features(da)
    if len(pct_features) == 0:
        pmw_frequencies = [PMWFrequency.from_string(freq) for freq in da["pmw_frequency"].data]
        pmw_frequencies = [freq.title() for freq in pmw_frequencies]
        raise ValueError(f"Impossible to compute polarized corrected temperature with channels: {pmw_frequencies}.")

    # Create PCTs dataset
    dict_pct = {name: get_pct(da, name=name).rename(name) for name in pct_features}
    ds_pct = xr.merge(dict_pct.values(), compat="minimal")
    return ds_pct


def retrieve_polarization_difference(xr_obj, variable=None):
    """Retrieve PMW Channels Polarized Difference (PD)."""
    # Retrieve DataArray with brightness temperatures
    if variable is None and isinstance(xr_obj, xr.Dataset):
        variable = get_default_variable(xr_obj, possible_variables=["Tb", "Tc"])
    da = get_xarray_variable(xr_obj, variable=variable)

    # If no combo, raise error
    pd_features = get_available_pd_features(da)
    if len(pd_features) == 0:
        pmw_frequencies = [PMWFrequency.from_string(freq) for freq in da["pmw_frequency"].data]
        pmw_frequencies = [freq.title() for freq in pmw_frequencies]
        raise ValueError(f"Impossible to compute polarized difference with channels: {pmw_frequencies}. No pairs.")

    # Create PDs dataset
    dict_pd = {name: get_pd(da, name=name).rename(name) for name in pd_features}
    ds_pd = xr.merge(dict_pd.values(), compat="minimal")
    return ds_pd


def retrieve_polarization_ratio(xr_obj, variable=None):
    """Retrieve PMW Channels Polarization Ratio (PR)."""
    # Retrieve DataArray with brightness temperatures
    if variable is None and isinstance(xr_obj, xr.Dataset):
        variable = get_default_variable(xr_obj, possible_variables=["Tb", "Tc"])
    da = get_xarray_variable(xr_obj, variable=variable)

    # Retrieve polarized frequencies couples
    pr_features = get_available_pr_features(da)

    # If no combo, raise error
    if len(pr_features) == 0:
        pmw_frequencies = [PMWFrequency.from_string(freq) for freq in da["pmw_frequency"].data]
        pmw_frequencies = [freq.title() for freq in pmw_frequencies]
        raise ValueError(f"Impossible to compute polarization ratio with channels: {pmw_frequencies}. No pairs.")

    # Create PDs dataset
    dict_pr = {name: get_pr(da, name=name).rename(name) for name in pr_features}
    ds_pr = xr.merge(dict_pr.values(), compat="minimal")
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


@check_software_availability(software="sklearn", conda_package="scikit-learn")
@check_software_availability(software="umap", conda_package="umap-learn")
def retrieve_UMAP_RGB(ds, scaler=None, n_neighbors=10, min_dist=0.01, random_state=None, **kwargs):
    """Create a UMAP RGB composite."""
    import umap
    from sklearn.preprocessing import MinMaxScaler

    # Check dataset has only spatial 2D variables
    check_is_spatial_2d(ds)

    # Define variables
    variables = list(ds.data_vars)

    # Convert to dataframe
    df = ds.gpm.to_pandas_dataframe(drop_index=False)

    # Retrieve dataset coordinates which are present in the dataframe
    coordinates = [column for column in list(ds.coords) if column in df]

    # Remove rows with non finite values
    df_valid = df[np.isfinite(df[variables]).all(axis=1) & (~np.isnan(df[variables])).all(axis=1)]

    # Retrieve dataframe with only variables
    df_data = df_valid[variables]

    # Define scaler
    if scaler is not None:
        scaler.fit(df_data)
        scaled_data = scaler.transform(df_data)
    else:
        scaled_data = df_data

    # Compute 3D embedding
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=3, random_state=random_state, **kwargs)
    embedding = reducer.fit_transform(scaled_data)

    # Define RGB scaler
    rgb_scaler = MinMaxScaler()
    rgb_scaler = rgb_scaler.fit(embedding)

    # Scale UMAP embedding between 0 and 1
    rgb_data = rgb_scaler.transform(embedding)
    rgb_data = np.clip(rgb_data, a_min=0, a_max=1)

    # Create RGB dataframe of valid pixels
    df_rgb_valid = pd.DataFrame(rgb_data, index=df_data.index, columns=["R", "G", "B"])

    # Create original RGB dataframe
    df_rgb = df[coordinates]
    df_rgb = df_rgb.merge(df_rgb_valid, how="outer", left_index=True, right_index=True)

    # Convert back to xarray
    ds_rgb = df_rgb.to_xarray()
    ds_rgb = ds_rgb.set_coords(coordinates)

    # Define RGB DataArray
    da_rgb = ds_rgb[["R", "G", "B"]].to_array(dim="rgb")

    # Add missing coordinates
    missing_coords = {coord: ds[coord] for coord in set(ds.coords) - set(da_rgb.coords)}
    da_rgb = da_rgb.assign_coords(missing_coords)

    # Return RGB DataArray
    return da_rgb


@check_software_availability(software="sklearn", conda_package="scikit-learn")
def retrieve_PCA_RGB(ds, scaler=None):
    """Create a PCA RGB composite."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    # Check dataset has only spatial 2D variables
    check_is_spatial_2d(ds)

    # Define variables
    variables = list(ds.data_vars)

    # Convert to dataframe
    df = ds.gpm.to_pandas_dataframe(drop_index=False)

    # Retrieve dataset coordinates which are present in the dataframe
    coordinates = [column for column in list(ds.coords) if column in df]

    # Remove rows with non finite values
    df_valid = df[np.isfinite(df[variables]).all(axis=1) & (~np.isnan(df[variables])).all(axis=1)]

    # Retrieve dataframe with only variables
    df_data = df_valid[variables]

    # Define scaler
    if scaler is not None:
        scaler.fit(df_data)
        scaled_data = scaler.transform(df_data)
    else:
        scaled_data = df_data

    # Compute 3D embedding
    pca = PCA(n_components=3)
    pca.fit(scaled_data)
    embedding = pca.transform(scaled_data)

    # Define RGB scaler
    rgb_scaler = MinMaxScaler()
    rgb_scaler = rgb_scaler.fit(embedding)

    # Scale UMAP embedding between 0 and 1
    rgb_data = rgb_scaler.transform(embedding)
    rgb_data = np.clip(rgb_data, a_min=0, a_max=1)

    # Create RGB dataframe of valid pixels
    df_rgb_valid = pd.DataFrame(rgb_data, index=df_data.index, columns=["R", "G", "B"])

    # Create original RGB dataframe
    df_rgb = df[coordinates]
    df_rgb = df_rgb.merge(df_rgb_valid, how="outer", left_index=True, right_index=True)

    # Convert back to xarray
    ds_rgb = df_rgb.to_xarray()
    ds_rgb = ds_rgb.set_coords(coordinates)

    # Define RGB DataArray
    da_rgb = ds_rgb[["R", "G", "B"]].to_array(dim="rgb")

    # Add missing coordinates
    missing_coords = {coord: ds[coord] for coord in set(ds.coords) - set(da_rgb.coords)}
    da_rgb = da_rgb.assign_coords(missing_coords)

    # Return RGB DataArray
    return da_rgb


####----------------------------------------------------------------------------------------.
