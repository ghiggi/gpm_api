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
"""This module contains functions to sanitize GPM-API Dataset coordinates."""
import numpy as np
import xarray as xr


def ensure_valid_coords(ds, raise_error=False):
    """Ensure geographic coordinates are within expected range."""
    ds["lon"] = ds["lon"].compute()
    ds["lat"] = ds["lat"].compute()
    da_invalid_coords = np.logical_or(
        np.logical_or(ds["lon"] < -180, ds["lon"] > 180),
        np.logical_or(ds["lat"] < -90, ds["lat"] > 90),
    )
    if np.any(da_invalid_coords.data):
        if raise_error:
            raise ValueError("Invalid geographic coordinate in the granule.")

        # # Add valid_geolocation flag
        # ds = ds.assign_coords({"valid_geolocation": da_invalid_coords})

        # # For each variable, set NaN value where invalid coordinates
        # # --> Only if variable has at the 2 dimensions of ds["lon"]
        # # --> Not good practice because it might mask quality flags DataArrays !
        # for var in ds.data_vars:
        #     if set(ds["lon"].dims).issubset(set(ds[var].dims)):
        #         ds[var] = ds[var].where(~da_invalid_coords)

        # Add NaN to longitude and latitude
        ds["lon"] = ds["lon"].where(~da_invalid_coords)
        ds["lat"] = ds["lat"].where(~da_invalid_coords)
    return ds


def add_lh_height(ds):
    """Add 'height' coordinate to latent heat CSH and SLH products."""
    # Fixed heights for 2HSLH and 2HCSH
    # - FileSpec v7: p.2395, 2463
    # NOTE: In SLH/CSH, the first row of the array correspond to the surface
    # Instead, for the other GPM RADAR products, is the last row that correspond to the surface !!!
    height = np.linspace(0.25 / 2, 20 - 0.25 / 2, 80) * 1000  # in meters
    ds = ds.assign_coords({"height": ("range", height)})
    ds["height"].attrs["units"] = "m a.s.l"
    return ds


def add_cmb_height(ds):
    """Add the 'height' coordinate to CMB products."""
    from gpm.utils.manipulations import get_vertical_datarray_prototype

    if "ellipsoidBinOffset" in ds and "localZenithAngle" in ds and "range" in ds.dims:
        # # Retrieve required DataArrays
        range_bin = get_vertical_datarray_prototype(ds, fill_value=1) * ds["range"]  # start at 1 !
        ellipsoidBinOffset = ds["ellipsoidBinOffset"].isel(
            radar_frequency=0,
            missing_dims="ignore",
        )  # values are equal !
        localZenithAngle = ds["localZenithAngle"]
        rangeBinSize = 250  # approximation
        # Compute height
        n_bins = len(ds["range"])
        height = ((n_bins - range_bin) * rangeBinSize + ellipsoidBinOffset) * np.cos(np.deg2rad(localZenithAngle))
        height = height.drop_vars("radar_frequency", errors="ignore")
        ds = ds.assign_coords({"height": height})
    return ds


# def _add_cmb_range_coordinate(ds, scan_mode):
#     """Add range coordinate to 2B-<CMB> products."""
#     if "range" in list(ds.dims) and scan_mode in ["NS", "KuKaGMI", "KuGMI", "KuTMI"]:
#         range_values = np.arange(0, 88 * 250, step=250)
#         ds = ds.assign_coords({"range_interval": ("range", range_values)})
#         ds["range_interval"].attrs["units"] = "m"
#     return ds


# def _add_radar_range_coordinate(ds, scan_mode):
#     """Add range coordinate to 2A-<RADAR> products."""
#     # - V6 and V7: 1BKu 260 bins NS and MS, 130 at HS
#     if "range" in list(ds.dims):
#         if scan_mode in ["HS"]:
#             range_values = np.arange(0, 88 * 250, step=250)
#             ds = ds.assign_coords({"range_interval": ("range", range_values)})
#         if scan_mode in ["FS", "MS", "NS"]:
#             range_values = np.arange(0, 176 * 125, step=125)
#             ds = ds.assign_coords({"range_interval": ("range", range_values)})
#         ds["range_interval"].attrs["units"] = "m"
#     return ds


def _add_cmb_coordinates(ds, product, scan_mode):
    """Set coordinates of 2B-GPM-CORRA product."""
    if "pmw_frequency" in list(ds.dims):
        pmw_frequency = get_pmw_frequency_corra(product)
        ds = ds.assign_coords({"pmw_frequency": pmw_frequency})

    if (scan_mode in ("KuKaGMI", "NS")) and "radar_frequency" in list(ds.dims):
        ds = ds.assign_coords({"radar_frequency": ["Ku", "Ka"]})

    # Add height coordinate
    ds = add_cmb_height(ds)

    # # Add range_interval coordinate
    # ds = _add_cmb_range_coordinate(ds, scan_mode)
    return ds


def _add_wished_coordinates(ds):
    """Add wished coordinates to the dataset."""
    from gpm.dataset.groups_variables import WISHED_COORDS

    for var in WISHED_COORDS:
        if var in list(ds):
            ds = ds.set_coords(var)
    return ds


def _add_radar_coordinates(ds, product, scan_mode):  # noqa ARG001
    """Add range, height, radar_frequency, paramDSD coordinates to <RADAR> products."""
    if product == "2A-DPR" and "radar_frequency" in list(ds.dims):
        ds = ds.assign_coords({"radar_frequency": ["Ku", "Ka"]})
    if product in ["2A-DPR", "2A-Ku", "2A-Ka", "2A-PR"] and "paramDSD" in list(ds):
        ds = ds.assign_coords({"DSD_params": ["Nw", "Dm"]})
    # # Add range_interval coordinate
    # ds = _add_radar_range_coordinate(ds, scan_mode)
    return ds


def get_pmw_frequency_corra(product):
    from gpm.utils.pmw import get_pmw_frequency

    if product == "2B-GPM-CORRA":
        return get_pmw_frequency("GMI", scan_mode="S1") + get_pmw_frequency("GMI", scan_mode="S2")
    if product == "2B-TRMM-CORRA":
        return (
            get_pmw_frequency("TMI", scan_mode="S1")
            + get_pmw_frequency("TMI", scan_mode="S2")
            + get_pmw_frequency("TMI", scan_mode="S3")
        )
    raise ValueError("Invalid (CORRA) product {product}.")


def _parse_sun_local_time(ds):
    """Ensure sunLocalTime to be in float type."""
    dtype = ds["sunLocalTime"].data.dtype
    if dtype == "timedelta64[ns]":
        ds["sunLocalTime"] = ds["sunLocalTime"].astype(float) / 10**9 / 60 / 60
    elif np.issubdtype(dtype, np.floating):
        pass
    else:
        raise ValueError("Expecting sunLocalTime as float or timedelta64[ns]")
    ds["sunLocalTime"].attrs["units"] = "decimal hours"  # to avoid open_dataset netCDF convert to timedelta64[ns]
    return ds


def _add_1c_pmw_frequency(ds, product, scan_mode):
    """Add the 'pmw_frequency' coordinates to 1C-<PMW> products."""
    from gpm.utils.pmw import get_pmw_frequency

    if "pmw_frequency" in list(ds.dims):
        pmw_frequency = get_pmw_frequency(sensor=product.split("-")[1], scan_mode=scan_mode)
        ds = ds.assign_coords({"pmw_frequency": pmw_frequency})
    return ds


def _add_pmw_coordinates(ds, product, scan_mode):
    """Add coordinates to PMW products."""
    ds = _add_1c_pmw_frequency(ds, product, scan_mode)
    return ds


def _deal_with_pmw_incidence_angle_index(ds):
    if "incidenceAngleIndex" in ds:
        if "incidence_angle" in ds.dims and ds.sizes["incidence_angle"] > 1:
            idx_incidence_angle = ds["incidenceAngleIndex"].isel(along_track=0).astype(int).compute() - 1
        else:
            # Some 1C files have bad incidenceAngleIndex values ( )
            idx_incidence_angle = xr.zeros_like(ds["Tc"].isel(cross_track=0), dtype=int).compute()

        if "incidenceAngle" in ds:
            ds["incidenceAngle"] = ds["incidenceAngle"].isel(incidence_angle=idx_incidence_angle)
        if "sunGlintAngle" in ds:
            ds["sunGlintAngle"] = ds["sunGlintAngle"].isel(incidence_angle=idx_incidence_angle)
        ds = ds.drop_vars("incidenceAngleIndex")

    #     if "incidenceAngle" in ds:
    #         ds["incidenceAngle"] = ds["incidenceAngle"].isel(incidence_angle=idx_incidence_angle)
    #     if "sunGlintAngle" in ds:
    #         ds["sunGlintAngle"] = ds["sunGlintAngle"].isel(incidence_angle=idx_incidence_angle)
    # ds = ds.drop_vars("incidenceAngleIndex")
    return ds


def set_coordinates(ds, product, scan_mode):
    # Compute spatial coordinates in memory
    ds["lon"] = ds["lon"].compute()
    ds["lat"] = ds["lat"].compute()

    # ORBIT objects
    if "cross_track" in list(ds.dims):
        # Ensure valid coordinates
        ds = ensure_valid_coords(ds, raise_error=False)

    # Add range and gpm_range_id coordinates
    # - range starts at 1 (for value-based selection with bin variables)
    # - gpm_range_id starts at 0 (following python based indexing)
    if "range" in list(ds.dims):
        range_size = ds.sizes["range"]
        range_coords = {
            "range": ("range", np.arange(1, range_size + 1)),
            "gpm_range_id": ("range", np.arange(0, range_size)),
        }
        ds = ds.assign_coords(range_coords)

    # Add wished coordinates
    ds = _add_wished_coordinates(ds)

    # Convert sunLocalTime to float
    if "sunLocalTime" in ds:
        ds = _parse_sun_local_time(ds)
        ds = ds.set_coords("sunLocalTime")

    #### PMW
    # - 1B and 1C products
    if product.startswith("1C") or product.startswith("1B"):
        ds = _add_pmw_coordinates(ds, product, scan_mode)
    # - Deal with incidenceAngleIndex in PMW 1C products
    if product.startswith("1C"):
        ds = _deal_with_pmw_incidence_angle_index(ds)
    #### RADAR
    if product in ["2A-DPR", "2A-Ku", "2A-Ka", "2A-PR", "2A-ENV-DPR", "2A-ENV-PR", "2A-ENV-Ka", "2A-ENV-Ku"]:
        ds = _add_radar_coordinates(ds, product, scan_mode)

    #### CMB
    if product in ["2B-GPM-CORRA", "2B-TRMM-CORRA"]:
        ds = _add_cmb_coordinates(ds, product, scan_mode)

    #### SLH and CSH products
    if product in ["2A-GPM-SLH", "2B-GPM-CSH"] and "range" in list(ds.dims):
        ds = add_lh_height(ds)

    #### IMERG
    if "HQobservationTime" in ds:
        da = ds["HQobservationTime"]
        da = da.where(da >= 0)  # -9999.9 --> NaN
        da.attrs["units"] = "minutes from the start of the current half hour"
        da.encoding["dtype"] = "uint8"
        da.encoding["_FillValue"] = 255
        da.attrs.pop("_FillValue", None)
        da.attrs.pop("CodeMissingValue", None)
        ds["HQobservationTime"] = da

    if "MWobservationTime" in ds:
        da = ds["MWobservationTime"]
        da = da.where(da >= 0)  # -9999.9 --> NaN
        da.attrs["units"] = "minutes from the start of the current half hour"
        da.attrs.pop("_FillValue", None)
        da.attrs.pop("CodeMissingValue", None)
        da.encoding["dtype"] = "uint8"
        da.encoding["_FillValue"] = 255
        ds["MWobservationTime"] = da
    return ds
