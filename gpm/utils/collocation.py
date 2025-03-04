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
"""This module contains utilities for GPM product collocation."""
import datetime

import numpy as np
import pyproj
import xarray as xr

import gpm
from gpm.utils.manipulations import get_spatial_2d_datarray_template


def _get_collocation_defaults_args(product, variables, groups, version, scan_modes):
    """Get collocation defaults arguments."""
    if scan_modes is None:
        scan_modes = gpm.available_scan_modes(product=product, version=version)
    if isinstance(scan_modes, str):
        scan_modes = [scan_modes]
    if product not in gpm.available_products(product_categories="PMW") and len(scan_modes) > 1:
        raise ValueError("Multiple scan modes can be specified only for PMW products!")
    # PMW defaults
    if variables is None and groups is None:
        if product in gpm.available_products(product_levels="2A", product_categories="PMW"):
            variables = [
                "surfacePrecipitation",
                "mostLikelyPrecipitation",
                "cloudWaterPath",  # kg/m2
                "rainWaterPath",  # kg/m2
                "iceWaterPath",  # kg/m2
            ]
        elif product.startswith("1C"):
            variables = ["Tc"]
        elif product.startswith("1B"):
            variables = ["Tb"]
        else:
            pass
    return scan_modes, variables, groups


def collocate_product(
    ds,
    product,
    product_type="RS",
    version=None,
    storage="GES_DISC",
    scan_modes=None,
    variables=None,
    groups=None,
    verbose=True,
    decode_cf=True,
    chunks={},
):
    """Collocate a product on the provided dataset.

    It assumes that along all the input dataset, there is an approximate collocated product.
    """
    # Get default collocation arguments
    scan_modes, variables, groups = _get_collocation_defaults_args(
        product=product,
        variables=variables,
        groups=groups,
        version=version,
        scan_modes=scan_modes,
    )

    # Define start_time, end_time around input dataset
    start_time = ds.gpm.start_time - datetime.timedelta(minutes=5)
    end_time = ds.gpm.end_time + datetime.timedelta(minutes=5)

    # Download PMW products (if necessary)
    gpm.download(
        product=product,
        product_type=product_type,
        start_time=start_time,
        end_time=end_time,
        version=version,
        storage=storage,
        force_download=False,
        verbose=verbose,
    )

    # Read datasets
    dt = gpm.open_datatree(
        product=product,
        start_time=start_time,
        end_time=end_time,
        # Optional
        version=version,
        variables=variables,
        groups=groups,
        product_type=product_type,
        scan_modes=scan_modes,
        chunks=chunks,
        decode_cf=decode_cf,
    )

    # Remap datasets
    list_remapped = [dt[scan_mode].to_dataset().gpm.remap_on(ds) for scan_mode in scan_modes]

    # Concatenate if necessary (PMW case)
    output_ds = xr.concat(list_remapped, dim="pmw_frequency") if len(list_remapped) > 1 else list_remapped[0]

    # Add time of dst dataset
    output_ds = output_ds.assign_coords({"time": ds.reset_coords()["time"]})

    # Assign attributes
    output_ds.attrs = dt[scan_modes[0]].attrs
    output_ds.attrs["ScanMode"] = scan_modes

    return output_ds


# def _expand_with_scan_mode(ds, scan_mode):
#     # Squeeze dataset
#     if "incidence_angle" in ds.dims:
#         ds = ds.squeeze(dim="incidence_angle")

#     # Define variables to exclude from expansion (always keep as is)
#     variables_to_exclude = [
#         "lon",
#         "lat",
#         "longitude",
#         "latitude",
#         "crsWGS84",
#         "gpm_id",
#         "gpm_time",
#         "gpm_granule_id",
#         "gpm_along_track_id",
#         "gpm_cross_track_id",
#     ]

#     # Define variables to preprocess
#     variables = set(ds.variables) - set(variables_to_exclude)
#     coords_to_expand = [
#         var
#         for var in variables
#         if (
#             var in ds.coords
#             and "pmw_frequency" not in ds[var].dims
#             and np.all(np.isin(("along_track", "cross_track"), ds[var].dims))
#         )
#     ]

#     variables_to_expand = [
#         var
#         for var in variables
#         if (
#             var not in ds.coords
#             and "pmw_frequency" not in ds[var].dims
#             and np.all(np.isin(("along_track", "cross_track"), ds[var].dims))
#         )
#     ]
#     variables_not_to_expand = [var for var in variables if "pmw_frequency" in ds[var].dims]

#     # If nothing to expand, return input dataset
#     if not variables_to_expand and not coords_to_expand:
#         return ds

#     # Expand dataset
#     if variables_to_expand:
#         ds_expanded = ds[variables_to_expand]
#         if coords_to_expand:
#             ds_expanded = ds_expanded.reset_coords(coords_to_expand)
#     else:  # only coords to expand
#         ds_expanded = ds.reset_coords(coords_to_expand)[coords_to_expand]
#     ds_expanded = ds_expanded.expand_dims(dim={"scan_mode": 1}, axis=-1)
#     ds_expanded = ds_expanded.assign_coords({"scan_mode": [scan_mode]})

#     # Add variables to not be expanded
#     ds_expanded.update(ds.reset_coords()[variables_not_to_expand])
#     return ds_expanded


# def _remap_pmw_datatree(dt, scan_modes, scan_mode_reference, radius_of_influence=20_000):
#     # Define grid template
#     ds_template = dt[scan_mode_reference].to_dataset()

#     # Remap each scan_mode dataset onto the template grid
#     list_remapped = [
#         dt[scan_mode].to_dataset().gpm.remap_on(ds_template, radius_of_influence=radius_of_influence)
#         for scan_mode in scan_modes
#     ]

#     # Insert the template dataset as the first element in the list
#     list_remapped.insert(0, ds_template)

#     # Concatenate common variables along pmw_frequency or scan_mode
#     vars_pmw_freq = [var for var in list_remapped[1].data_vars if "pmw_frequency" in ds_template[var].dims]
#     vars_scan_mode = [var for var in list_remapped[1].data_vars if "scan_mode" in ds_template[var].dims]
#     ds_pmw = xr.concat([ds[vars_pmw_freq] for ds in list_remapped], dim="pmw_frequency")
#     ds_scan_mode = xr.concat([ds[vars_scan_mode] for ds in list_remapped], dim="scan_mode")
#     output_ds = xr.merge([ds_pmw, ds_scan_mode])

#     # Add back some variables
#     # - TODO: SClatitude, SClongitude, SCaltitude, FractionalGranuleNumber
#     # - TODO: incidenceAngleIndex
#     return output_ds


def _remap_pmw_datatree(dt_expanded, scan_modes, scan_mode_reference, radius_of_influence=20_000):
    # Define grid template
    ds_template = dt_expanded[scan_mode_reference].to_dataset()

    # Remap each scan_mode dataset onto the template grid
    list_remapped = [ds_template]
    for scan_mode in scan_modes:
        ds = dt_expanded[scan_mode].to_dataset().gpm.remap_on(ds_template, radius_of_influence=radius_of_influence)
        list_remapped.append(ds)

    # Concatenate common variables along pmw_frequency or scan_mode, then add unique variables
    list_pmw_freq = [ds[[var for var in ds.data_vars if "pmw_frequency" in ds[var].dims]] for ds in list_remapped]
    list_scan_mode = [ds[[var for var in ds.data_vars if "scan_mode" in ds[var].dims]] for ds in list_remapped]
    list_unique = [
        ds[[var for var in ds.data_vars if not np.isin(["scan_mode", "pmw_frequency"], ds[var].dims).any()]]
        for ds in list_remapped
    ]
    ds_unique = xr.merge(list_unique)
    ds_pmw = xr.concat(list_pmw_freq, dim="pmw_frequency", coords="minimal", compat="override")
    ds_scan_mode = xr.concat(list_scan_mode, dim="scan_mode", coords="minimal", compat="override")
    ds = xr.merge([ds_pmw, ds_scan_mode, ds_unique], compat="override")

    # Add back missing variables of dt[scan_mode_reference]
    # TODO: SClatitude, SClongitude, SCaltitude, FractionalGranuleNumber

    return ds


def preprocess_datatree(dt, exclude_vars=None, fixed_vars=None):
    """Preprocess DataTree for remapping by handling variables consistently.

    Parameters
    ----------
    dt : xarray.DataTree
        DataTree with multiple scan modes.
    exclude_vars : list, optional
        Variables to exclude from processing.
    fixed_vars : list, optional
        Variables to preserve as-is.

    Returns
    -------
    xarray.DataTree
        Preprocessed DataTree ready for remapping.
    """
    # Prepare datasets for remapping
    # - Remove 'exclude_vars' variables from all datatree nodes
    #   --> SClatitude, SClongitude, SCaltitude, FractionalGranuleNumbe
    #
    # - Variables with only (along_track) dimension should be broadcasted to (along_track, cross_track)
    #   unless being 'fixed_vars'
    #   --> Quality and sunLocalTime
    #      --> If user broadcast such variables to have the pmw_frequency dimension
    #          -->  The final dataset will be concatenate along pmw_frequency dimension
    #      --> Otherwise such variables will be concatenated along the scan_mode dimension
    # - Remove all variables without (along_track, cross-track) dimensions unless being in 'fixed_vars'

    # List scan modes
    scan_modes = list(dt)

    # Move coordinates to be remapped as variables
    coords_variables = ["sunLocalTime", "Quality"]
    for scan_mode in scan_modes:
        ds = dt[scan_mode].to_dataset()
        dt[scan_mode] = ds.reset_coords([var for var in coords_variables if var in list(ds.coords)], drop=False)

    # Set default values
    if exclude_vars is None:
        exclude_vars = [
            "SClatitude",
            "SClongitude",
            "SCaltitude",
            "FractionalGranuleNumber",
        ]
    if fixed_vars is None:
        fixed_vars = [
            "lon",
            "lat",
            "longitude",
            "latitude",
            "crsWGS84",
            "gpm_id",
            "gpm_time",
            "gpm_granule_id",
            "gpm_along_track_id",
            "gpm_cross_track_id",
        ]

    # - Remove 'exclude_vars' variables from all datatree nodes
    dt = dt.map_over_datasets(lambda ds: ds.drop_vars(exclude_vars, errors="ignore"))

    # - Variables with only (along_track) or (cross-track) dimension should be broadcasted
    #   to (along_track, cross_track) unless being 'fixed_vars'
    def _broadcast_along_track_only(ds, fixed_vars):
        ds_spatial_2d_template = get_spatial_2d_datarray_template(ds)
        variables = [var for var in ds.data_vars if var not in fixed_vars]
        for var in variables:
            if len(ds[var].gpm.spatial_dimensions) == 1:
                ds[var] = ds[var].broadcast_like(ds_spatial_2d_template)
        return ds

    for scan_mode in scan_modes:
        dt[scan_mode] = _broadcast_along_track_only(dt[scan_mode].to_dataset(), fixed_vars=fixed_vars)

    # -------------------------------------------------------------------------
    # Preprocess variables with (along_track, cross-track) dimensions
    # - Unique variables:
    #   - Variables with (along_track, cross-track) dimensions not shared across any datatree dataset
    #     are kept as they are
    #   - Can be just added to the final dataset
    # - Partial variables:
    #    - Variables with (along_track, cross-track) dimensions present in some but not all datatree dataset
    #    - NaN DataArray variable should be added to dataset where missing --> Then treated as shared variables
    # - Shared variables with pmw_frequency:
    #   - Variables with (along_track, cross-track, pmw_frequency) dimensions shared across all datatree dataset
    #     are kept as they are
    #   - Such variables are concatenated along pmw_frequency in the final dataset
    # - Shared variables without pmw_frequency:
    #    - Variables with (along_track, cross-track) dimensions shared across all datatree dataset
    #      are expand with the scan_mode dimension.
    #  - Such variables are concatenated along scan_mode in the final dataset

    # List variables across datatree
    dict_vars = {scan_mode: list(dt[scan_mode].data_vars) for scan_mode in scan_modes}

    # Find list of all variables
    all_variables = [var for vars_list in dict_vars.values() for var in vars_list]

    # Find shared variables (present in every scan_mode)
    shared_variables = set(dict_vars[scan_modes[0]])
    for scan_mode in scan_modes[1:]:
        shared_variables &= set(dict_vars[scan_mode])

    # Find unique variables across scan modes
    unique_variables = {scan_mode: set(dict_vars[scan_mode]) - shared_variables for scan_mode in scan_modes}
    unique_variables = set().union(*unique_variables.values())

    # Find partial variables (occur only in some scan modes)
    partial_variables = set(all_variables) - unique_variables - shared_variables  # noqa F841

    # Add dummy NaN array in scan_modes where partial_variables missing
    # TODO: TODO

    # Expand shared variables without pmw_frequency dimension
    for scan_mode in scan_modes:
        ds = dt[scan_mode].to_dataset()
        for var in ds.data_vars:
            if var in shared_variables and "pmw_frequency" not in ds[var].dims:
                if "scan_mode" not in ds:
                    ds = ds.assign_coords({"scan_mode": scan_mode})
                ds[var] = ds[var].expand_dims(dim={"scan_mode": 1}, axis=-1)
        dt[scan_mode] = ds
    return dt


def _define_time_blocks(dt, scan_mode_reference, window_duration=3600, overlap_duration=60):
    """
    Define overlapping time periods.

    Parameters
    ----------
        dt (dict): Dictionary of xarray datasets by scan mode.
        scan_mode_reference (str): Reference scan mode key.
        window_duration (int): Duration of the time window in seconds (default: 1 hour).
        overlap_duration (int): Duration of the overlap in seconds (default: 1 minute).

    Returns
    -------
        list: List of tuples containing (start_time, end_time, gpm_id_start, gpm_id_stop).
    """
    # Retrieve reference dataset
    ds = dt[scan_mode_reference]

    # Extract the reference time dimension
    start_time = np.datetime64(ds["time"].gpm.start_time)
    end_time = np.datetime64(ds["time"].gpm.end_time)

    # Calculate the time window step with overlap
    step_duration = window_duration - overlap_duration

    # Generate reference time blocks
    start_times = np.arange(start_time, end_time, np.timedelta64(step_duration, "s"))
    end_times = start_times + np.timedelta64(step_duration, "s")
    end_times = np.minimum(end_times, end_time)

    # Now define window time blocks and gpm_id at each time block
    time_blocks = [
        [
            np.maximum(start_time - np.timedelta64(window_duration, "s"), start_times[0]),
            np.minimum(end_time + np.timedelta64(window_duration, "s"), end_times[-1]),
            ds["gpm_id"].gpm.sel(time=start_time, method="nearest").to_numpy().item(),
            ds["gpm_id"].gpm.sel(time=end_time, method="nearest").to_numpy().item(),
        ]
        for start_time, end_time in zip(start_times, end_times, strict=False)
    ]

    # Count number of blocks
    n_blocks = len(time_blocks)

    # Enforce first and last gpm_id
    gpm_ids = ds["gpm_id"].data
    time_blocks[0][2] = gpm_ids[0].item()
    time_blocks[-1][3] = gpm_ids[-1].item()

    # If only 1 block, return it
    if n_blocks == 1:
        return time_blocks

    # Otherwise, ensure no missing gpm_id between blocks and no repeated gpm_id
    for i in range(1, n_blocks):
        next_gpm_id_start = time_blocks[i][2]
        previous_gpm_id_stop = time_blocks[i - 1][3]
        # Retrieve gpm_id position
        next_idx = np.where(gpm_ids == next_gpm_id_start)[0]
        previous_idx = np.where(gpm_ids == previous_gpm_id_stop)[0]
        # If same gpm_id, modify next_gpm_id_start with the next gpm_id value
        if next_idx == previous_idx:
            time_blocks[i][2] = gpm_ids[previous_idx + 1].item()
        if next_idx > previous_idx + 1:
            time_blocks[i][2] = gpm_ids[previous_idx + 1].item()

    return time_blocks


def regrid_pmw_l1(dt, scan_mode_reference="S1", radius_of_influence=20_000):
    """
    Regrid the scan modes of a PMW Level 1 product into a common grid.

    Parameters
    ----------
    dt : xarray.DataTree
        DataTree containing multiple scan modes (nodes).
    scan_mode_reference : str, optional
        The scan mode/node with the spatial coordinates to use as reference grid.

    Returns
    -------
    xarray.Dataset
        The collocated dataset, with PMW channels concatenated along a 'pmw_frequency' dimension.
    """
    # Retrieve available scan modes
    scan_modes = list(dt)

    # Check template in datatree
    if scan_mode_reference not in dt:
        raise ValueError(f"The 'scan_mode_reference' '{scan_mode_reference}' is not found in the provided DataTree.")

    # Remove template scan mode from scan_modes
    scan_modes.remove(scan_mode_reference)

    # Ensure at least one scan mode to collocate
    if len(scan_modes) == 0:
        return dt[scan_mode_reference].to_dataset()

    # Retrieve datatree attributes
    attrs = dt[scan_mode_reference].attrs.copy()
    if attrs.get("gpm_api_product") not in gpm.available_products(
        product_categories="PMW",
        product_levels=["1B", "1C"],
    ):
        raise ValueError("The DataTree does not contain a 1B or 1C PMW product.")

    # - TODO: Variables without (along_track, cross-track) dimensions are currently not remapped !
    # Prepare DataTree to remap
    # - Expand the required variables
    # - Infill missing variables
    # dict_scan_modes = {
    #     scan_mode: _expand_with_scan_mode(dt[scan_mode].to_dataset(), scan_mode=scan_mode) for scan_mode in list(dt)
    # }
    # dt_expanded = xr.DataTree.from_dict(dict_scan_modes)

    dt_expanded = preprocess_datatree(dt, exclude_vars=None, fixed_vars=None)

    # If GPM product, remap by blocks to avoid orbit intersections
    if "gpm_id" in dt[scan_mode_reference]:
        # Define time blocks over which to remap
        time_blocks = _define_time_blocks(
            dt_expanded,
            scan_mode_reference=scan_mode_reference,
            window_duration=60 * 30,
            overlap_duration=120,
        )

        # Loop and remap over block of time (to avoid orbit intersection)
        list_ds = []
        # start_time, end_time, gpm_id_start, gpm_id_stop = time_blocks[0]
        for start_time, end_time, gpm_id_start, gpm_id_stop in time_blocks:
            dict_subset = {
                scan_mode: dt_expanded[scan_mode].to_dataset().gpm.sel(time=slice(start_time, end_time))
                for scan_mode in list(dt_expanded)
            }
            dt_subset = xr.DataTree.from_dict(dict_subset)
            output_ds = _remap_pmw_datatree(
                dt_subset,
                scan_mode_reference=scan_mode_reference,
                scan_modes=scan_modes,
                radius_of_influence=radius_of_influence,
            )
            output_ds = output_ds.gpm.sel(gpm_id=slice(gpm_id_start, gpm_id_stop))
            list_ds.append(output_ds)

        # Concatenate dataset
        ds = xr.concat(list_ds, dim="along_track")
    # If i.e. TCPRIMED product, remap full datatree directly
    else:
        ds = _remap_pmw_datatree(
            dt_expanded,
            scan_mode_reference=scan_mode_reference,
            scan_modes=scan_modes,
            radius_of_influence=radius_of_influence,
        )

    # Assign attributes
    ds.attrs = attrs
    ds.attrs["ScanModes"] = sorted([*scan_modes, scan_mode_reference])
    return ds


def remap_era5(ds, variables):
    """Remap ERA5 variables onto the input dataset using nearest neighbour."""
    from gpm.dataset.crs import set_dataset_crs

    # Open ERA5 archive on Google Cloud Bucket
    # - Currently available from 1940 to May 2023
    # - x axis longitudes goes from 0 to 360 (pm=0)
    # - y axis is decreasing !
    ds_era5 = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        chunks=None,
        storage_options={"token": "anon"},
    )

    # ds_era5["2m_temperature"].sel(time="2000-01-01 12:00:00").gpm.plot_map()

    # Check variables
    if isinstance(variables, str):
        variables = [variables]
    available_variables = list(ds_era5.data_vars)
    invalid_variables = [var for var in variables if var not in available_variables]
    if len(invalid_variables) > 1:
        raise ValueError(
            f"{invalid_variables} are invalid ERA5 variables. Available variables are {available_variables}.",
        )

    # Define time window over which to retrieve ERA5 data
    start_time = ds.gpm.start_time - datetime.timedelta(minutes=60)
    end_time = ds.gpm.end_time + datetime.timedelta(minutes=60)

    # Retrieve ERA5 data globally at given time period
    ds_era5 = ds_era5[variables].sel(time=slice(start_time, end_time))
    ds_era5 = ds_era5.compute()

    # Set CRS to dataset
    crs_wgs84 = pyproj.CRS(proj="longlat", ellps="WGS84")
    ds_era5 = set_dataset_crs(ds_era5, crs=crs_wgs84, grid_mapping_name="spatial_ref", inplace=False)

    # Map data to swath
    ds_env = ds_era5.gpm.remap_on(ds)

    # Select data closest to sensor observation
    ds_env["delta_time"] = ds_env["time"] - ds["time"]
    idx_time = np.abs(ds_env["delta_time"]).argmin(dim="time")
    ds_env = ds_env.isel(time=idx_time)
    ds_env = ds_env.assign_coords({"time": (*ds["time"].dims, ds["time"].data)})
    return ds_env
