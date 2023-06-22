#!/usr/bin/env python3
"""
Created on Thu Jun 22 15:01:43 2023

@author: ghiggi
"""
import re

DIM_DICT = {
    "nscan": "along_track",
    "nray": "cross_track",
    "npixel": "cross_track",
    "nrayMS": "cross_track",
    "nrayHS": "cross_track",
    "nrayNS": "cross_track",
    "nrayFS": "cross_track",
    "nbin": "range",
    "nbinMS": "range",
    "nbinHS": "range",
    "nbinFS": "range",
    "nfreq": "frequency",
    # PMW 1B-TMI
    "npixelev1": "cross_track",
    "npixelev2": "cross_track",
    # PMW 1B-GMI (V7)
    "npix1": "cross_track",  # PMW (i.e. GMI)
    "npix2": "cross_track",  # PMW (i.e. GMI)
    "nfreq1": "frequency",
    "nfreq2": "frequency",
    "nchan1": "channel",
    "nchan2": "channel",
    # PMW 1C-GMI (V7)
    "npixel1": "cross_track",
    "npixel2": "cross_track",
    # PMW 1A-GMI and 1C-GMI (V7)
    "nscan1": "along_track",
    "nscan2": "along_track",
    "nchannel1": "channel",
    "nchannel2": "channel",
    # PMW 1A-GMI (V7)
    "npixelev": "cross_track",
    "npixelht": "cross_track",
    "npixelcs": "cross_track",
    "npixelfr": "cross_track",  # S4 mode
    # 2A-DPR, 2A-Ku, 2A-Ka
    "nDSD": "DSD_params",
    # nBnPSD --> "range" in CORRA --> 88, 250 M interval
    # "nfreqHI":
    # nlayer --> in CSH, SLH --> converted to height in decoding
}


def _decode_dimensions(ds):
    """Decode dimensions."""
    list(ds.dims)
    dataset_vars = list(ds.data_vars)
    rename_dim_dict = {}
    for var in dataset_vars:
        da = ds[var]
        dim_names_str = da.attrs.get("DimensionNames", None)
        if dim_names_str is not None:
            dim_names = dim_names_str.split(",")
            for dim, new_dim in zip(list(da.dims), dim_names):
                if dim not in rename_dim_dict:
                    rename_dim_dict[dim] = new_dim
                else:
                    if rename_dim_dict[dim] == new_dim:
                        pass
                    else:  # when more variable share same dimension length (and same phony_dim_<n>)
                        ds[var] = ds[var].rename({dim: new_dim})
    if len(rename_dim_dict) > 0:
        ds = ds.rename_dims(rename_dim_dict)
    return ds


def assign_dataset_dimensions(ds):
    """Assign standard dimension names to xarray Dataset."""
    dataset_dims = list(ds.dims)

    # Do not assign dimension name if already exists (i.e. IMERG)
    if not re.match("phony_dim", dataset_dims[0]):
        return ds

    # Get dimension name from DimensionNames attribute
    ds = _decode_dimensions(ds)

    # Switch dimensions to gpm_api standard dimension names
    ds_dims = list(ds.dims)
    rename_dim_dict = {}
    for dim in ds_dims:
        new_dim = DIM_DICT.get(dim, None)
        if new_dim:
            rename_dim_dict[dim] = new_dim
    ds = ds.rename_dims(rename_dim_dict)
    return ds
