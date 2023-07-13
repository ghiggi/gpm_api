#!/usr/bin/env python3
"""
Created on Thu Jun 22 15:01:43 2023

@author: ghiggi
"""
import re

# TODO: write dict in a config YAML file ?


DIM_DICT = {
    # 2A-DPR
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
    "nDSD": "DSD_params",
    # 2B-GPM-CORRA
    "nBnPSD": "range",  # 88 bins (250 m each bin)
    # "nBnEnv": "nBnEnv",
    "nemiss": "frequency_gmi",
    "nKuKa": "frequency_dpr",
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
    # PMW 1A and 1C-GMI (V7)
    "nscan1": "along_track",
    "nscan2": "along_track",
    "nscan3": "along_track",
    "nscan4": "along_track",
    "nscan5": "along_track",
    "nscan6": "along_track",
    "npixelev1": "cross_track",
    "npixelev2": "cross_track",
    "npixelev3": "cross_track",
    "npixelev4": "cross_track",
    "npixelev5": "cross_track",
    "npixelev6": "cross_track",
    "nchannel1": "channel",
    "nchannel2": "channel",
    "nchannel3": "channel",
    "nchannel4": "channel",
    "nchannel5": "channel",
    "nchannel6": "channel",
    # PMW 1A-GMI (V7)
    "npixelev": "cross_track",
    "npixelht": "cross_track",
    "npixelcs": "cross_track",
    "npixelfr": "cross_track",  # S4 mode
    # nfreqHI
    # nlayer --> in CSH, SLH --> converted to height in decoding
}

SPATIAL_DIMS = [
    "along_track",
    "cross_track",
    "lat",
    "lon",  # choose whether to use instead latitude/longitude
    "latitude",
    "longitude",
    "x",
    "y",  # compatibility with satpy/gpm_geo i.e.
]
VERTICAL_DIMS = ["range", "nBnEnv"]
FREQUENCY_DIMS = ["channel", "frequency", "frequency_gmi", "frequency_dpr"]
GRID_SPATIAL_DIMS = ("lon", "lat")
ORBIT_SPATIAL_DIMS = ("cross_track", "along_track")


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
