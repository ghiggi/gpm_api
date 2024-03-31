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
"""This module contains functions to retrieve the dimensions associated to each GPM variable."""
import numpy as np
import xarray as xr

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
    "nfreq": "radar_frequency",
    "nDSD": "DSD_params",
    # 2B-GPM-CORRA
    "nemiss": "pmw_frequency",
    "nKuKa": "radar_frequency",
    "nBnPSD": "range",  # V7 88 bins (250 m each bin)
    "nBnPSDhi": "range",  # V6 88 bins (250 m each bin)
    # "nBnEnv": "nBnEnv",
    # 2B-GPM SLH/CSH
    "nlayer": "range",
    # PMW 1B-GMI (V7)
    "npix1": "cross_track",  # PMW (i.e. GMI)
    "npix2": "cross_track",  # PMW (i.e. GMI)
    "nchan1": "pmw_frequency",
    "nchan2": "pmw_frequency",
    # PMW 1C-<PMW> (V7)
    "npixel1": "cross_track",
    "npixel2": "cross_track",
    "npixel3": "cross_track",
    "npixel4": "cross_track",
    "npixel5": "cross_track",
    "npixel6": "cross_track",
    # PMW 1A and 1C-<PMW> (V7)
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
    "nchannel1": "pmw_frequency",
    "nchannel2": "pmw_frequency",
    "nchannel3": "pmw_frequency",
    "nchannel4": "pmw_frequency",
    "nchannel5": "pmw_frequency",
    "nchannel6": "pmw_frequency",
    # PMW 1A-GMI (V7)
    "npixelev": "cross_track",
    "npixelht": "cross_track",
    "npixelcs": "cross_track",
    "npixelfr": "cross_track",  # S4 mode
}

SPATIAL_DIMS = [
    ["along_track", "cross_track"],
    ["lat", "lon"],  # choose whether to use instead latitude/longitude
    ["latitude", "longitude"],
    ["x", "y"],  # compatibility with satpy/gpm_geo i.e.
]
VERTICAL_DIMS = ["range", "nBnEnv", "height"]
FREQUENCY_DIMS = ["radar_frequency", "pmw_frequency"]
GRID_SPATIAL_DIMS = ("lon", "lat")
ORBIT_SPATIAL_DIMS = ("cross_track", "along_track")


# TODO: read dictionary from yaml config


def _has_a_phony_dim(xr_obj):
    """Check if the xarray object has a phony_dim_<number> dimension."""
    return np.any([dim.startswith("phony_dim") for dim in list(xr_obj.dims)]).item()


def _get_dataarray_dim_dict(da):
    """Return a dictionary mapping each DataArray phony_dim to the actual dimension name."""
    dim_dict = {}
    dim_names_str = da.attrs.get("DimensionNames", None)
    if dim_names_str is not None:
        dim_names = dim_names_str.split(",")
        for dim, new_dim in zip(list(da.dims), dim_names):
            # Deal with missing DimensionNames in
            # - sunVectorInBodyFrame variable in V5 products
            # - 1B-Ku/Ka V5 *Temp products
            if new_dim == "":
                new_dim = dim
            dim_dict[dim] = new_dim
    return dim_dict


def _get_dataset_dim_dict(ds):
    """Return a dictionary mapping each Dataset phony_dim to the actual dimension name."""
    rename_dim_dict = {}
    for var in list(ds.data_vars):
        rename_dim_dict.update(_get_dataarray_dim_dict(ds[var]))
    return rename_dim_dict


def _get_datatree_dim_dict(dt):
    """Return a dictionary mapping each DataTree phony_dim to the actual dimension name."""
    rename_dim_dict = {}
    for group in dt.groups:
        ds = dt[group]
        if _has_a_phony_dim(ds):
            rename_dim_dict.update(_get_dataset_dim_dict(ds))
    return rename_dim_dict


def _get_gpm_api_dims_dict(ds):
    """Get dictionary to rename dimensions following gpm-api defaults."""
    rename_dim_dict = {}
    for dim in list(ds.dims):
        new_dim = DIM_DICT.get(dim)
        if new_dim is not None:
            rename_dim_dict[dim] = new_dim
    return rename_dim_dict


def _rename_datarray_dimensions(da):
    """Rename DataArray dimensions."""
    if _has_a_phony_dim(da):
        da = da.rename(_get_dataarray_dim_dict(da))
    return da


def _rename_dataset_dimensions(ds, use_api_defaults=True):
    """Rename xr.Dataset dimension to the actual dimension names.

    The actual dimensions names are retrieved from the DataArrays DimensionNames attribute.
    The dimension renaming is performed at each Dataset level.
    If use_api_defaults is True (the default), it sets the GPM-API dimension names.
    """
    dict_da = {var: _rename_datarray_dimensions(ds[var]) for var in ds.data_vars}
    ds = xr.Dataset(dict_da, attrs=ds.attrs)
    if use_api_defaults:
        ds = ds.rename_dims(_get_gpm_api_dims_dict(ds))
    return ds


def _rename_datatree_dimensions(dt, use_api_defaults=True):
    """Rename datatree dimension to the actual dimension names.

    The actual dimensions names are retrieved from the DataArrays DimensionNames attribute.
    The renaming is performed at the DataArray level because DataArrays sharing same dimension
    size (but semantic different dimension) are given the same phony_dim_number within xr.Dataset !

    The dimension renaming is performed at each Dataset level.
    If use_api_defaults is True (the default), it sets the GPM-API dimension names.
    """
    # TODO: report reproducible issue on datatree repo ...
    # --> dt[group].rename_dims(dim_dict) does not work
    # --> dt.rename_dims(dim_dict) does work only at node level !
    dt = dt.copy()  # otherwise rename input dt
    for group in dt.groups:
        dt[group].ds = _rename_dataset_dimensions(dt[group], use_api_defaults=use_api_defaults)

    return dt
