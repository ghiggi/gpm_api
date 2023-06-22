#!/usr/bin/env python3
"""
Created on Mon Aug 15 00:12:47 2022

@author: ghiggi
"""
import warnings

import numpy as np
import xarray as xr

from gpm_api.utils.warnings import GPM_Warning

####--------------------------------------------------------------------------.
#### Attributes cleaning


def clean_dataarrays_attrs(ds, product):
    for var, da in ds.items():
        ds[var] = _format_dataarray_attrs(da, product)
    return ds


def _format_dataarray_attrs(da, product=None):
    attrs = da.attrs

    # Convert CodeMissingValue' to _FillValue if available
    if not attrs.get("_FillValue", False) and attrs.get("CodeMissingValue", False):
        attrs["_FillValue"] = attrs.pop("CodeMissingValue")

    # Remove 'CodeMissingValue'
    attrs.pop("CodeMissingValue", None)

    # Convert 'Units' to 'units'
    if not attrs.get("units", False) and attrs.get("Units", False):
        attrs["units"] = attrs.pop("Units")

    # Remove 'Units'
    attrs.pop("Units", None)

    # Remove 'DimensionNames'
    attrs.pop("DimensionNames", None)

    # Add source dtype from encoding
    # print(da.name)
    # print(da.encoding)
    if da.encoding.get("dtype", False):
        attrs["source_dtype"] = da.encoding["dtype"]

    # Add gpm_api product name
    if product is not None:
        attrs["gpm_api_product"] = product

    # Attach attributes
    da.attrs = attrs
    return da


####--------------------------------------------------------------------------.
##################
#### Decoding ####
##################
#  Decode a posteriori
# https://docs.xarray.dev/en/stable/generated/xarray.decode_cf.html


def decode_dataset(ds):
    # Decode with xr.decode_cf
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        ds = xr.decode_cf(ds)

    # Clean the DataArray attributes and encodings
    for var, da in ds.items():
        # When decoding with xr.decode_cf, _FillValue and the source dtype are automatically
        # added to the encoding attribute
        ds[var].attrs.pop("source_dtype", None)
        ds[var].attrs.pop("_FillValue", None)
        # Remove hdf encodings
        ds[var].encoding.pop("szip", None)
        ds[var].encoding.pop("zstd", None)
        ds[var].encoding.pop("bzip2", None)
        ds[var].encoding.pop("blosc", None)

    # = _format_dataarray_attrs(da, product)
    # TODO: preprocess attribute and convert offset_scale
    # dataset_var = list(ds.data_vars)
    # for var in dataset_var:
    #     da = ds[var]
    #     fillValue = da.attrs.get("_FillValue", False)
    #     if fillValue:
    #         ds[var] = xr.where(da.isin(fillValue), np.nan, da)
    # ## Add scale and offset
    # if len(variables_dict[var]["offset_scale"]) == 2:
    #     da = (da / variables_dict[var]["offset_scale"][1] - variables_dict[var]["offset_scale"][0])
    return ds


####------------------------------------------------------------------------.
#########################
#### Custom decoding ####
#########################
# Add optional coordinates
# - altitude...

# Maybe modify source_dtype to facilitate encoding to new netcdf

# TODO ENV
# if (var == 'cloudLiquidWater'):
#     # nwater ,
# if (var == 'waterVapor'):
#     # nwater

# if (var == 'phase'):
#   print('Decoding of phase not yet implemented')

# if (var == 'typePrecip'):
#   print('Decoding of typePrecip not yet implemented')


def ensure_valid_coords(ds, raise_error=False):
    # invalid_coords = np.logical_or(ds["lon"].data == -9999.9,
    #                                ds["lat"].data == -9999.9)
    invalid_coords = np.logical_or(
        np.logical_or(ds["lon"].data < -180, ds["lon"].data > 180),
        np.logical_or(ds["lat"].data < -90, ds["lat"].data > 90),
    )
    if np.any(invalid_coords):
        # Raise error or add warning
        msg = "Invalid coordinate in the granule."
        if raise_error:
            raise ValueError(msg)
        else:
            warnings.warn(msg, GPM_Warning)

        da_invalid_coords = ds["lon"].copy()
        da_invalid_coords.data = invalid_coords
        # For each variable, set NaN value where invalid coordinates
        ds = ds.where(~da_invalid_coords)
        # Add NaN to longitude and latitude
        ds["lon"] = ds["lon"].where(~da_invalid_coords)
        ds["lat"] = ds["lat"].where(~da_invalid_coords)
    return ds


def apply_custom_decoding(ds, product):
    # Ensure valid coordinates
    if "cross_track" in list(ds.dims):
        ds = ensure_valid_coords(ds, raise_error=False)

    # Add range_id coordinate
    if "range" in list(ds.dims):
        range_id = np.arange(ds.dims["range"])
        ds = ds.assign_coords({"gpm_range_id": ("range", range_id)})

    # Clean attributes
    ds = clean_dataarrays_attrs(ds, product)

    # Add coordinates
    if product == "2A-DPR" and "frequency" in list(ds.dims):
        ds = ds.assign_coords({"frequency": ["Ku", "Ka"]})

    if product in ["2A-DPR", "2A-Ku", "2A-Ka"] and "paramDSD" in list(ds):
        ds = ds.assign_coords({"DSD_params": ["Nw", "Dm"]})

    if product in ["2A-DPR", "2A-Ku", "2A-Ka"] and "height" in list(ds):
        ds = ds.set_coords("height")

    if product in ["2A-GPM-SLH", "2B-GPM-CSH"] and "nlayer" in list(ds.dims):
        # Fixed heights for 2HSLH and 2HCSH
        # - FileSpec v7: p.2395, 2463
        height = np.linspace(0.25 / 2, 20 - 0.25 / 2, 80) * 1000  # in meters
        ds = ds.rename_dims({"nlayer": "height"})
        ds = ds.assign_coords({"height": height})
        ds["height"].attrs["units"] = "km a.s.l"

    # Modify variables
    dataset_vars = list(ds.data_vars)
    if "precipWaterIntegrated" in dataset_vars:
        ds["precipWaterIntegrated_Liquid"] = ds["precipWaterIntegrated"][:, :, 0]
        ds["precipWaterIntegrated_Solid"] = ds["precipWaterIntegrated"][:, :, 1]
        ds = ds.drop_vars(names="precipWaterIntegrated")
        ds["precipWaterIntegrated"] = (
            ds["precipWaterIntegrated_Liquid"] + ds["precipWaterIntegrated_Solid"]
        )

    if "flagBB" in dataset_vars and product == "2A-DPR":
        ds["flagBB"].attrs[
            "description"
        ] = """Flag for Bright Band:
                                                0 : BB not detected
                                                1 : Bright Band detected by Ku and DFRm
                                                2 : Bright Band detected by Ku only
                                                3 : Bright Band detected by DFRm only
                                            """
    # if ds.attrs.get("TotalQualityCode"):
    #     TotalQualityCode = ds.attrs.get("TotalQualityCode")
    #     ds["TotalQualityCode"] = xr.DataArray(
    #         np.repeat(TotalQualityCode, ds.dims["along_track"]), dims=["along_track"]
    #     )

    # Correct for misreported _FillValue
    if "surfacePrecipitation" in dataset_vars:
        # _FillValue often reported as -9999.9, but in data the values are -9999.0 !
        # --> Example 2A-MHS-METOB
        da = ds["surfacePrecipitation"]
        ds["surfacePrecipitation"] = da.where(da != -9999.0)

    return ds
