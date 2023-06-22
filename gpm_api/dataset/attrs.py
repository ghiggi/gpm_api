#!/usr/bin/env python3
"""
Created on Thu Jun 22 14:52:38 2023

@author: ghiggi
"""
import datetime

from gpm_api.utils.utils_HDF5 import hdf5_file_attrs

STATIC_GLOBAL_ATTRS = (
    ## FileHeader
    "DOI",
    "DOIauthority",
    "AlgorithmID",
    "AlgorithmVersion",
    "ProductVersion",
    "SatelliteName",
    "InstrumentName",
    "ProcessingSystem",  # "PPS" or "JAXA"
    # EmptyGranule (granule discarded if empty)
    ## FileInfo,
    "DataFormatVersion",
    "MetadataVersion",
    ## JaxaInfo
    "ProcessingMode",  # "STD", "NRT"
    ## SwathHeader
    "ScanType",
    ## GSMaPInfo
    "AlgorithmName",
    ## GprofInfo
    "Satellite",
    "Sensor",
)


GRANULE_ONLY_GLOBAL_ATTRS = (
    ## FileHeader
    "FileName",
    ## Navigation Record
    "EphemerisFileName",
    "AttitudeFileName",
    ## JaxaInfo
    "TotalQualityCode",
    "DielectricFactorKa",
    "DielectricFactorKu",
)

DYNAMIC_GLOBAL_ATTRS = (
    "MissingData",  # number of missing scans
    "NumberOfRainPixelsFS",
    "NumberOfRainPixelsHS",
)


def get_granule_attrs(hdf):
    """Get granule global attributes."""
    # Retrieve attributes dictionary (per group)
    hdf_attr = hdf5_file_attrs(hdf)
    # Flatten attributes (without group)
    attrs = {}
    _ = [attrs.update(group_attrs) for group, group_attrs in hdf_attr.items()]
    # Subset only required attributes
    valid_keys = GRANULE_ONLY_GLOBAL_ATTRS + DYNAMIC_GLOBAL_ATTRS + STATIC_GLOBAL_ATTRS
    attrs = {key: attrs[key] for key in valid_keys if key in attrs}
    return attrs


def add_history(ds):
    current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    history = f"Created by ghiggi/gpm_api software on {current_time}"
    ds.attrs["history"] = history
    return ds
