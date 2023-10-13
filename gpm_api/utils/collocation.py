#!/usr/bin/env python3
"""
Created on Mon Jul 31 17:18:10 2023

@author: ghiggi
"""
import datetime

import xarray as xr

import gpm_api


def _get_collocation_defaults_args(product, variables, groups, version, scan_modes):
    """Get collocation defaults arguments."""
    if scan_modes is None:
        scan_modes = gpm_api.available_scan_modes(product=product, version=version)
    if isinstance(scan_modes, str):
        scan_modes = [scan_modes]
    if product not in gpm_api.available_products(product_category="PMW"):
        if len(scan_modes) > 1:
            raise ValueError("Multiple scan modes can be specified only for PMW products!")
    # PMW defaults
    if variables is None and groups is None:
        if product in gpm_api.available_products(product_level="2A", product_category="PMW"):
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
    scan_modes=None,
    variables=None,
    groups=None,
    verbose=True,
    chunks={},
):
    """Collocate a product on the provided dataset."""
    # Get default collocation arguments
    scan_modes, variables, groups = _get_collocation_defaults_args(
        product=product, variables=variables, groups=groups, version=version, scan_modes=scan_modes
    )

    # Define start_time, end_time around input dataset
    start_time = ds.gpm_api.start_time - datetime.timedelta(minutes=5)
    end_time = ds.gpm_api.end_time + datetime.timedelta(minutes=5)

    # Download PMW products (if necessary)
    gpm_api.download(
        product=product,
        product_type=product_type,
        start_time=start_time,
        end_time=end_time,
        version=version,
        force_download=False,
        verbose=verbose,
    )

    # Read datasets
    list_ds = [
        gpm_api.open_dataset(
            product=product,
            start_time=start_time,
            end_time=end_time,
            # Optional
            version=version,
            variables=variables,
            groups=groups,
            product_type=product_type,
            scan_mode=scan_mode,
            chunks={},
            decode_cf=True,
            prefix_group=False,
        )
        for scan_mode in scan_modes
    ]

    # Remap datasets
    list_remapped = [src_ds.gpm_api.remap_on(ds) for src_ds in list_ds]

    # Concatenate if necessary (PMW case)
    if len(list_remapped) > 1:
        output_ds = xr.concat(list_remapped, dim="pmw_frequency")
    else:
        output_ds = list_remapped[0]

    # Assign attributes
    output_ds.attrs = list_ds[0].attrs
    output_ds.attrs["ScanMode"] = scan_modes

    return output_ds
