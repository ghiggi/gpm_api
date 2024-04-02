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

import xarray as xr

import gpm


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
    scan_modes=None,
    variables=None,
    groups=None,
    verbose=True,
    decode_cf=True,
    chunks={},
):
    """Collocate a product on the provided dataset."""
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
        force_download=False,
        verbose=verbose,
    )

    # Read datasets
    list_ds = [
        gpm.open_dataset(
            product=product,
            start_time=start_time,
            end_time=end_time,
            # Optional
            version=version,
            variables=variables,
            groups=groups,
            product_type=product_type,
            scan_mode=scan_mode,
            chunks=chunks,
            decode_cf=decode_cf,
            prefix_group=False,
        )
        for scan_mode in scan_modes
    ]

    # Remap datasets
    list_remapped = [src_ds.gpm.remap_on(ds) for src_ds in list_ds]

    # Concatenate if necessary (PMW case)
    output_ds = xr.concat(list_remapped, dim="pmw_frequency") if len(list_remapped) > 1 else list_remapped[0]

    # Assign attributes
    output_ds.attrs = list_ds[0].attrs
    output_ds.attrs["ScanMode"] = scan_modes

    return output_ds
