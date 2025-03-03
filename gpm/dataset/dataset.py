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
"""This module contains functions to read files into a GPM-API Dataset or DataTree."""
import warnings
from functools import partial

import xarray as xr

from gpm.configs import get_base_dir
from gpm.dataset.conventions import finalize_dataset
from gpm.dataset.granule import _multi_file_closer, get_scan_modes_datasets
from gpm.io.checks import (
    check_groups,
    check_product,
    check_scan_mode,
    check_scan_modes,
    check_start_end_time,
    check_valid_time_request,
    check_variables,
)
from gpm.io.find import find_filepaths
from gpm.utils.checks import has_missing_granules
from gpm.utils.warnings import GPM_Warning

# from gpm.utils.dask import get_scheduler


def _concat_datasets(l_datasets):
    """Concatenate datasets together."""
    dims = list(l_datasets[0].dims)
    is_grid = "time" in dims
    concat_dim = "time" if is_grid else "along_track"

    # Concatenate the datasets
    return xr.concat(
        l_datasets,
        dim=concat_dim,
        coords="minimal",  # "all"
        compat="override",
        combine_attrs="override",
    )


def _try_open_granule(filepath, scan_modes, decode_cf, variables, groups, prefix_group, chunks, **kwargs):
    """Try open a granule."""
    try:
        dict_ds_scan_modes, dt_closer = get_scan_modes_datasets(
            filepath=filepath,
            scan_modes=scan_modes,
            groups=groups,
            variables=variables,
            decode_cf=decode_cf,
            chunks=chunks,
            prefix_group=prefix_group,
            **kwargs,
        )
    except Exception as e:
        msg = f"The following error occurred while opening the {filepath} granule: {e}"
        warnings.warn(msg, GPM_Warning, stacklevel=2)
        dict_ds_scan_modes = None
        dt_closer = None
    return dict_ds_scan_modes, dt_closer


def _get_scan_modes_datasets_and_closers(filepaths, parallel, scan_modes, decode_cf=False, **open_kwargs):
    """Open the granule in parallel with dask delayed."""
    # Define functions to open files
    if parallel:
        import dask

        open_ = dask.delayed(_try_open_granule)
    else:
        open_ = _try_open_granule

    # ----------------------------------------------------.
    # Open files
    list_info = [open_(filepath, scan_modes=scan_modes, decode_cf=decode_cf, **open_kwargs) for filepath in filepaths]

    # If parallel=True, compute the delayed datasets lists here
    # - The underlying data are stored as dask arrays (and are not computed !)
    if parallel:
        list_info = dask.compute(*list_info)

    # ----------------------------------------------------.
    # Retrieve datatree closers
    list_dt_closers = [dt_closer for _, dt_closer in list_info]

    # Retrieve scan modes closers
    list_dict_scan_modes = [dict_scan_modes for dict_scan_modes, _ in list_info]

    # Remove None elements from the list
    list_dt_closers = [closer for closer in list_dt_closers if closer is not None]
    list_dict_scan_modes = [dict_scan_modes for dict_scan_modes in list_dict_scan_modes if dict_scan_modes is not None]

    # Check there are valid granules
    if len(list_dict_scan_modes) == 0:
        raise ValueError("Impossible to open GPM granules with current request.")

    # Define list of dataset for each scan_mode
    dict_scan_modes_datasets = {
        scan_mode: [dict_scan_modes[scan_mode] for dict_scan_modes in list_dict_scan_modes] for scan_mode in scan_modes
    }
    # Define list of closers for each scan_mode
    dict_scan_modes_closers = {
        scan_mode: [dict_scan_modes[scan_mode]._close for dict_scan_modes in list_dict_scan_modes]
        for scan_mode in scan_modes
    }

    # Concat datasets within each scan mode
    dict_scan_modes_dataset = {
        scan_mode: _concat_datasets(list_datasets) for scan_mode, list_datasets in dict_scan_modes_datasets.items()
    }

    # Specify scan modes closers
    for scan_mode, scan_modes_closers in dict_scan_modes_closers.items():
        dict_scan_modes_dataset[scan_mode].set_close(partial(_multi_file_closer, scan_modes_closers))

    return dict_scan_modes_dataset, list_dt_closers


def open_dataset(
    product,
    start_time,
    end_time,
    variables=None,
    groups=None,
    scan_mode=None,
    version=None,
    product_type="RS",
    chunks=-1,
    decode_cf=True,
    parallel=False,
    prefix_group=False,
    verbose=False,
    base_dir=None,
    **kwargs,
):
    """Lazily map HDF5 data into xarray.Dataset with relevant GPM data and attributes.

    Note:

    - ``gpm.open_dataset`` does not load GPM granules with the FileHeader flag ``'EmptyGranule' != 'NOT_EMPTY'``.
    - The coordinates ``Quality`` or ``dataQuality`` provide an overall quality flag status.
    - The coordinate ``SCorientation`` provides the orientation of the sensor from the forward track of the satellite.

    Parameters
    ----------
    product : str
        GPM product acronym.
    start_time :  datetime.datetime, datetime.date, numpy.datetime64 or str
        Start time.
        Accepted types: ``datetime.datetime``, ``datetime.date``, ``numpy.datetime64`` or ``str``.
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.
    end_time :  datetime.datetime, datetime.date, numpy.datetime64 or str
        End time.
        Accepted types: ``datetime.datetime``, ``datetime.date``, ``numpy.datetime64`` or ``str``.
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.
    variables : list, str, optional
        Variables to read from the HDF5 file.
        The default is ``None`` (all variables).
    groups : list, str, optional
        HDF5 Groups from which to read all variables.
        The default is ``None`` (all groups).
    scan_mode : str, optional
        Scan mode of the GPM product. The default is ``None``.
        Use ``gpm.available_scan_modes(product, version)`` to get the available scan modes for a specific product.
        The radar products have the following scan modes:

        - ``'FS'``: Full Scan. For Ku, Ka and DPR (since version 7 products).
        - ``'NS'``: Normal Scan. For Ku band and DPR (till version 6 products).
        - ``'MS'``: Matched Scan. For Ka band and DPR (till version 6 products).
        - ``'HS'``: High-sensitivity Scan. For Ka band and DPR.

    product_type : str, optional
        GPM product type. Either ``'RS'`` (Research) or ``'NRT'`` (Near-Real-Time).
        The default is ``'RS'``.
    version : int, optional
        GPM version of the data to retrieve if ``product_type = "RS"``.
        GPM data readers currently support version 4, 5, 6 and 7.
    chunks : int, dict, str or None, optional
        Chunk size for dask array:

        - ``chunks=-1`` loads the dataset with dask using a single chunk for each granule arrays.
        - ``chunks={}`` loads the dataset with dask using the file chunks.
        - ``chunks='auto'`` will use dask ``auto`` chunking taking into account the file chunks.

        If you want to load data in memory directly, specify ``chunks=None``.
        The default is ``auto``.

        Hint: xarray's lazy loading of remote or on-disk datasets is often but not always desirable.
        Before performing computationally intense operations, load the dataset
        entirely into memory by invoking ``ds.compute()``.
    decode_cf: bool, optional
        Whether to decode the dataset. The default is ``False``.
    prefix_group: bool, optional
        Whether to add the group as a prefix to the variable names.
        If you aim to save the Dataset to disk as netCDF or Zarr, you need to set ``prefix_group=False``
        or later remove the prefix before writing the dataset.
        The default is ``False``.
    parallel : bool
        If ``True``, the dataset are opened in parallel using :py:class:`dask.delayed.delayed`.
        If ``parallel=True``, ``'chunks'`` can not be ``None``.
        The underlying data must be :py:class:`dask.array.Array`.
        The default is ``False``.
    **kwargs : dict
        Additional keyword arguments passed to :py:func:`~xarray.open_dataset` for each group.

    Returns
    -------
    xarray.Dataset

    """
    ## Check valid product and variables
    base_dir = get_base_dir(base_dir=base_dir)
    product = check_product(product, product_type=product_type)
    variables = check_variables(variables)
    groups = check_groups(groups)

    ## Check scan_mode
    scan_mode = check_scan_mode(scan_mode, product, version=version)

    # Check valid start/end time
    start_time, end_time = check_start_end_time(start_time, end_time)
    start_time, end_time = check_valid_time_request(start_time, end_time, product)

    # Check parallel and chunks arguments
    if parallel and chunks is None:
        raise ValueError("If parallel=True, 'chunks' can not be None.")

    ##------------------------------------------------------------------------.
    # Find filepaths
    filepaths = find_filepaths(
        storage="LOCAL",
        version=version,
        product=product,
        product_type=product_type,
        start_time=start_time,
        end_time=end_time,
        verbose=verbose,
        base_dir=base_dir,
    )

    ##------------------------------------------------------------------------.
    # Check that files have been downloaded on disk
    if len(filepaths) == 0:
        raise ValueError("No files found on disk. Please download them before.")

    ##------------------------------------------------------------------------.
    # Open and concatenate the scan mode of each granule
    dict_scan_modes, list_dt_closers = _get_scan_modes_datasets_and_closers(
        filepaths=filepaths,
        parallel=parallel,
        scan_modes=[scan_mode],
        decode_cf=False,
        # Custom options
        variables=variables,
        groups=groups,
        prefix_group=prefix_group,
        chunks=chunks,
        **kwargs,
    )
    ds = dict_scan_modes[scan_mode]

    ##-------------------------------------------------------------------------.
    # TODO - Extract attributes and add as coordinate ?
    # - From each granule, select relevant (discard/sum values/copy)
    # - Sum of MissingData, NumberOfRainPixels
    # - MissingData in FileHeaderGroup: The number of missing scans.

    ##-------------------------------------------------------------------------.
    # Finalize dataset
    ds = finalize_dataset(
        ds=ds,
        product=product,
        scan_mode=scan_mode,
        decode_cf=decode_cf,
        start_time=start_time,
        end_time=end_time,
    )

    ##------------------------------------------------------------------------.
    # Warns about missing granules
    if has_missing_granules(ds):
        msg = "The GPM Dataset has missing granules !"
        warnings.warn(msg, GPM_Warning, stacklevel=1)

    ##-------------------------------------------------------------------------.
    # Specify files closers
    ds.set_close(partial(_multi_file_closer, list_dt_closers))

    ##------------------------------------------------------------------------.
    # Return Dataset
    return ds


def open_datatree(
    product,
    start_time,
    end_time,
    variables=None,
    groups=None,
    scan_modes=None,
    version=None,
    product_type="RS",
    chunks=-1,
    decode_cf=True,
    parallel=False,
    prefix_group=False,
    verbose=False,
    base_dir=None,
    **kwargs,
):
    """Lazily map HDF5 data into xarray.DataTree objects with relevant GPM data and attributes.

    Note:

    - ``gpm.open_datatree`` does not load GPM granules with the FileHeader flag
    ``'EmptyGranule' != 'NOT_EMPTY'``.
    - The coordinates ``Quality`` or ``dataQuality`` provide an overall quality flag status.
        If the flag value is 0, no issues have been detected.
    - The coordinate ``SCorientation`` provides the orientation of the sensor
      from the forward track of the satellite.

    Parameters
    ----------
    product : str
        GPM product acronym.
    start_time :  datetime.datetime, datetime.date, numpy.datetime64 or str
        Start time.
        Accepted types: ``datetime.datetime``, ``datetime.date``, ``numpy.datetime64`` or ``str``.
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.
    end_time :  datetime.datetime, datetime.date, numpy.datetime64 or str
        End time.
        Accepted types: ``datetime.datetime``, ``datetime.date``, ``numpy.datetime64`` or ``str``.
        If string type, it expects the isoformat ``YYYY-MM-DD hh:mm:ss``.
    variables : list, str, optional
        Variables to read from the HDF5 file.
        The default is ``None`` (all variables).
    groups : list, str, optional
        HDF5 Groups from which to read all variables.
        The default is ``None`` (all groups).
    scan_modes : str, optional
        Scan mode of the GPM product. If ``None`` (the default), loads all scan modes.
        Use ``gpm.available_scan_modes(product, version)`` to see the available scan modes for a specific product.
        The radar products have the following scan modes:
        - ``'FS'``: Full Scan. For Ku, Ka and DPR (since version 7 products).
        - ``'NS'``: Normal Scan. For Ku band and DPR (till version 6 products).
        - ``'MS'``: Matched Scan. For Ka band and DPR (till version 6 products).
        - ``'HS'``: High-sensitivity Scan. For Ka band and DPR.
    product_type : str, optional
        GPM product type. Either ``'RS'`` (Research) or ``'NRT'`` (Near-Real-Time).
        The default is ``'RS'``.
    version : int, optional
        GPM version of the data to retrieve if ``product_type = "RS"``.
        GPM data readers currently support version 4, 5, 6 and 7.
    chunks : int, dict, str or None, optional
        Chunk size for dask array:

        - ``chunks=-1`` loads the dataset with dask using a single chunk for each granule arrays.
        - ``chunks={}`` loads the dataset with dask using the file chunks.
        - ``chunks='auto'`` will use dask ``auto`` chunking taking into account the file chunks.

        If you want to load data in memory directly, specify ``chunks=None``.
        The default is ``auto``.

        Hint: xarray's lazy loading of remote or on-disk datasets is often but not always desirable.
        Before performing computationally intense operations, load the dataset
        entirely into memory by invoking ``ds.compute()``.
    decode_cf: bool, optional
        Whether to decode the dataset. The default is ``False``.
    prefix_group: bool, optional
        Whether to add the group as a prefix to the variable names.
        If you aim to save the Dataset to disk as netCDF or Zarr, you need to set ``prefix_group=False``
        or later remove the prefix before writing the dataset.
        The default is ``False``.
    parallel : bool
        If ``True``, the dataset are opened in parallel using :py:class:`dask.delayed.delayed`.
        If ``parallel=True``, ``'chunks'`` can not be ``None``.
        The underlying data must be :py:class:`dask.array.Array`.
        The default is ``False``.
    **kwargs : dict
        Additional keyword arguments passed to :py:func:`~xarray.open_datatree` for each group.

    Returns
    -------
    xarray.DataTree

    """
    ## Check valid product and variables
    product = check_product(product, product_type=product_type)
    variables = check_variables(variables)
    groups = check_groups(groups)
    base_dir = get_base_dir(base_dir=base_dir)
    # Check scan_modes
    scan_modes = check_scan_modes(scan_modes=scan_modes, product=product, version=version)

    # Check valid start/end time
    start_time, end_time = check_start_end_time(start_time, end_time)
    start_time, end_time = check_valid_time_request(start_time, end_time, product)

    # Check parallel and chunks arguments
    if parallel and chunks is None:
        raise ValueError("If parallel=True, 'chunks' can not be None.")

    ##------------------------------------------------------------------------.
    # Find filepaths
    filepaths = find_filepaths(
        storage="LOCAL",
        version=version,
        product=product,
        product_type=product_type,
        start_time=start_time,
        end_time=end_time,
        verbose=verbose,
        base_dir=base_dir,
    )

    ##------------------------------------------------------------------------.
    # Check that files have been downloaded on disk
    if len(filepaths) == 0:
        raise ValueError("No files found on disk. Please download them before.")

    ##------------------------------------------------------------------------.
    dict_scan_modes, list_dt_closers = _get_scan_modes_datasets_and_closers(
        filepaths=filepaths,
        parallel=parallel,
        scan_modes=scan_modes,
        decode_cf=False,
        # Custom options
        variables=variables,
        groups=groups,
        prefix_group=prefix_group,
        chunks=chunks,
        **kwargs,
    )

    # Finalize datatree
    dict_scan_modes = {
        scan_mode: finalize_dataset(
            ds=ds,
            product=product,
            scan_mode=scan_mode,
            decode_cf=decode_cf,
            start_time=start_time,
            end_time=end_time,
        )
        for scan_mode, ds in dict_scan_modes.items()
    }

    # Create datatree
    dt = xr.DataTree.from_dict(dict_scan_modes)

    # Specify scan modes closers
    for scan_mode, ds in dict_scan_modes.items():
        dt[scan_mode].set_close(ds._close)

    # Specify files closers
    dt.set_close(partial(_multi_file_closer, list_dt_closers))

    ##------------------------------------------------------------------------.
    # Warns about missing granules
    if has_missing_granules(dt[scan_mode]):
        msg = "The GPM DataTree has missing granules !"
        warnings.warn(msg, GPM_Warning, stacklevel=1)

    ##------------------------------------------------------------------------.
    return dt
