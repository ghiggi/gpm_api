#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:47:45 2023

@author: ghiggi
"""
import dask


def _get_valid_granule(
    filepath,
    scan_mode,
    variables,
    # groups
    decode_cf,
    prefix_group,
    chunks,
):
    if _is_valid_granule(filepath):
        ds = open_granule(
            filepath,
            scan_mode=scan_mode,
            variables=variables,
            # groups=groups, # TODO
            decode_cf=decode_cf,
            prefix_group=prefix_group,
            chunks=chunks,
        )
        return ds
    else:
        return None


def _get_list_ds_granule(
    filepaths,
    scan_mode,
    variables,
    # groups,
    decode_cf,
    prefix_group,
    chunks,
):
    l_datasets = []
    for filepath in filepaths:
        ds = _get_valid_granule(
            filepath,
            scan_mode=scan_mode,
            variables=variables,
            # groups=groups, # TODO
            decode_cf=decode_cf,
            prefix_group=prefix_group,
            chunks=chunks,
        )
        if ds is not None:
            l_datasets.append(ds)
    return l_datasets


def _get_delayed_list_ds_granule(
    filepaths,
    scan_mode,
    variables,
    # groups,
    decode_cf,
    prefix_group,
    chunks,
):
    l_tasks = []
    for filepath in filepaths:
        # Retrieve data if granule is not empty
        task = dask.delayed(_get_valid_granule)(
            filepath,
            scan_mode=scan_mode,
            variables=variables,
            # groups=groups, # TODO
            decode_cf=decode_cf,
            prefix_group=prefix_group,
            chunks=chunks,
        )
        l_tasks.append(task)

    l_datasets = dask.compute(*l_tasks)
    return l_datasets


from gpm_api.utils.archive import print_elapsed_time


@print_elapsed_time
def get_list_granules(
    filepaths,
    scan_mode,
    variables,
    # groups,
    decode_cf,
    prefix_group,
    chunks,
    parallel=False,
):
    # scheduler = "threads" if parallel else "single-threaded"
    if parallel:
        l_datasets = _get_delayed_list_ds_granule(
            filepaths=filepaths,
            scan_mode=scan_mode,
            variables=variables,
            # groups=groups, # TODO
            decode_cf=decode_cf,
            prefix_group=prefix_group,
            chunks=chunks,
        )
    else:
        l_datasets = _get_list_ds_granule(
            filepaths=filepaths,
            scan_mode=scan_mode,
            variables=variables,
            # groups=groups, # TODO
            decode_cf=decode_cf,
            prefix_group=prefix_group,
            chunks=chunks,
        )

    return l_datasets


l_datasets = get_list_granules(
    filepaths=filepaths,
    scan_mode=scan_mode,
    variables=variables,
    # groups=groups, # TODO
    decode_cf=decode_cf,
    prefix_group=prefix_group,
    chunks=chunks,
    parallel=False,
)
l_datasets = [ds for ds in l_datasets if ds is not None]
ds = xr.concat(l_datasets, dim="along_track")

with dask.config.set(scheduler="threads"):
    l_datasets = get_list_granules(
        filepaths=filepaths,
        scan_mode=scan_mode,
        variables=variables,
        # groups=groups, # TODO
        decode_cf=decode_cf,
        prefix_group=prefix_group,
        chunks=chunks,
        parallel=True,
    )
