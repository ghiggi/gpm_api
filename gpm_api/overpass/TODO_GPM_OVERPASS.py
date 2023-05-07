# /usr/bin/env python3
"""
Created on Tue Aug  2 10:43:23 2022

@author: ghiggi
"""
import datetime
import os

import dask
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar

from gpm_api.dataset import GPM_Dataset
from gpm_api.io import GPM_PMW_2A_GPROF_RS_products, download_GPM_data

BASE_DIR = "/home/ghiggi"
OVERPASS_TABLE_DIR = "/home/ghiggi/Overpass/Tables"
username = "gionata.ghiggi@epfl.ch"

#### Define analysis time period
start_time = datetime.datetime.strptime("2020-08-01 12:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-08-10 12:00:00", "%Y-%m-%d %H:%M:%S")

#### Define PMW products
products = GPM_PMW_2A_GPROF_RS_products()
product_type = "RS"

####--------------------------------------------------------------------------.
### Download products
for product in products:
    print(product)
    download_GPM_data(
        base_DIR=BASE_DIR,
        username=username,
        product=product,
        product_type=product_type,
        start_time=start_time,
        end_time=end_time,
        force_download=True,
        transfer_tool="curl",
        progress_bar=True,
        n_threads=10,
    )

####--------------------------------------------------------------------------.
#### Compute overpass tables
def generate_chunk_slices(chunks, shape):
    if len(shape) != 1:
        raise NotImplementedError()
    steps = np.arange(0, shape[0] + chunks, chunks)  # + 1 to include all?
    list_slices = [
        slice(steps[i], steps[i + 1]) for i in range(len(steps) - 1)
    ]  # +1 to the final end?
    return list_slices


@dask.delayed
def _reshape_ds_to_df_overpass(ds):
    df = ds.to_dataframe()
    df = df.reset_index(drop=True)
    df["time"] = df["time"].astype("M8[m]")
    # Drop rows with -9999 lat/lon
    df = df[df["lat"].values != -9999]
    df = df[df["lon"].values != -9999]
    # Restrict to 0.1° resolution cells
    df["lat"] = df["lat"].values * 10
    df["lat"] = df["lat"].astype("int16")
    df["lon"] = df["lon"].values * 10
    df["lon"] = df["lon"].astype("int16")
    # Drop duplicates
    df_unique = df.drop_duplicates()
    return df_unique


def generate_df_overpass_time(ds, chunks=1000):
    # Remove not relevant variables and coordinates
    ds_vars = list(ds.data_vars)
    ds = ds.drop(["granule_id", "scan_id", "gpm_id", *ds_vars])

    # Apply function to each swath chunk in parallel
    list_slices = generate_chunk_slices(chunks, shape=ds["along_track"].shape)
    list_df = []
    for slc in list_slices:
        df = _reshape_ds_to_df_overpass(ds.isel(along_track=slc))
        list_df.append(df)
    list_df = dask.compute(list_df)
    # Concat dataframes
    df = pd.concat(*list_df)
    return df


# -----------------------------------------------------------------------------.
os.makedirs(OVERPASS_TABLE_DIR, exist_ok=True)

for product in products:
    print(product)
    try:
        ds = GPM_Dataset(
            base_DIR=BASE_DIR,
            product=product,
            variables=["surfacePrecipitation"],
            start_time=start_time,
            end_time=end_time,
            scan_mode=None,
            GPM_version=6,
            product_type=product_type,
            bbox=None,
            enable_dask=False,
            chunks="auto",
        )
        # Monitor progress
        with ProgressBar():
            df = generate_df_overpass_time(ds, chunks=1000)

        df["satellite"] = product
        # Save to disk
        fpath = os.path.join(OVERPASS_TABLE_DIR, product + ".parquet")
        df.to_parquet(fpath)

    except ValueError:
        continue

# -----------------------------------------------------------------------------.
