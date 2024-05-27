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
"""This module contains general utility to convert xarray objects to dataframes."""
import dask

from gpm.dataset.granule import remove_unused_var_dims
from gpm.utils.xarray import ensure_unique_chunking


def get_df_object_columns(df):
    """Get the dataframe columns which have 'object' type."""
    return list(df.select_dtypes(include="object").columns)


def ensure_pyarrow_string_columns(df):
    """Convert 'object' type columns to pyarrow strings."""
    for column in get_df_object_columns(df):
        df[column] = df[column].astype("string[pyarrow]")
    return df


def drop_undesired_columns(df):
    """Drop undesired columns like dataset dimensions without coordinates."""
    undesired_columns = ["cross_track", "along_track", "crsWGS84"]
    undesired_columns = [column for column in undesired_columns if column in df.columns]
    return df.drop(columns=undesired_columns)


def to_pandas_dataframe(ds):
    """Convert an `xarray.Dataset` to a `pandas.DataFrame`."""
    # Drop unrelevant coordinates
    ds = remove_unused_var_dims(ds)

    # Convert to pandas dataframe
    # - strings are converted to object !
    df = ds.to_dataframe(dim_order=None)

    # Convert object columns to pyarrow string
    df = ensure_pyarrow_string_columns(df)

    # Remove MultiIndex
    df = df.reset_index()

    # Drop unrequired columns (previous dataset dimensions)
    return drop_undesired_columns(df)


def to_dask_dataframe(ds):
    """Convert an `xarray.Dataset` to a `dask.Dataframe`."""
    # Drop unrelevant coordinates
    ds = remove_unused_var_dims(ds)

    # Check dataset uniform chunking
    ds = ensure_unique_chunking(ds)

    # Convert to to dask dataframe
    # - strings are converted to object !
    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        df = ds.to_dask_dataframe(dim_order=None, set_index=False)

    # Convert object columns to pyarrow string
    df = ensure_pyarrow_string_columns(df)

    # Drop unrequired columns (previous dataset dimensions)
    return drop_undesired_columns(df)
