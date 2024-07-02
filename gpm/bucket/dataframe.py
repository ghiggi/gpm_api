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
"""This module implements manipulation wrappers for multiple DataFrame classes."""
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa

pd.options.mode.copy_on_write = True


def check_valid_dataframe(df):
    """Check the dataframe class."""
    valid_types = (pl.DataFrame, pl.LazyFrame, pa.Table, dd.DataFrame, pd.DataFrame)
    if not isinstance(df, valid_types):
        class_name = repr(df.__class__)
        raise TypeError(f"Dataframe operations not yet implemented for {class_name}")


def df_is_column_in(df, column):
    if isinstance(df, pa.Table):
        return column in df.column_names
    return column in df


def df_get_column(df, column):
    """Get the dataframe column."""
    if isinstance(df, pl.LazyFrame):
        return df.select(column).collect()[column]
    return df[column]


def df_select_valid_rows(df, valid_rows):
    """Select only dataframe rows with valid rows (using boolean array)."""
    if isinstance(df, (pl.DataFrame, pl.LazyFrame, pa.Table)):
        return df.filter(valid_rows)
    if isinstance(df, dd.DataFrame):
        return df.loc[np.where(valid_rows)[0]]  # BUG when providing boolean when npartitions>1
    # else: #  if isinstance(df, pd.DataFrame):
    return df.loc[valid_rows]


def df_add_column(df, column, values):
    """Add column to dataframe."""
    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return df.with_columns(pl.Series(column, values))
    if isinstance(df, pd.DataFrame):
        # Use assign to not modify input df
        # Do not use pd.Series(values) because mess up if df has Index/Multindex
        return df.assign(**{column: values})
    if isinstance(df, dd.DataFrame):
        # 'df[column] = pd.Series(values)' conserve npartitions
        # 'df[column] = pd.Series(values)' does not work if npartitions=1
        # BUG: THIS DOES NOT WORK IF DF HAS A MULTINDEX !
        if df.npartitions > 1:
            df[column] = pd.Series(values)
        else:  # npartitions=1
            df[column] = dask.array.from_array(values)  # does not conserve npartition
        return df
    # else: # pyarrow.Table
    return df.append_column(column, pa.array(values))


def df_to_pandas(df):
    """Convert dataframe to pandas."""
    if isinstance(df, pd.DataFrame):
        return df
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pl.LazyFrame):
        return df.collect().to_pandas()
    if isinstance(df, dd.DataFrame):
        return df.compute()
    # else: if isinstance(df, pa.Table):
    return df.to_pandas()
