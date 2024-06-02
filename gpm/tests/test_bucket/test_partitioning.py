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
"""This module tests the Spatial Partitioning classes."""
import os

import dask.dataframe as dd
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
import xarray as xr

from gpm.bucket.partitioning import (
    LonLatPartitioning,
    TilePartitioning,
    XYPartitioning,
    get_array_combinations,
    get_bounds,
    get_n_decimals,
)


def test_get_n_decimals():
    """Ensure decimal count is accurate."""
    assert get_n_decimals(123.456) == 3
    assert get_n_decimals(100) == 0
    assert get_n_decimals(123.0001) == 4


def test_get_bounds():
    """Verify the correct calculation of bounds."""
    bounds = get_bounds(0.5, 0, 2)
    assert np.array_equal(bounds, np.array([0, 0.5, 1.0, 1.5, 2]))


def test_get_array_combinations():
    x = np.array([1, 2, 3])
    y = np.array([4, 5])
    x_out, y_out = get_array_combinations(x, y)
    np.testing.assert_allclose(x_out, [1, 2, 3, 1, 2, 3])
    np.testing.assert_allclose(y_out, [4, 4, 4, 5, 5, 5])


class TestXYPartitioning:
    """Tests for the XYPartitioning class."""

    def test_initialization(self):
        """Test proper initialization of XYPartitioning objects."""
        partitioning = XYPartitioning(size=(1, 2), extent=[0, 10, 0, 10])
        assert partitioning.size == (1, 2)
        assert partitioning.order == ["xbin", "ybin"]  # default levels
        assert list(partitioning.extent) == [0, 10, 0, 10]
        assert partitioning.shape == (5, 10)
        assert partitioning.n_partitions == 50
        assert partitioning.n_x == 10
        assert partitioning.n_y == 5
        np.testing.assert_allclose(partitioning.x_bounds, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        np.testing.assert_allclose(partitioning.y_bounds, [0, 2, 4, 6, 8, 10])
        np.testing.assert_allclose(partitioning.x_centroids, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
        np.testing.assert_allclose(partitioning.y_centroids, [1.0, 3.0, 5.0, 7.0, 9.0])
        assert partitioning.x_labels.tolist() == ["0.5", "1.5", "2.5", "3.5", "4.5", "5.5", "6.5", "7.5", "8.5", "9.5"]
        assert partitioning.y_labels.tolist() == ["1.0", "3.0", "5.0", "7.0", "9.0"]

    def test_invalid_initialization(self):
        """Test initialization with invalid extent and size."""
        # Invalid names types
        with pytest.raises(TypeError):
            XYPartitioning(size=(0.1, 0.2), extent=[0, 10, 0, 10], levels=0)

        # Invalid extent
        with pytest.raises(ValueError):
            XYPartitioning(size=(0.1, 0.2), extent=[10, 0, 0, 10])
        # Invalid size
        with pytest.raises(TypeError):
            XYPartitioning(size="invalid", extent=[0, 10, 0, 10])

        # Mismatch partitions_order and names types
        with pytest.raises(ValueError):
            XYPartitioning(
                size=(0.1, 0.2),
                extent=[0, 10, 0, 10],
                levels=["x", "y"],
                order=["y", "another_name_instead_of_x"],
            )
        # Invalid names types
        with pytest.raises(ValueError):
            XYPartitioning(size=(0.1, 0.2), extent=[0, 10, 0, 10], flavor="bad_one")

    def test_labels(self):
        """Test labels property (origin="top")."""
        partitioning = XYPartitioning(size=(1, 2), extent=[0, 10, 0, 10])
        assert partitioning.labels.shape == (5, 10, 2)
        assert partitioning.labels[0, 0, :].tolist() == ["0.5", "1.0"]
        assert partitioning.labels[-1, 0, :].tolist() == ["0.5", "9.0"]
        assert partitioning.labels[0, -1, :].tolist() == ["9.5", "1.0"]
        assert partitioning.labels[-1, -1, :].tolist() == ["9.5", "9.0"]

    def test_centroids(self):
        """Test centroids property."""
        size = (120, 90)  # 3x2 partitions
        extent = [-180, 180, -90, 90]
        partitioning = XYPartitioning(
            size=size,
            extent=extent,
        )
        centroids = partitioning.centroids
        # Test results
        assert centroids.shape == (2, 3, 2)
        x_centroids = np.array([[-120.0, 0.0, 120.0], [-120.0, 0.0, 120.0]])
        y_centroids = np.array([[-45.0, -45.0, -45.0], [45.0, 45.0, 45.0]])
        np.testing.assert_allclose(centroids[:, :, 0], x_centroids)
        np.testing.assert_allclose(centroids[:, :, 1], y_centroids)

    def test_bounds(self):
        """Test bounds property."""
        # Create partitioning
        size = (25, 20)
        extent = [0, 50, 60, 80]
        partitioning = XYPartitioning(size=size, extent=extent)
        # Test bounds
        x_bounds, y_bounds = partitioning.bounds
        np.testing.assert_allclose(x_bounds, [0, 25, 50])
        np.testing.assert_allclose(y_bounds, [60, 80])

    def test_add_labels_pandas(self):
        """Test valid partitions are added to a pandas dataframe."""
        # Create test dataframe
        df = pd.DataFrame(
            {
                "x": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
                "y": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
            },
        )
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        levels = ["partition_name_x", "partition_name_y"]
        partitioning = XYPartitioning(size=size, extent=extent, levels=levels)

        # Add partitions
        df_out = partitioning.add_labels(df, x="x", y="y", remove_invalid_rows=True)

        # Test results
        assert isinstance(df_out, pd.DataFrame)
        # assert df_out["partition_name_x"].dtype.name == "category", "X bin are not of categorical type."
        # assert df_out["partition_name_y"].dtype.name == "category", "Y bin are not of categorical type."

        expected_x_labels = ["0.25", "0.25", "0.25", "0.75", "1.25", "1.75"]
        expected_y_labels = ["0.125", "0.125", "0.375", "0.875", "1.375", "1.875"]
        assert df_out["partition_name_x"].astype(str).tolist() == expected_x_labels, "X bin are incorrect."
        assert df_out["partition_name_y"].astype(str).tolist() == expected_y_labels, "Y bin are incorrect."

    def test_add_labels_polars(self):
        """Test valid partitions are added to a polars dataframe."""
        # Create test dataframe
        df = pl.DataFrame(
            pd.DataFrame(
                {
                    "x": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
                    "y": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
                },
            ),
        )
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        levels = ["partition_name_x", "partition_name_y"]
        partitioning = XYPartitioning(size=size, extent=extent, levels=levels)

        # Add partitions
        df_out = partitioning.add_labels(df, x="x", y="y", remove_invalid_rows=True)

        # Test results
        assert isinstance(df_out, pl.DataFrame)
        # assert df_out["partition_name_x"].dtype == pl.datatypes.Categorical, "X bin are not of categorical type."
        # assert df_out["partition_name_y"].dtype == pl.datatypes.Categorical, "X bin are not of categorical type."

        expected_x_labels = ["0.25", "0.25", "0.25", "0.75", "1.25", "1.75"]
        expected_y_labels = ["0.125", "0.125", "0.375", "0.875", "1.375", "1.875"]
        assert df_out["partition_name_x"].cast(str).to_list() == expected_x_labels, "X bin are incorrect."
        assert df_out["partition_name_y"].cast(str).to_list() == expected_y_labels, "Y bin are incorrect."

    def test_add_labels_polars_lazy(self):
        """Test valid partitions are added to a polars lazy dataframe."""
        # Create test dataframe
        df = pl.DataFrame(
            pd.DataFrame(
                {
                    "x": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
                    "y": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
                },
            ),
        ).lazy()
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        levels = ["partition_name_x", "partition_name_y"]
        partitioning = XYPartitioning(size=size, extent=extent, levels=levels)

        # Add partitions
        df_out = partitioning.add_labels(df, x="x", y="y", remove_invalid_rows=True)

        # Test results
        assert isinstance(df_out, pl.LazyFrame)
        df_out = df_out.collect()
        # assert df_out["partition_name_x"].dtype == pl.datatypes.Categorical, "X bin are not of categorical type."
        # assert df_out["partition_name_y"].dtype == pl.datatypes.Categorical, "X bin are not of categorical type."

        expected_x_labels = ["0.25", "0.25", "0.25", "0.75", "1.25", "1.75"]
        expected_y_labels = ["0.125", "0.125", "0.375", "0.875", "1.375", "1.875"]
        assert df_out["partition_name_x"].cast(str).to_list() == expected_x_labels, "X bin are incorrect."
        assert df_out["partition_name_y"].cast(str).to_list() == expected_y_labels, "Y bin are incorrect."

    def test_add_dask_pandas(self):
        """Test valid partitions are added to a dask dataframe."""
        # Create test dataframe
        df = pd.DataFrame(
            {
                "x": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
                "y": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
            },
        )
        df = dd.from_pandas(df, npartitions=2)
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        levels = ["partition_name_x", "partition_name_y"]
        partitioning = XYPartitioning(size=size, extent=extent, levels=levels)

        # Add partitions
        df_out = partitioning.add_labels(df, x="x", y="y", remove_invalid_rows=True)

        # Test results
        assert isinstance(df_out, dd.DataFrame)
        # assert df_out["partition_name_x"].dtype.name == "category", "X bin are not of categorical type."
        # assert df_out["partition_name_y"].dtype.name == "category", "Y bin are not of categorical type."
        df_out = df_out.compute()

        expected_x_labels = ["0.25", "0.25", "0.25", "0.75", "1.25", "1.75"]
        expected_y_labels = ["0.125", "0.125", "0.375", "0.875", "1.375", "1.875"]
        assert df_out["partition_name_x"].astype(str).tolist() == expected_x_labels, "X bin are incorrect."
        assert df_out["partition_name_y"].astype(str).tolist() == expected_y_labels, "Y bin are incorrect."

    def test_add_labels_pyarrow(self):
        """Test valid partitions are added to a pyarrow dataframe."""
        # Create test dataframe
        df = pd.DataFrame(
            {
                "x": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
                "y": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
            },
        )
        df = pa.Table.from_pandas(df)
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        levels = ["partition_name_x", "partition_name_y"]
        partitioning = XYPartitioning(size=size, extent=extent, levels=levels)

        # Add partitions
        df_out = partitioning.add_labels(df, x="x", y="y", remove_invalid_rows=True)

        # Test results
        assert isinstance(df_out, pa.Table)
        # assert df_out["partition_name_x"].dtype.name == "category", "X bin are not of categorical type."
        # assert df_out["partition_name_y"].dtype.name == "category", "Y bin are not of categorical type."

        expected_x_labels = ["0.25", "0.25", "0.25", "0.75", "1.25", "1.75"]
        expected_y_labels_ = ["0.125", "0.125", "0.375", "0.875", "1.375", "1.875"]
        assert df_out["partition_name_x"].to_numpy().astype(str).tolist() == expected_x_labels, "X bin are incorrect."
        assert df_out["partition_name_y"].to_numpy().astype(str).tolist() == expected_y_labels_, "Y bin are incorrect."

    def test_add_labels_invalid_args(self):
        """Test error is raised if invalid arguments are passed to add_labels."""
        df = pl.DataFrame(
            pd.DataFrame(
                {
                    "x": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
                    "y": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
                },
            ),
        )
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        levels = ["partition_name_x", "partition_name_y"]
        partitioning = XYPartitioning(size=size, extent=extent, levels=levels)

        # Test invalid arguments
        with pytest.raises(TypeError):
            partitioning.add_labels(df="invalid_type", x="x", y="y")

        with pytest.raises(ValueError):
            partitioning.add_labels(df=df, x="invalid_x", y="y")

        with pytest.raises(ValueError):
            partitioning.add_labels(df=df, x="x", y="invalid_y")

    @pytest.mark.parametrize("df_type", ["pandas", "polars"])
    def test_add_labels_invalid_values(self, df_type):
        """Test error is raised if invalid values are present in the dataframe."""
        # Create test pandas.DataFrame
        df_null_values = pd.DataFrame(
            {
                "x": [0, 0.5, 1.0, np.nan],
                "y": [0, 0.5, np.nan, 1.0],
            },
        )

        df_out_of_extent_values = pd.DataFrame(
            {
                "x": [-0.001, 0, 2.0, 2.1],
                "y": [-0.001, 0, 2.0, 2.1],
            },
        )

        # Convert to polars.DataFrame
        if df_type == "polars":
            df_out_of_extent_values = pl.DataFrame(df_out_of_extent_values)
            df_null_values = pl.DataFrame(df_null_values)

        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        partitioning = XYPartitioning(size=size, extent=extent)

        # Test error is raised if NaN are present in x or y column
        with pytest.raises(ValueError):
            partitioning.add_labels(df=df_null_values, x="x", y="y", remove_invalid_rows=False)

        # Test error is raised if out of extent values are present in x and y column
        with pytest.raises(ValueError):
            partitioning.add_labels(df=df_out_of_extent_values, x="x", y="y", remove_invalid_rows=False)

    def test_add_centroids_pandas(self):
        """Test valid partitions centroids are added to a pandas dataframe."""
        # Create test dataframe
        df = pd.DataFrame(
            {
                "x": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
                "y": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
            },
        )
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        levels = ["partition_name_x", "partition_name_y"]
        partitioning = XYPartitioning(size=size, extent=extent, levels=levels)

        # Add partitions
        df_out = partitioning.add_centroids(df, x="x", y="y", remove_invalid_rows=True)

        # Test results
        assert isinstance(df_out, pd.DataFrame)
        expected_x_labels = [0.25, 0.25, 0.25, 0.75, 1.25, 1.75]
        expected_y_labels = [0.125, 0.125, 0.375, 0.875, 1.375, 1.875]
        assert df_out["x_c"].dtype.name == "float64", "X Centroids are not of float64 type."
        assert df_out["y_c"].dtype.name == "float64", "Y Centroids are not of float64 type."
        assert df_out["x_c"].tolist() == expected_x_labels, "X Centroids are incorrect."
        assert df_out["y_c"].tolist() == expected_y_labels, "Y Centroids are incorrect."

    def test_add_centroids_invalid_values(self):
        """Test error is raised if invalid values are present in the dataframe."""
        # Create test pandas.DataFrame
        df_null_values = pd.DataFrame(
            {
                "x": [0, 0.5, 1.0, np.nan],
                "y": [0, 0.5, np.nan, 1.0],
            },
        )

        df_out_of_extent_values = pd.DataFrame(
            {
                "x": [-0.001, 0, 2.0, 2.1],
                "y": [-0.001, 0, 2.0, 2.1],
            },
        )

        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        partitioning = XYPartitioning(size=size, extent=extent)

        # Test error is raised if NaN are present in x or y column
        with pytest.raises(ValueError):
            partitioning.add_centroids(df=df_null_values, x="x", y="y", remove_invalid_rows=False)

        # Test error is raised if out of extent values are present in x and y column
        with pytest.raises(ValueError):
            partitioning.add_centroids(df=df_out_of_extent_values, x="x", y="y", remove_invalid_rows=False)

    @pytest.mark.parametrize("df_type", ["pandas", "polars"])
    def test_to_xarray(self, df_type):
        """Test valid partitions are added to a pandas dataframe."""
        # Create test pandas.DataFrame
        df = pd.DataFrame(
            {
                "x": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
                "y": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
            },
        )
        # Convert to polars.DataFrame
        if df_type == "polars":
            df = pl.DataFrame(df)

        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        levels = ["partition_name_x", "partition_name_y"]
        partitioning = XYPartitioning(size=size, extent=extent, levels=levels)

        # Add partitions
        df = partitioning.add_labels(df, x="x", y="y", remove_invalid_rows=True)

        # Aggregate by partitions
        if df_type == "polars":
            df_grouped = df.group_by(partitioning.order).median()
            df_grouped = df_grouped.with_columns(pl.lit(2).alias("dummy_var"))
        else:  # pandas or dask
            df_grouped = df.groupby(partitioning.order, observed=True).median()
            df_grouped["dummy_var"] = 2

        # Convert to Dataset
        df_grouped = partitioning.add_centroids(df_grouped, x="x", y="y")
        ds = partitioning.to_xarray(df_grouped)

        # Test results
        expected_x_centroids = [0.25, 0.75, 1.25, 1.75]
        expected_y_centroids = [0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875]
        assert isinstance(ds, xr.Dataset), "Not a xr.Dataset"
        assert "dummy_var" in ds, "The x columns has not become a xr.Dataset variable"
        assert ds["x_c"].data.dtype.name != "object", "xr.Dataset coordinates should not be a string."
        assert ds["y_c"].data.dtype.name != "object", "xr.Dataset coordinates should not be a string."
        assert ds["x_c"].data.dtype.name == "float64", "xr.Dataset coordinates are not float64."
        assert ds["y_c"].data.dtype.name == "float64", "xr.Dataset coordinates are not float64."
        np.testing.assert_allclose(ds["x_c"].data, expected_x_centroids)
        np.testing.assert_allclose(ds["y_c"].data, expected_y_centroids)
        assert ds["dummy_var"].data[0, 0] == 2

    @pytest.mark.parametrize("index_type", ["set", "unset"])
    def test_to_xarray_multindex(self, index_type):
        """Test valid partitions are added to a pandas dataframe."""
        # Create test pandas.DataFrame
        df = pd.DataFrame(
            {
                "x": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
                "y": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
            },
        )
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        levels = ["partition_name_x", "partition_name_y"]
        partitioning = XYPartitioning(size=size, extent=extent, levels=levels)

        # Add partitions
        df = partitioning.add_labels(df, x="x", y="y", remove_invalid_rows=True)

        # Group over partitions
        df_grouped = df.groupby(partitioning.order, observed=True).median()
        df_grouped["dummy_var"] = 2

        # Create df with additional index (i.e. time)
        df_grouped1 = df_grouped.copy()
        df_grouped2 = df_grouped.copy()
        df_grouped1["frequency"] = "low"
        df_grouped2["frequency"] = "high"
        df_grouped1["month"] = 1
        df_grouped2["month"] = 2

        df_full = pd.concat((df_grouped1, df_grouped2))

        # Test categorical dtype is converted !
        df_full["frequency"] = pd.Categorical(df_full["frequency"])

        # Test
        df_full = partitioning.add_centroids(df_full, x="x", y="y")

        # Convert to Dataset
        if index_type == "set":
            df_full = df_full.reset_index()
            df_full = df_full.set_index([*levels, "frequency", "month"])
            aux_coords = None
        else:
            aux_coords = ["frequency", "month"]
        ds = partitioning.to_xarray(df_full, aux_coords=aux_coords)

        # Test results
        assert isinstance(ds, xr.Dataset), "Not a xr.Dataset"
        assert "dummy_var" in ds, "The x columns has not become a xr.Dataset variable."
        assert "frequency" in ds.coords, "'frequency' is not a xr.Dataset coordinate."
        assert "month" in ds.coords, "'month' is not a xr.Dataset coordinate."
        assert "partition_label_x" not in ds.coords, "The x partition label has been set as a xr.Dataset coordinate."
        assert "partition_label_y" not in ds.coords, "The y partition label has been set as a xr.Dataset coordinate."
        assert ds["x_c"].data.dtype.name == "float64", "xr.Dataset 'lon' coordinate is not float64."
        assert ds["y_c"].data.dtype.name == "float64", "xr.Dataset 'lat' coordinate is not float64."
        assert ds["frequency"].data.dtype.name == "object", "xr.Dataset 'frequency' coordinate is not an object."
        assert ds["month"].data.dtype.name == "int64", "xr.Dataset 'month' coordinate is not int64."

        da = ds["dummy_var"].isel(frequency=1, month=0, x_c=slice(0, 2), y_c=slice(0, 2))
        expected_arr = np.array([[2.0, 2.0], [np.nan, np.nan]])
        np.testing.assert_allclose(da.data, expected_arr)

    def test_to_xarray_invalid_args(self):
        """Test invalid arguments of to_xarray() method."""
        # Create test pandas.DataFrame
        df = pd.DataFrame(
            {
                "x": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
                "y": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
            },
        )
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        levels = ["partition_name_x", "partition_name_y"]
        partitioning = XYPartitioning(size=size, extent=extent, levels=levels)
        # Add partitions
        df = partitioning.add_labels(df, x="x", y="y", remove_invalid_rows=True)
        # Aggregate by partitions
        df_grouped = df.groupby(partitioning.order, observed=True).median()
        df_grouped["dummy_var"] = 2
        # Test raise error because no centroids
        with pytest.raises(ValueError):
            partitioning.to_xarray(df_grouped)
        # Convert to Dataset
        df_grouped = partitioning.add_centroids(df_grouped, x="x", y="y", x_coord="x_dummy", y_coord="y_dummy")
        # Test raise error because spatial_coords not specified (and defaults are not there)
        with pytest.raises(ValueError):
            partitioning.to_xarray(df_grouped)
        # Test raise error because spatial_coords are mispecified
        with pytest.raises(ValueError):
            partitioning.to_xarray(df_grouped, spatial_coords=["bad_x", "bad_y"])
        with pytest.raises(ValueError):
            partitioning.to_xarray(df_grouped, spatial_coords="dummy")
        # Test raise error because invalid aux_coords
        with pytest.raises(ValueError):
            partitioning.to_xarray(df_grouped, spatial_coords=["x_dummy", "y_dummy"], aux_coords=["bad"])

    def test_query_labels(self):
        """Test valid labels queries."""
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        partitioning = XYPartitioning(size=size, extent=extent)
        # Test with various input type
        # - float and 0D np.array
        x_labels, y_labels = partitioning.query_labels(1, np.array(1))
        assert x_labels.tolist() == ["0.75"]
        assert y_labels.tolist() == ["0.875"]
        # - list or 1D np.array
        x_labels, y_labels = partitioning.query_labels([1, 2], np.array([0, 1]))
        assert x_labels.tolist() == ["0.75", "1.75"]
        assert y_labels.tolist() == ["0.125", "0.875"]
        # - 2D np.array
        arr = np.ones((2, 2))
        x_labels, y_labels = partitioning.query_labels(arr, arr)
        assert x_labels.shape == (2, 2)
        assert x_labels.flatten().tolist() == ["0.75"] * 4
        assert y_labels.flatten().tolist() == ["0.875"] * 4

        # Test out of extent it returns "nan"
        x_labels, y_labels = partitioning.query_labels(-1, 1)
        assert x_labels.tolist() == ["nan"]
        assert y_labels.tolist() == ["nan"]  # also the other (valid) value is set to NaN !

        # Test with input None
        x_labels, y_labels = partitioning.query_labels(None, 1)
        assert x_labels.tolist() == ["nan"]
        assert y_labels.tolist() == ["nan"]  # also the other (valid) value is set to NaN !

        # Test with input nan
        x_labels, y_labels = partitioning.query_labels(np.nan, 1)
        assert x_labels.tolist() == ["nan"]
        assert y_labels.tolist() == ["nan"]  # also the other (valid) value is set to NaN !

        # Test with input string
        with pytest.raises(ValueError):
            partitioning.query_labels("dummy", "dummy")

    def test_query_centroids(self):
        """Test valid midpoint queries."""
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        partitioning = XYPartitioning(size=size, extent=extent)

        # Test with various input type
        # - float and 0D np.array
        x_centroids, y_centroids = partitioning.query_centroids(1, np.array(1))
        assert x_centroids.tolist() == [0.75]
        assert y_centroids.tolist() == [0.875]
        # - list or 1D np.array
        x_centroids, y_centroids = partitioning.query_centroids([1, 2], np.array([0, 1]))
        assert x_centroids.tolist() == [0.75, 1.75]
        assert y_centroids.tolist() == [0.125, 0.875]
        # - 2D np.array
        arr = np.ones((2, 2))
        x_centroids, y_centroids = partitioning.query_centroids(arr, arr)
        assert x_centroids.shape == (2, 2)
        assert x_centroids.flatten().tolist() == [0.75] * 4
        assert y_centroids.flatten().tolist() == [0.875] * 4

        # Test out of extent it returns np.nan
        x_centroids, y_centroids = partitioning.query_centroids(-1, 1)
        np.testing.assert_allclose(x_centroids, [np.nan])
        np.testing.assert_allclose(y_centroids, [np.nan])  # also the other (valid) value is set to NaN !

        # Test with input nan
        x_centroids, y_centroids = partitioning.query_centroids(np.nan, 1)
        np.testing.assert_allclose(x_centroids, [np.nan])
        np.testing.assert_allclose(y_centroids, [np.nan])  # also the other (valid) value is set to NaN !

        # Test with input None
        x_centroids, y_centroids = partitioning.query_centroids(None, 1)
        np.testing.assert_allclose(x_centroids, [np.nan])
        np.testing.assert_allclose(y_centroids, [np.nan])  # also the other (valid) value is set to NaN !

        # Test with input string
        with pytest.raises(ValueError):
            partitioning.query_centroids("dummy", "dummy")

    def test_get_partitions_by_extent(self):
        """Test get_partitions_by_extent."""
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        x_name = "partition_name_x"
        y_name = "partition_name_y"
        levels = [x_name, y_name]
        partitioning = XYPartitioning(size=size, extent=extent, levels=levels)
        # Test results with extent within
        new_extent = [0, 0.5, 0, 0.5]
        dict_labels = partitioning.get_partitions_by_extent(new_extent)
        assert dict_labels[x_name].tolist() == ["0.25", "0.25"]
        assert dict_labels[y_name].tolist() == ["0.125", "0.375"]

        # Test results with extent outside
        new_extent = [3, 4, 3, 4]
        dict_labels = partitioning.get_partitions_by_extent(new_extent)
        assert dict_labels[x_name].size == 0
        assert dict_labels[y_name].size == 0

        # Test results with extent partially overlapping
        new_extent = [1.5, 4, 1.75, 4]
        dict_labels = partitioning.get_partitions_by_extent(new_extent)
        assert dict_labels[x_name].tolist() == ["1.25", "1.75", "1.25", "1.75"]
        assert dict_labels[y_name].tolist() == ["1.625", "1.625", "1.875", "1.875"]

    def test_get_partitions_around_point(self):
        """Test get_partitions_around_point."""
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        x_name = "partition_name_x"
        y_name = "partition_name_y"
        levels = [x_name, y_name]
        partitioning = XYPartitioning(size=size, extent=extent, levels=levels)
        # Test results with point within
        dict_labels = partitioning.get_partitions_around_point(x=1, y=1, distance=0)
        assert dict_labels[x_name].tolist() == ["0.75"]
        assert dict_labels[y_name].tolist() == ["0.875"]

        # Test results with point outside
        dict_labels = partitioning.get_partitions_around_point(x=3, y=3, distance=0)
        assert dict_labels[x_name].size == 0
        assert dict_labels[y_name].size == 0

        # Test results with point outside but area within
        dict_labels = partitioning.get_partitions_around_point(x=3, y=3, distance=1)
        assert dict_labels[x_name].tolist() == ["1.75"]
        assert dict_labels[y_name].tolist() == ["1.875"]

    def test_directories_around_point(self):
        """Test directories_around_point."""
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        partitioning = XYPartitioning(size=size, extent=extent)

        # Test results with point and aoi outside extent
        directories = partitioning.directories_around_point(x=3, y=3, distance=0)
        assert directories.size == 0

        # Test results with point outside but partial aoi inside extent
        directories = partitioning.directories_around_point(x=3, y=3, distance=1)
        assert directories.tolist() == [f"1.75{os.sep}1.875"]

    def test_quadmesh(self):
        """Test quadmesh."""
        size = (1, 1)
        extent = [0, 2, 1, 3]
        partitioning = XYPartitioning(size=size, extent=extent)
        # Test shape
        assert partitioning.quadmesh(origin="bottom").shape == (3, 3, 2)
        assert partitioning.quadmesh(origin="top").shape == (3, 3, 2)
        # Test values
        x_mesh = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        y_mesh = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        quadmesh = partitioning.quadmesh()  # origin="bottom"
        quadmesh_top = partitioning.quadmesh(origin="top")
        np.testing.assert_allclose(quadmesh_top[:, :, 0], x_mesh)
        np.testing.assert_allclose(quadmesh_top[:, :, 1], y_mesh)
        np.testing.assert_allclose(quadmesh[:, :, 0], x_mesh)
        np.testing.assert_allclose(quadmesh[::-1, :, 1], y_mesh)

    def test_to_dict(self):
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        levels = ["name1", "name2"]
        partitioning = XYPartitioning(size=size, extent=extent, levels=levels, order=levels[::-1])
        # Test results
        expected_dict = {
            "partitioning_class": "XYPartitioning",
            "extent": list(extent),
            "size": list(size),
            "levels": levels,
            "order": levels[::-1],
            "flavor": "directory",  # default
            "labels_decimals": [2, 3],
        }
        assert partitioning.to_dict() == expected_dict


class TestLonLatPartitioning:
    """Tests for the LonLatPartitioning class."""

    def test_get_partitions_by_extent(self):
        """Test get_partitions_by_extent."""
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        partitioning = LonLatPartitioning(size=size, extent=extent)
        # Test results with extent within
        new_extent = [0, 0.5, 0, 0.5]
        dict_labels = partitioning.get_partitions_by_extent(new_extent)
        assert dict_labels["lon_bin"].tolist() == ["0.25", "0.25"]
        assert dict_labels["lat_bin"].tolist() == ["0.125", "0.375"]

        # Test results with extent outside
        new_extent = [3, 4, 3, 4]
        dict_labels = partitioning.get_partitions_by_extent(new_extent)
        assert dict_labels["lon_bin"].size == 0
        assert dict_labels["lat_bin"].size == 0

        # Test results with extent partially overlapping
        new_extent = [1.5, 4, 1.75, 4]
        dict_labels = partitioning.get_partitions_by_extent(new_extent)
        assert dict_labels["lon_bin"].tolist() == ["1.25", "1.75", "1.25", "1.75"]
        assert dict_labels["lat_bin"].tolist() == ["1.625", "1.625", "1.875", "1.875"]

    def test_get_partitions_around_point(self):
        """Test get_partitions_around_point."""
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        partitioning = LonLatPartitioning(size=size, extent=extent)
        # Test results with point within
        dict_labels = partitioning.get_partitions_around_point(lon=1, lat=1, distance=0)
        assert dict_labels["lon_bin"].tolist() == ["0.75"]
        assert dict_labels["lat_bin"].tolist() == ["0.875"]

        # Test results with point outside
        dict_labels = partitioning.get_partitions_around_point(lon=1, lat=3, distance=0)
        assert dict_labels["lon_bin"].size == 0
        assert dict_labels["lat_bin"].size == 0

        # Test results with point and aoi outside
        dict_labels = partitioning.get_partitions_around_point(lon=3, lat=3, distance=150_000)  # 150 km
        assert dict_labels["lon_bin"].tolist() == ["1.75", "1.75"]
        assert dict_labels["lat_bin"].tolist() == ["1.625", "1.875"]

    def test_get_partitions_by_country(self):
        """Test get_partitions_by_country."""
        # Create partitioning
        size = (60, 60)
        extent = [-180, 180, -30, 30]
        partitioning = LonLatPartitioning(size=size, extent=extent)
        # Test country within extent
        dict_labels = partitioning.get_partitions_by_country("Nigeria")
        assert dict_labels["lon_bin"].tolist() == ["30.0"]
        assert dict_labels["lat_bin"].tolist() == ["0.0"]

        # Test country outside extent
        dict_labels = partitioning.get_partitions_by_country("Switzerland")
        assert dict_labels["lon_bin"].size == 0
        assert dict_labels["lat_bin"].size == 0

    def test_get_partitions_by_continent(self):
        """Test get_partitions_by_continent."""
        # Create partitioning
        size = (60, 60)
        extent = [-180, 180, -30, 30]
        partitioning = LonLatPartitioning(size=size, extent=extent)

        # Test continent within extent
        dict_labels = partitioning.get_partitions_by_continent("Africa")
        assert dict_labels["lon_bin"].tolist() == ["-30.0", "30.0"]
        assert dict_labels["lat_bin"].tolist() == ["0.0", "0.0"]

        # Test continent outside extent
        dict_labels = partitioning.get_partitions_by_continent("Europe")
        assert dict_labels["lon_bin"].size == 0
        assert dict_labels["lat_bin"].size == 0

    def test_directories_by_country(self):
        """Test directories_by_country."""
        # Create partitioning
        size = (60, 60)
        extent = [-180, 180, -30, 30]
        partitioning = LonLatPartitioning(size=size, extent=extent)
        # Test country within extent
        directories = partitioning.directories_by_country("Nigeria")
        assert directories.tolist() == [f"lon_bin=30.0{os.sep}lat_bin=0.0"]

        # Test country outside extent
        directories = partitioning.directories_by_country("Switzerland")
        assert directories.size == 0

    def test_directories_by_continent(self):
        """Test directories_by_continent."""
        # Create partitioning
        size = (60, 60)
        extent = [-180, 180, -30, 30]
        partitioning = LonLatPartitioning(size=size, extent=extent)
        # Test continent within directories_by_continent
        directories = partitioning.directories_by_continent("Africa")
        assert directories.tolist() == [f"lon_bin=-30.0{os.sep}lat_bin=0.0", f"lon_bin=30.0{os.sep}lat_bin=0.0"]

        # Test continent outside extent
        directories = partitioning.directories_by_continent("Europe")
        assert directories.size == 0

    def test_directories_around_point(self):
        """Test directories_around_point."""
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        partitioning = LonLatPartitioning(size=size, extent=extent)

        # Test results with point and aoi outside
        directories = partitioning.directories_around_point(lon=3, lat=3, distance=0)
        assert directories.size == 0

        # Test results with point outside extent but aoi inside
        directories = partitioning.directories_around_point(lon=3, lat=3, distance=150_000)  # 150 km
        assert directories.tolist() == [f"lon_bin=1.75{os.sep}lat_bin=1.625", f"lon_bin=1.75{os.sep}lat_bin=1.875"]

    def test_quadmesh(self):
        # Create partitioning
        size = (20, 10)
        extent = [0, 60, 60, 80]
        partitioning = LonLatPartitioning(size=size, extent=extent)
        # Test quadmesh array (M+1, N+1, 2)
        quadmesh = partitioning.quadmesh()
        assert quadmesh.shape == (3, 4, 2)
        x_mesh = np.array([[0, 20, 40, 60], [0, 20, 40, 60], [0, 20, 40, 60]])
        y_mesh = np.array([[80, 80, 80, 80], [70, 70, 70, 70], [60, 60, 60, 60]])
        np.testing.assert_allclose(quadmesh[:, :, 0], x_mesh)
        np.testing.assert_allclose(quadmesh[:, :, 1], y_mesh)

    def test_vertices_origin_bottom(self):
        # Create partitioning
        size = (20, 10)
        extent = [0, 60, 60, 80]
        partitioning = LonLatPartitioning(size=size, extent=extent)
        # Test vertices array (M, N, 4, 2)
        vertices = partitioning.vertices(ccw=True)  # origin="bottom"
        assert vertices.shape == (2, 3, 4, 2)
        # Test bottom left cell (geographic top left)
        expected_vertices = np.array([[0, 80], [0, 70], [20, 70], [20, 80]])
        np.testing.assert_allclose(vertices[0, 0, :, :], expected_vertices)
        # Test clockwise order
        vertices = partitioning.vertices(ccw=False)
        assert vertices.shape == (2, 3, 4, 2)
        # Test bottom left cell (geographic top left)
        expected_vertices = np.array([[0, 80], [20, 80], [20, 70], [0, 70]])
        np.testing.assert_allclose(vertices[0, 0, :, :], expected_vertices)

    def test_vertices_origin_top(self):
        # Create partitioning
        size = (20, 10)
        extent = [0, 60, 60, 80]
        partitioning = LonLatPartitioning(size=size, extent=extent)
        # Test vertices array (M, N, 4, 2)
        vertices = partitioning.vertices(origin="top", ccw=True)
        assert vertices.shape == (2, 3, 4, 2)
        # Test bottom left cell (geographic top left)
        expected_vertices = np.array([[0, 70], [0, 60], [20, 60], [20, 70]])
        np.testing.assert_allclose(vertices[0, 0, :, :], expected_vertices)
        # Test clockwise order
        vertices = partitioning.vertices(origin="top", ccw=False)
        assert vertices.shape == (2, 3, 4, 2)
        # Test bottom left cell (geographic top left)
        expected_vertices = np.array([[20, 60], [0, 60], [0, 70], [20, 70]])
        np.testing.assert_allclose(vertices[0, 0, :, :], expected_vertices)

    def test_query_vertices_by_indices_flat(self):
        # Create partitioning
        size = (20, 10)
        extent = [0, 60, 60, 80]
        partitioning = LonLatPartitioning(size=size, extent=extent)
        # Test vertices array (M, N, 4, 2)
        vertices = partitioning.query_vertices_by_indices(x_indices=0, y_indices=[0], ccw=True)
        vertices1 = partitioning.query_vertices_by_indices(x_indices=0, y_indices=np.array(0), ccw=True)
        np.testing.assert_allclose(vertices, vertices1)
        assert vertices.shape == (1, 4, 2)
        expected_vertices = np.array([[0, 70], [0, 60], [20, 60], [20, 70]])
        np.testing.assert_allclose(vertices[0, :, :], expected_vertices)
        # Test clockwise
        vertices = partitioning.query_vertices_by_indices(x_indices=[0, 0], y_indices=[0, 0], ccw=False)
        assert vertices.shape == (2, 4, 2)
        expected_vertices = np.array([[0, 70], [20, 70], [20, 60], [0, 60]])
        np.testing.assert_allclose(vertices[0, :, :], expected_vertices)
        np.testing.assert_allclose(vertices[0, :, :], vertices[1, :, :])

    def test_query_vertices_by_indices_2d(self):
        # Create partitioning
        size = (20, 10)
        extent = [0, 60, 60, 80]
        partitioning = LonLatPartitioning(size=size, extent=extent)
        # Test vertices array with query_vertices_by_indices match vertices(origin="top") method
        x_indices, y_indices = np.meshgrid(np.arange(partitioning.n_x), np.arange(partitioning.n_y))
        vertices = partitioning.query_vertices_by_indices(x_indices=x_indices, y_indices=y_indices, ccw=True)
        assert vertices.shape == (2, 3, 4, 2)
        expected_vertices = np.array([[0, 70], [0, 60], [20, 60], [20, 70]])
        np.testing.assert_allclose(vertices[0, 0, :, :], expected_vertices)
        np.testing.assert_allclose(vertices, partitioning.vertices(origin="top"))

    def test_query_vertices(self):
        # Create partitioning
        size = (20, 10)
        extent = [0, 60, 60, 80]
        partitioning = LonLatPartitioning(size=size, extent=extent)
        # Test vertices array (M, N, 4, 2)
        vertices = partitioning.query_vertices(x=10, y=[65], ccw=True)
        vertices1 = partitioning.query_vertices(x=10, y=np.array(65), ccw=True)
        np.testing.assert_allclose(vertices, vertices1)
        assert vertices.shape == (1, 4, 2)
        expected_vertices = np.array([[0, 70], [0, 60], [20, 60], [20, 70]])
        np.testing.assert_allclose(vertices[0, :, :], expected_vertices)

    def test_query_vertices_2d(self):
        # Create partitioning
        size = (20, 10)
        extent = [0, 60, 60, 80]
        partitioning = LonLatPartitioning(size=size, extent=extent)
        # Test vertices array with query_vertices_by_indices match vertices(origin="top") method
        x_centroids, y_centroids = np.meshgrid(partitioning.x_centroids, partitioning.y_centroids)
        vertices = partitioning.query_vertices(x=x_centroids, y=y_centroids, ccw=True)
        assert vertices.shape == (2, 3, 4, 2)
        expected_vertices = np.array([[0, 70], [0, 60], [20, 60], [20, 70]])
        np.testing.assert_allclose(vertices[0, 0, :, :], expected_vertices)
        np.testing.assert_allclose(vertices, partitioning.vertices(origin="top"))


class TestTilePartitioning:
    """Tests for the TilePartitioning class."""

    def test_invalid_arguments(self):
        """Test invalid TilePartitioning arguments."""
        size = (120, 90)  # 3x2 partitions
        extent = [-180, 180, -90, 90]

        # Invalid levels
        with pytest.raises(ValueError):
            TilePartitioning(size=size, extent=extent, n_levels=3)

        # Invalid levels and names
        with pytest.raises(ValueError):
            TilePartitioning(size=size, extent=extent, n_levels=2, levels="one_name")
        with pytest.raises(ValueError):
            TilePartitioning(size=size, extent=extent, n_levels=1, levels=["one", "two"])

        # Invalid origin
        with pytest.raises(ValueError):
            TilePartitioning(size=size, extent=extent, n_levels=1, origin="bad")

        # Invalid direction
        with pytest.raises(ValueError):
            TilePartitioning(size=size, extent=extent, n_levels=1, direction="bad")

    def test_xy_bottom_partitioning(self):
        """Test two-levels 2D TilePartitioning à la GoogleMap (origin='bottom')."""
        size = (120, 90)  # 3x2 partitions
        extent = [-180, 180, -90, 90]
        n_levels = 2
        origin = "bottom"
        justify = False
        levels = None  # ["x","y"]
        partitioning = TilePartitioning(
            size=size,
            extent=extent,
            n_levels=n_levels,
            levels=levels,
            origin=origin,
            justify=justify,
        )
        # Check initialization
        assert partitioning.n_partitions == 6
        assert partitioning.shape == (2, 3)
        assert partitioning.n_levels == n_levels
        assert partitioning.levels == ["x", "y"]
        assert partitioning.order == ["x", "y"]

        # Check values for origin="bottom"
        x_labels, y_labels = partitioning.query_labels(-150, 90)
        assert y_labels.tolist() == ["0"]
        assert x_labels.tolist() == ["0"]

        x_labels, y_labels = partitioning.query_labels(150, 90)
        assert y_labels.tolist() == ["0"]
        assert x_labels.tolist() == ["2"]

        x_labels, y_labels = partitioning.query_labels(150, -90)
        assert y_labels.tolist() == ["1"]
        assert x_labels.tolist() == ["2"]

        # Test labels shape
        labels = partitioning.labels
        assert labels.shape == (2, 3, 2)
        expected_x_labels = np.array([["0", "1", "2"], ["0", "1", "2"]])
        expected_y_labels = np.array([["1", "1", "1"], ["0", "0", "0"]])
        assert labels[:, :, 0].tolist() == expected_x_labels.tolist()
        assert labels[:, :, 1].tolist() == expected_y_labels.tolist()

    def test_xy_top_partitioning(self):
        """Test two-levels TilePartitioning à la TMS (origin='top')."""
        size = (120, 90)  # 3x2 partitions
        extent = [-180, 180, -90, 90]
        n_levels = 2
        origin = "top"
        justify = False
        levels = None  # ["x","y"]
        partitioning = TilePartitioning(
            size=size,
            extent=extent,
            n_levels=n_levels,
            levels=levels,
            origin=origin,
            justify=justify,
        )
        # Check initialization
        assert partitioning.n_partitions == 6
        assert partitioning.shape == (2, 3)
        assert partitioning.n_levels == n_levels
        assert partitioning.order == ["x", "y"]

        # Check values for origin="bottom"
        x_labels, y_labels = partitioning.query_labels(-150, 90)
        assert y_labels.tolist() == ["1"]
        assert x_labels.tolist() == ["0"]

        x_labels, y_labels = partitioning.query_labels(150, 90)
        assert y_labels.tolist() == ["1"]
        assert x_labels.tolist() == ["2"]

        x_labels, y_labels = partitioning.query_labels(150, -90)
        assert y_labels.tolist() == ["0"]
        assert x_labels.tolist() == ["2"]

        # Test labels shape
        labels = partitioning.labels
        assert labels.shape == (2, 3, 2)
        expected_x_labels = np.array([["0", "1", "2"], ["0", "1", "2"]])
        expected_y_labels = np.array([["0", "0", "0"], ["1", "1", "1"]])
        assert labels[:, :, 0].tolist() == expected_x_labels.tolist()
        assert labels[:, :, 1].tolist() == expected_y_labels.tolist()

    @pytest.mark.parametrize("origin", ["bottom", "top"])
    def test_single_level_x_partitioning(self, origin):
        """Test single-level TilePartitioning id rows-by-rows."""
        size = (120, 90)  # 3x2 partitions
        extent = [-180, 180, -90, 90]
        n_levels = 1
        justify = False
        levels = None  # ["x","y"]
        partitioning = TilePartitioning(
            size=size,
            extent=extent,
            n_levels=n_levels,
            levels=levels,
            # direction="x", # default
            origin=origin,
            justify=justify,
        )
        # Check initialization
        assert partitioning.n_partitions == 6
        assert partitioning.shape == (2, 3)
        assert partitioning.n_levels == n_levels
        assert partitioning.order == ["tile"]

        # Test labels
        labels = partitioning.labels
        assert labels.shape == (2, 3)
        if origin == "bottom":
            expected_labels = np.array([["3", "4", "5"], ["0", "1", "2"]])
        else:
            expected_labels = np.array([["0", "1", "2"], ["3", "4", "5"]])
        assert labels.tolist() == expected_labels.tolist()

    @pytest.mark.parametrize("origin", ["bottom", "top"])
    def test_single_level_y_partitioning(self, origin):
        """Test single-level TilePartitioning id column-by-column."""
        size = (120, 90)  # 3x2 partitions
        extent = [-180, 180, -90, 90]
        n_levels = 1
        direction = "y"
        justify = False
        levels = None  # ["x","y"]
        partitioning = TilePartitioning(
            size=size,
            extent=extent,
            n_levels=n_levels,
            levels=levels,
            origin=origin,
            direction=direction,
            justify=justify,
        )
        # Check initialization
        assert partitioning.n_partitions == 6
        assert partitioning.shape == (2, 3)
        assert partitioning.n_levels == n_levels
        assert partitioning.order == ["tile"]

        # Test labels
        labels = partitioning.labels
        assert labels.shape == (2, 3)
        if origin == "bottom":
            expected_labels = np.array([["1", "3", "5"], ["0", "2", "4"]])
        else:
            expected_labels = np.array([["0", "2", "4"], ["1", "3", "5"]])
        assert labels.tolist() == expected_labels.tolist()

    def test_justified_label_xy(self):
        """Test labels justification of two-levels 2D TilePartitioning."""
        size = (10, 10)
        extent = [-180, 180, -90, 90]
        n_levels = 2
        origin = "bottom"
        justify = True
        partitioning = TilePartitioning(
            size=size,
            extent=extent,
            n_levels=n_levels,
            origin=origin,
            justify=justify,
        )
        # Test id justification
        assert partitioning.n_x == 36
        x_labels, y_labels = partitioning.query_labels(-180, 90)
        assert x_labels.tolist() == ["00"]
        assert y_labels.tolist() == ["00"]

    def test_justified_labels_single_level(self):
        """Test labels justification of single-level 2D TilePartitioning."""
        size = (10, 10)  #
        extent = [-180, 180, -90, 90]
        n_levels = 1
        origin = "bottom"
        justify = True
        partitioning = TilePartitioning(
            size=size,
            extent=extent,
            n_levels=n_levels,
            origin=origin,
            justify=justify,
        )
        # Test id justification
        assert partitioning.n_partitions == 648
        labels = partitioning.query_labels(-180, 90)
        assert labels.tolist() == ["000"]

    def test_add_labels_single_level_pandas(self):
        """Test valid partitions are added to a pandas dataframe."""
        # Create test dataframe
        df = pd.DataFrame(
            {
                "x": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
                "y": [-0.001, -0.0, 0, 0.5, 1.0, 1.5, 2.0, 2.1, np.nan],
            },
        )
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        n_levels = 1
        partitioning = TilePartitioning(
            size=size,
            extent=extent,
            n_levels=n_levels,
        )
        # Add partitions
        df_out = partitioning.add_labels(df, x="x", y="y", remove_invalid_rows=True)
        # Test results
        assert isinstance(df_out, pd.DataFrame)
        expected_ids = ["28", "28", "24", "17", "10", "3"]
        assert df_out["tile"].astype(str).tolist() == expected_ids, "Tile ids are incorrect."

    def test_to_dict_xy(self):
        """Test to dict."""
        size = (120, 90)  # 3x2 partitions
        extent = [-180, 180, -90, 90]
        n_levels = 2
        origin = "bottom"
        direction = "x"
        justify = True
        levels = ["xx", "yy"]
        order = ["yy", "xx"]
        flavor = "hive"
        partitioning = TilePartitioning(
            size=size,
            extent=extent,
            n_levels=n_levels,
            levels=levels,
            origin=origin,
            justify=justify,
            order=order,
            flavor=flavor,
        )
        # Test results
        expected_dict = {
            "partitioning_class": "TilePartitioning",
            "extent": list(extent),
            "size": list(size),
            "n_levels": n_levels,
            "levels": levels,
            "origin": origin,
            "direction": direction,
            "justify": justify,
            "order": order,
            "flavor": flavor,
        }
        assert partitioning.to_dict() == expected_dict

    def test_to_dict_single_level(self):
        """Test to dict."""
        size = (120, 90)  # 3x2 partitions
        extent = [-180, 180, -90, 90]
        n_levels = 1
        levels = "my_tile_id"
        origin = "top"
        direction = "y"
        justify = False
        flavor = None
        partitioning = TilePartitioning(
            size=size,
            extent=extent,
            n_levels=n_levels,
            levels=levels,
            origin=origin,
            direction=direction,
            justify=justify,
            flavor=flavor,
        )
        # Test results
        expected_dict = {
            "partitioning_class": "TilePartitioning",
            "extent": list(extent),
            "size": list(size),
            "n_levels": n_levels,
            "levels": ["my_tile_id"],
            "origin": origin,
            "direction": direction,
            "justify": justify,
            "order": ["my_tile_id"],
            "flavor": "directory",
        }
        assert partitioning.to_dict() == expected_dict

    def test_directories_single_level(self):
        """Test directories property."""
        size = (120, 90)  # 3x2 partitions
        extent = [-180, 180, -90, 90]
        n_levels = 1
        levels = "my_tile_id"
        origin = "top"
        direction = "y"
        justify = False
        flavor = None
        partitioning = TilePartitioning(
            size=size,
            extent=extent,
            n_levels=n_levels,
            levels=levels,
            origin=origin,
            direction=direction,
            justify=justify,
            flavor=flavor,
        )
        directories = partitioning.directories
        assert directories.tolist() == ["0", "2", "4", "1", "3", "5"]
