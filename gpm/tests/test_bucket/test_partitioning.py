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
import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from gpm.bucket.partitioning import (
    XYPartitioning,
    get_array_combinations,
    get_breaks,
    get_breaks_and_labels,
    get_centroids,
    get_labels,
    get_n_decimals,
)


def test_get_n_decimals():
    """Ensure decimal count is accurate."""
    assert get_n_decimals(123.456) == 3
    assert get_n_decimals(100) == 0
    assert get_n_decimals(123.0001) == 4


def test_get_breaks():
    """Verify the correct calculation of breaks."""
    breaks = get_breaks(0.5, 0, 2)
    assert np.array_equal(breaks, np.array([0, 0.5, 1.0, 1.5, 2]))


def test_get_labels():
    """Verify correct label generation."""
    labels = get_labels(0.5, 0, 2)
    expected_labels = ["0.25", "0.75", "1.25", "1.75"]
    assert labels.tolist() == expected_labels

    labels = get_labels(0.999, 0, 2)
    expected_labels = ["0.4995", "1.4985", "2.4975"]
    assert labels.tolist() == expected_labels


def test_get_centroids():
    """Verify correct midpoint generation."""
    centroids = get_centroids(0.5, 0, 2)
    expected_centroids = [0.25, 0.75, 1.25, 1.75]
    np.testing.assert_allclose(centroids, expected_centroids)

    centroids = get_centroids(0.999, 0, 2)
    expected_centroids = [0.4995, 1.4985, 2.4975]
    np.testing.assert_allclose(centroids, expected_centroids)


def test_get_breaks_and_labels():
    """Ensure both breaks and labels are returned and accurate."""
    breaks, labels = get_breaks_and_labels(0.5, 0, 2)
    assert np.array_equal(breaks, np.array([0, 0.5, 1.0, 1.5, 2]))
    assert labels.tolist() == ["0.25", "0.75", "1.25", "1.75"]


def test_get_array_combinations():
    x = np.array([1, 2, 3])
    y = np.array([4, 5])
    expected_result = np.array([[1, 4], [2, 4], [3, 4], [1, 5], [2, 5], [3, 5]])
    np.testing.assert_allclose(get_array_combinations(x, y), expected_result)


class TestXYPartitioning:
    """Tests for the XYPartitioning class."""

    def test_initialization(self):
        """Test proper initialization of XYPartitioning objects."""
        partitioning = XYPartitioning(xbin="xbin", ybin="ybin", size=(1, 2), extent=[0, 10, 0, 10])
        assert partitioning.size == (1, 2)
        assert partitioning.partitions == ["xbin", "ybin"]
        assert list(partitioning.extent) == [0, 10, 0, 10]
        assert partitioning.shape == (10, 5)
        assert partitioning.n_partitions == 50
        assert partitioning.n_x == 10
        assert partitioning.n_y == 5
        np.testing.assert_allclose(partitioning.x_breaks, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        np.testing.assert_allclose(partitioning.y_breaks, [0, 2, 4, 6, 8, 10])
        np.testing.assert_allclose(partitioning.x_centroids, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
        np.testing.assert_allclose(partitioning.y_centroids, [1.0, 3.0, 5.0, 7.0, 9.0])
        assert partitioning.x_labels.tolist() == ["0.5", "1.5", "2.5", "3.5", "4.5", "5.5", "6.5", "7.5", "8.5", "9.5"]
        assert partitioning.y_labels.tolist() == ["1.0", "3.0", "5.0", "7.0", "9.0"]

    def test_invalid_initialization(self):
        """Test initialization with invalid extent and size."""
        with pytest.raises(ValueError):
            XYPartitioning(xbin="xbin", ybin="ybin", size=(0.1, 0.2), extent=[10, 0, 0, 10])

        with pytest.raises(TypeError):
            XYPartitioning(xbin="xbin", ybin="ybin", size="invalid", extent=[0, 10, 0, 10])

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
        partitioning = XYPartitioning(xbin="my_xbin", ybin="my_ybin", size=size, extent=extent)

        # Add partitions
        df_out = partitioning.add_labels(df, x="x", y="y", remove_invalid_rows=True)

        # Test results
        expected_xbin = [0.25, 0.25, 0.25, 0.75, 1.25, 1.75]
        expected_ybin = [0.125, 0.125, 0.375, 0.875, 1.375, 1.875]
        assert df_out["my_xbin"].dtype.name == "category", "X bin are not of categorical type."
        assert df_out["my_ybin"].dtype.name == "category", "Y bin are not of categorical type."
        assert df_out["my_xbin"].astype(float).tolist() == expected_xbin, "X bin are incorrect."
        assert df_out["my_ybin"].astype(float).tolist() == expected_ybin, "Y bin are incorrect."

    # add dask

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
        partitioning = XYPartitioning(xbin="my_xbin", ybin="my_ybin", size=size, extent=extent)

        # Add partitions
        df_out = partitioning.add_labels(df, x="x", y="y", remove_invalid_rows=True)

        # Test results
        expected_xbin = [0.25, 0.25, 0.25, 0.75, 1.25, 1.75]
        expected_ybin = [0.125, 0.125, 0.375, 0.875, 1.375, 1.875]
        assert df_out["my_xbin"].dtype == pl.datatypes.Categorical, "X bin are not of categorical type."
        assert df_out["my_ybin"].dtype == pl.datatypes.Categorical, "X bin are not of categorical type."
        assert df_out["my_xbin"].cast(float).to_list() == expected_xbin, "X bin are incorrect."
        assert df_out["my_ybin"].cast(float).to_list() == expected_ybin, "Y bin are incorrect."

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
        partitioning = XYPartitioning(xbin="my_xbin", ybin="my_ybin", size=size, extent=extent)

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
        partitioning = XYPartitioning(xbin="my_xbin", ybin="my_ybin", size=size, extent=extent)

        # Test error is raised if NaN are present in x or y column
        with pytest.raises(ValueError):
            partitioning.add_labels(df=df_null_values, x="x", y="y", remove_invalid_rows=False)

        # Test error is raised if out of extent values are present in x and y column
        with pytest.raises(ValueError):
            partitioning.add_labels(df=df_out_of_extent_values, x="x", y="y", remove_invalid_rows=False)

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
        xbin = "my_xbin"
        ybin = "my_ybin"
        partitioning = XYPartitioning(xbin=xbin, ybin=ybin, size=size, extent=extent)

        # Add partitions
        df = partitioning.add_labels(df, x="x", y="y", remove_invalid_rows=True)

        # Aggregate by partitions
        if df_type == "polars":
            df_grouped = df.group_by(partitioning.partitions).median()
            df_grouped = df_grouped.with_columns(pl.lit(2).alias("dummy_var"))
        else:
            df_grouped = df.groupby(partitioning.partitions, observed=True).median()
            df_grouped["dummy_var"] = 2

        # Convert to Dataset
        ds = partitioning.to_xarray(df_grouped, new_x="lon", new_y="lat")

        # Test results
        expected_xbin = [0.25, 0.75, 1.25, 1.75]
        expected_ybin = [0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.625, 1.875]
        assert isinstance(ds, xr.Dataset), "Not a xr.Dataset"
        assert "dummy_var" in ds, "The x columns has not become a xr.Dataset variable"
        assert ds["lon"].data.dtype.name != "object", "xr.Dataset coordinates should not be a string."
        assert ds["lat"].data.dtype.name != "object", "xr.Dataset coordinates should not be a string."
        assert ds["lon"].data.dtype.name == "float64", "xr.Dataset coordinates are not float64."
        assert ds["lat"].data.dtype.name == "float64", "xr.Dataset coordinates are not float64."
        np.testing.assert_allclose(ds["lon"].data, expected_xbin)
        np.testing.assert_allclose(ds["lat"].data, expected_ybin)

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
        xbin = "my_xbin"
        ybin = "my_ybin"
        partitioning = XYPartitioning(xbin=xbin, ybin=ybin, size=size, extent=extent)

        # Add partitions
        df = partitioning.add_labels(df, x="x", y="y", remove_invalid_rows=True)

        # Group over partitions
        df_grouped = df.groupby(partitioning.partitions, observed=True).median()
        df_grouped["dummy_var"] = 2

        # Create df with additional index (i.e. time)
        df_grouped1 = df_grouped.copy()
        df_grouped2 = df_grouped.copy()
        df_grouped1["frequency"] = "low"
        df_grouped2["frequency"] = "high"
        df_grouped1["month"] = 1
        df_grouped2["month"] = 2

        df = pd.concat((df_grouped1, df_grouped2))

        # Test categorical dtype is converted !
        df["frequency"] = pd.Categorical(df["frequency"])

        # Convert to Dataset
        if index_type == "set":
            df = df.reset_index()
            df = df.set_index([xbin, ybin, "frequency", "month"])
            indices = None
        else:
            indices = ["frequency", "month"]
        ds = partitioning.to_xarray(df, new_x="lon", new_y="lat", indices=indices)

        # Test results
        assert isinstance(ds, xr.Dataset), "Not a xr.Dataset"
        assert "dummy_var" in ds, "The x columns has not become a xr.Dataset variable."
        assert "frequency" in ds.coords, "'frequency' is not a xr.Dataset coordinate."
        assert "month" in ds.coords, "'month' is not a xr.Dataset coordinate."
        assert ds["lon"].data.dtype.name == "float64", "xr.Dataset 'lon' coordinate is not float64."
        assert ds["lat"].data.dtype.name == "float64", "xr.Dataset 'lat' coordinate is not float64."
        assert ds["frequency"].data.dtype.name == "object", "xr.Dataset 'frequency' coordinate is not an object."
        assert ds["month"].data.dtype.name == "int64", "xr.Dataset 'month' coordinate is not int64."

        da = ds["dummy_var"].isel(frequency=1, month=0, lon=slice(0, 2), lat=slice(0, 2))
        expected_arr = np.array([[2.0, 2.0], [np.nan, np.nan]])
        np.testing.assert_allclose(da.data, expected_arr)

    def test_query_labels(self):
        """Test valid labels queries."""
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        partitioning = XYPartitioning(xbin="my_xbin", ybin="my_ybin", size=size, extent=extent)
        # Test results
        assert partitioning._query_x_labels(1).tolist() == ["0.75"]
        assert partitioning._query_y_labels(1).tolist() == ["0.875"]
        assert partitioning._query_x_labels(np.array(1)).tolist() == ["0.75"]
        assert partitioning._query_x_labels(np.array([1])).tolist() == ["0.75"]
        assert partitioning._query_x_labels(np.array([1, 1])).tolist() == ["0.75", "0.75"]
        assert partitioning._query_x_labels([1, 1]).tolist() == ["0.75", "0.75"]

        x_labels, y_labels = partitioning.query_labels([1, 2], [0, 1])
        assert x_labels.tolist() == ["0.75", "1.75"]
        assert y_labels.tolist() == ["0.125", "0.875"]

        # Test out of extent
        assert partitioning._query_x_labels([-1, 1]).tolist() == ["nan", "0.75"]

        # Test with input nan
        assert partitioning._query_x_labels(np.nan).tolist() == ["nan"]
        assert partitioning._query_x_labels(None).tolist() == ["nan"]

        # Test with input string
        with pytest.raises(ValueError):
            partitioning._query_x_labels("dummy")

    def test_query_centroids(self):
        """Test valid midpoint queries."""
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        partitioning = XYPartitioning(xbin="my_xbin", ybin="my_ybin", size=size, extent=extent)
        # Test results
        np.testing.assert_allclose(partitioning._query_x_centroids(1), [0.75])
        np.testing.assert_allclose(partitioning._query_y_centroids(1).tolist(), [0.875])
        np.testing.assert_allclose(partitioning._query_x_centroids(np.array(1)), [0.75])
        np.testing.assert_allclose(partitioning._query_x_centroids(np.array([1])), [0.75])
        np.testing.assert_allclose(partitioning._query_x_centroids(np.array([1, 1])), [0.75, 0.75])
        np.testing.assert_allclose(partitioning._query_x_centroids([1, 1]), [0.75, 0.75])

        x_centroids, y_centroids = partitioning.query_centroids([1, 2], [0, 1])
        np.testing.assert_allclose(x_centroids.tolist(), [0.75, 1.75])
        np.testing.assert_allclose(y_centroids.tolist(), [0.125, 0.875])

        # Test out of extent
        np.testing.assert_allclose(partitioning._query_x_centroids([-1, 1]), [np.nan, 0.75])

        # Test with input nan or None
        np.testing.assert_allclose(partitioning._query_x_centroids(np.nan).tolist(), [np.nan])
        np.testing.assert_allclose(partitioning._query_x_centroids(None).tolist(), [np.nan])

        # Test with input string
        with pytest.raises(ValueError):
            partitioning._query_x_centroids("dummy")

    def test_get_partitions_by_extent(self):
        """Test get_partitions_by_extent."""
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        xbin = "my_xbin"
        ybin = "my_ybin"
        partitioning = XYPartitioning(xbin=xbin, ybin=ybin, size=size, extent=extent)
        # Test results with extent within
        new_extent = [0, 0.5, 0, 0.5]
        dict_labels = partitioning.get_partitions_by_extent(new_extent)
        assert dict_labels[xbin].tolist() == ["0.25", "0.25"]
        assert dict_labels[ybin].tolist() == ["0.125", "0.375"]

        # Test results with extent outside
        new_extent = [3, 4, 3, 4]
        dict_labels = partitioning.get_partitions_by_extent(new_extent)
        assert dict_labels[xbin].size == 0
        assert dict_labels[ybin].size == 0

        # Test results with extent partially overlapping
        new_extent = [1.5, 4, 1.75, 4]
        dict_labels = partitioning.get_partitions_by_extent(new_extent)
        assert dict_labels[xbin].tolist() == ["1.25", "1.75", "1.25", "1.75"]
        assert dict_labels[ybin].tolist() == ["1.625", "1.625", "1.875", "1.875"]

    def test_get_partitions_around_point(self):
        """Test get_partitions_around_point."""
        # Create partitioning
        size = (0.5, 0.25)
        extent = [0, 2, 0, 2]
        xbin = "my_xbin"
        ybin = "my_ybin"
        partitioning = XYPartitioning(xbin=xbin, ybin=ybin, size=size, extent=extent)
        # Test results with point within
        dict_labels = partitioning.get_partitions_around_point(x=1, y=1, distance=0)
        assert dict_labels[xbin].tolist() == ["0.75"]
        assert dict_labels[ybin].tolist() == ["0.875"]

        # Test results with point outside
        dict_labels = partitioning.get_partitions_around_point(x=3, y=3, distance=0)
        assert dict_labels[xbin].size == 0
        assert dict_labels[ybin].size == 0

        # Test results with point outside but area within
        dict_labels = partitioning.get_partitions_around_point(x=3, y=3, distance=1)
        assert dict_labels[xbin].tolist() == ["1.75"]
        assert dict_labels[ybin].tolist() == ["1.875"]

    def test_quadmesh(self):
        """Test quadmesh."""
        size = (1, 1)
        extent = [0, 2, 1, 3]
        partitioning = XYPartitioning(xbin="my_xbin", ybin="my_ybin", size=size, extent=extent)
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
        xbin = "my_xbin"
        ybin = "my_ybin"
        partitioning = XYPartitioning(xbin=xbin, ybin=ybin, size=size, extent=extent)
        # Test results
        expected_dict = {
            "name": "XYPartitioning",
            "extent": list(extent),
            "size": list(size),
            "xbin": xbin,
            "ybin": ybin,
            "partitions": [xbin, ybin],
            "partitioning_flavor": None,
        }
        assert partitioning.to_dict() == expected_dict
