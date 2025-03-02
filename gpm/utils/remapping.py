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
"""This module contains tools for coordinates transformation and data remapping."""
import os

import dask
import numpy as np
import pyproj
import xarray as xr
from dask.array import map_blocks
from dask.array.core import normalize_chunks
from pyproj import Transformer  # pyproj >= 3.1 to be thread safe

# Future
# - reproject_dataset
# - get_active_crs()   # gpm.crs (crs.active, crs.availables)

# - Drop CRS coordinate after reprojection?
# - Set new CRS coordinate after reprojection


####------------------------------------------------------------------------------.
#### Coordinates transformations


def _transform_numpy_coords(x, y, z, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    result = transformer.transform(x, y, z)
    return result


def _transform_fun(x, y, z, src_crs, dst_crs):
    """Perform pyproj transformation and stack the results on last position.

    If using pyproj >= 3.1, it employs thread-safe pyproj.transformer.Transformer.
    If using pyproj < 3.1, it employs pyproj.transform.
    Docs: https://pyproj4.github.io/pyproj/stable/advanced_examples.html#multithreading
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return np.stack(transformer.transform(x, y, z, radians=False), axis=-1)


def _transform_dask_coords(x, y, z, src_crs, dst_crs):
    # Determine the number of dimensions and shape
    ndim = x.ndim
    chunks = (*x.chunks, 3)  # The 3 is used by map_blocks to determine output shape !
    # Conversion from geographic coordinate system to geocentric cartesian
    res = map_blocks(
        _transform_fun,
        x,
        y,
        z,
        new_axis=[ndim],
        chunks=chunks,
        dtype=x.dtype,
        src_crs=src_crs,
        dst_crs=dst_crs,
    )
    # Create a list of results by splitting the last axis of the result
    result = [res[(..., i)] for i in range(3)]
    return result


def _transform_coords(x, y, z, src_crs, dst_crs):
    if hasattr(x, "chunks"):
        return _transform_dask_coords(x, y, z, src_crs, dst_crs)
    return _transform_numpy_coords(x, y, z, src_crs, dst_crs)


def _define_n_blocks_chunks(shape, n_blocks):
    # Calculate chunk sizes for each dimension
    ndim = len(shape)
    chunk_size_per_dim = [max(1, s // int(n_blocks ** (1 / ndim))) for s in shape]
    # Normalize chunks to handle any leftover dimensions
    chunks = normalize_chunks(tuple(chunk_size_per_dim), shape=shape)
    return chunks


def chunks_inputs(x, n_blocks=None):
    """Split array into n chunks.

    Parameters
    ----------
    x : numpy.ndarray or xarray.DataArray
        The input array to be chunked.
    n_blocks: int, optional
        Number of blocks.
        If None (the default), is set equal to the number of CPUs available.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        The chunked array.
    """
    # Set n_blocks to the number of CPU cores available
    if n_blocks is None:
        n_blocks = os.cpu_count()

    # Retrieve appropriate chunks
    # TODO: improve chunks definition
    chunks = _define_n_blocks_chunks(x.shape, n_blocks)

    # Alternative
    # with dask.config.set({"array.chunk-size": f"{int(x.nbytes/n_blocks)}B"}):
    #     x = dask.array.from_array(x, chunks="auto")

    # Chunk numpy/xarray DataArray
    if isinstance(x, np.ndarray):
        return dask.array.from_array(x, chunks=chunks)
    return x.chunk(dict(zip(x.dims, chunks, strict=False)))


def reproject_coords(x, y, z=None, parallel=False, **kwargs):
    """
    Transform coordinates from a source projection to a target projection.

    Longitude coordinates should be provided as x, latitude as y.

    Parameters
    ----------
    x : numpy.ndarray, dask.array.Array or xarray.DataArray
        Array of x coordinates.
    y : numpy.ndarray, dask.array.Array or xarray.DataArray
        Array of y coordinates.
    z : numpy.ndarray, dask.array.Array or xarray.DataArray, optional
        Array of z coordinates.
    parallel: bool, optional
        Whether to use multiple cores to transform coordinates when
        input arrays are backed by numpy arrays.
        The default is ``False``.

    Keyword Arguments
    -----------------
    src_crs : pyproj.crs.CRS
        Source CRS
    dst_crs : pyproj.crs.CRS
        Destination CRS

    Returns
    -------
    trans : tuple of numpy.ndarray, dask.array.Array or xarray.DataArray
        Arrays of reprojected coordinates (X, Y) or (X, Y, Z) depending on input.
    """
    # Retrieve src and dst CRS
    if "src_crs" not in kwargs:
        raise ValueError("'src_crs' argument not specified !")
    if "dst_crs" not in kwargs:
        raise ValueError("'dst_crs' argument not specified !")
    src_crs = kwargs.get("src_crs")
    dst_crs = kwargs.get("dst_crs")

    # Check CRS validity
    if not isinstance(src_crs, pyproj.CRS) or not isinstance(dst_crs, pyproj.CRS):
        raise TypeError("'src_crs' and 'dst_crs' must be instances of pyproj.CRS")

    # Check x and y are of the same type
    if not isinstance(x, type(y)):
        raise TypeError("x and y must be of the same type.")

    # Check x and y have the same shape
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    # Check if input is xarray
    is_xarray_input = isinstance(x, xr.DataArray)

    # Check if backend is dask
    is_dask_backend = hasattr(x.data, "chunks") if is_xarray_input else hasattr(x, "chunks")

    # Initialize z if None and check
    if z is None:
        if is_xarray_input:
            z = xr.zeros_like(x)
        elif is_dask_backend:
            z = dask.array.zeros_like(x)
        else:
            z = np.zeros_like(x)

    # Chunks arrays if parallel is True
    flag_compute = False
    if parallel and not is_dask_backend:
        x = chunks_inputs(x)
        y = chunks_inputs(y)
        z = chunks_inputs(z)
        flag_compute = True

    # If xarray input, collect dims and coords and extract arrays
    if is_xarray_input:
        dims = x.dims
        coords = [x.coords, y.coords, z.coords]
        x = x.data
        y = y.data
        z = z.data

    # Transform coordinates
    result = _transform_coords(x, y, z, src_crs=src_crs, dst_crs=dst_crs)

    # Compute if necessary (when parallel=True for non-dask input data)
    if flag_compute:
        result = dask.compute(*result)

    # Return tuple of xarray objects if input is xarray
    if is_xarray_input:
        result = [xr.DataArray(result[i], coords=coords[i], dims=dims) for i in range(0, len(result))]
    return tuple(result)


####------------------------------------------------------------------------------.
