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
"""This module tests quadmesh computations."""

# # Extract cell vertices
# import numpy as np
# import xarray as xr
# import shapely

# x = da_dem_src["x"].data
# y = da_dem_src["y"].data

# x.shape
# y.shape

# #### Compute quadmesh
# x_bounds = xr.plot.utils._infer_interval_breaks(x)
# y_bounds = xr.plot.utils._infer_interval_breaks(y)
# x_bounds.shape
# y_bounds.shape


# x_corners, y_corners = np.meshgrid(x_bounds, y_bounds)
# # Quadmesh
# corners = np.stack((x_corners, y_corners), axis=2)
# corners.shape
# #### Compute Vertices
# ccw = True
# top_left = corners[:-1, :-1]
# top_right = corners[:-1, 1:]
# bottom_right = corners[1:, 1:]
# bottom_left = corners[1:, :-1]
# if ccw:
#     list_vertices = [top_left, bottom_left, bottom_right, top_right]
# else:
#     list_vertices = [top_left, top_right, bottom_right, bottom_left]
# vertices = np.stack(list_vertices, axis=2)
# vertices.shape
# vertices_flat = vertices.reshape(-1, 4, 2)
# vertices_flat.nbytes/1024/1024/1024 # 6GB
