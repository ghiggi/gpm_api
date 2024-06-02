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
"""This module contains basic functions for GPM-API data visualization."""
import inspect

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Geod
from scipy.interpolate import griddata

import gpm
from gpm import get_plot_kwargs


def is_generator(obj):
    return inspect.isgeneratorfunction(obj) or inspect.isgenerator(obj)


def _call_optimize_layout(self):
    """Optimize the figure layout."""
    adapt_fig_size(ax=self.axes)
    self.figure.tight_layout()


def add_optimize_layout_method(p):
    """Add a method to optimize the figure layout using monkey patching."""
    p.optimize_layout = _call_optimize_layout.__get__(p, type(p))
    return p


def adapt_fig_size(ax, nrow=1, ncol=1):
    """Adjusts the figure height of the plot based on the aspect ratio of cartopy subplots.

    This function is intended to be called after all plotting has been completed.
    It operates under the assumption that all subplots within the figure share the same aspect ratio.

    Assumes that the first axis in the collection of axes is representative of all others.
    This means that all subplots are expected to have the same aspect ratio and size.

    The implementation is inspired by Mathias Hauser's mplotutils set_map_layout function.
    """
    # Determine the number of rows and columns of subplots in the figure.
    # This information is crucial for calculating the new height of the figure.
    # nrow, ncol, __, __ = ax.get_subplotspec().get_geometry()

    # Access the figure object from the axis to manipulate its properties.
    fig = ax.get_figure()

    # Retrieve the current size of the figure in inches.
    width, original_height = fig.get_size_inches()

    # A call to draw the canvas is required to make sure the geometry of the figure is up-to-date.
    # This ensures that subsequent calculations for adjusting the layout are based on the latest state.
    fig.canvas.draw()

    # Extract subplot parameters to understand the figure's layout.
    # These parameters include the margins of the figure and the spaces between subplots.
    bottom = fig.subplotpars.bottom
    top = fig.subplotpars.top
    left = fig.subplotpars.left
    right = fig.subplotpars.right
    hspace = fig.subplotpars.hspace  # vertical space between subplots
    wspace = fig.subplotpars.wspace  # horizontal space between subplots

    # Calculate the aspect ratio of the data in the subplot.
    # This ratio is used to adjust the height of the figure to match the aspect ratio of the data.
    aspect = ax.get_data_ratio()

    # Calculate the width of a single plot, considering the left and right margins,
    # the number of columns, and the space between columns.
    wp = (width - width * (left + (1 - right))) / (ncol + (ncol - 1) * wspace)

    # Calculate the height of a single plot using its width and the data aspect ratio.
    hp = wp * aspect

    # Calculate the new height of the figure, taking into account the number of rows,
    # the space between rows, and the top and bottom margins.
    height = (hp * (nrow + ((nrow - 1) * hspace))) / (1.0 - (bottom + (1 - top)))

    # Check if the new height is significantly reduced (more than halved).
    if original_height / height > 2:
        # Calculate the scale factor to adjust the figure size closer to the original.
        scale_factor = original_height / height / 2

        # Apply the scale factor to both width and height to maintain the aspect ratio.
        width *= scale_factor
        height *= scale_factor

    # Apply the calculated width and height to adjust the figure size.
    fig.set_figwidth(width)
    fig.set_figheight(height)


####--------------------------------------------------------------------------.


def get_antimeridian_mask(lons):
    """Get mask of longitude coordinates neighbors crossing the antimeridian."""
    from scipy.ndimage import binary_dilation

    # Initialize mask
    n_y, n_x = lons.shape
    mask = np.zeros((n_y - 1, n_x - 1))
    # Check vertical edges
    row_idx, col_idx = np.where(np.abs(np.diff(lons, axis=0)) > 180)
    col_idx = np.clip(col_idx - 1, 0, n_x - 1)
    mask[row_idx, col_idx] = 1
    # Check horizontal edges
    row_idx, col_idx = np.where(np.abs(np.diff(lons, axis=1)) > 180)
    row_idx = np.clip(row_idx - 1, 0, n_y - 1)
    mask[row_idx, col_idx] = 1
    # Buffer by 1 in all directions to avoid plotting cells neighbour to those crossing the antimeridian
    # --> This should not be needed, but it's needed to avoid cartopy bugs !
    return binary_dilation(mask)


def infill_invalid_coords(xr_obj, x="lon", y="lat"):
    """Infill invalid coordinates.

    Interpolate the coordinates within the convex hull of data.
    Use nearest neighbour outside the convex hull of data.
    """
    # Copy object
    xr_obj = xr_obj.copy()
    lon = np.asanyarray(xr_obj[x].data)
    lat = np.asanyarray(xr_obj[y].data)
    # Retrieve infilled coordinates
    lon, lat, _ = get_valid_pcolormesh_inputs(x=lon, y=lat, data=None, mask_data=False)
    xr_obj[x].data = lon
    xr_obj[y].data = lat
    return xr_obj


def get_valid_pcolormesh_inputs(x, y, data, rgb=False, mask_data=True):
    """Infill invalid coordinates.

    Interpolate the coordinates within the convex hull of data.
    Use nearest neighbour outside the convex hull of data.

    This operation is required to plot with pcolormesh since it
    does not accept non-finite values in the coordinates.

    If  ``mask_data=True``, data values with invalid coordinates are masked
    and a numpy masked array is returned.
    Masked data values are not displayed in pcolormesh !
    If ``rgb=True``, it assumes the RGB dimension is the last data dimension.

    """
    # Retrieve mask of invalid coordinates
    x_invalid = ~np.isfinite(x)
    y_invalid = ~np.isfinite(y)
    mask = np.logical_or(x_invalid, y_invalid)

    # If no invalid coordinates, return original data
    if np.all(~mask):
        return x, y, data

    # Check at least ome valid coordinates
    if np.all(mask):
        raise ValueError("No valid coordinates.")

    # Mask the data
    if mask_data:
        if rgb:
            data_mask = np.broadcast_to(np.expand_dims(mask, axis=-1), data.shape)
            data_masked = np.ma.masked_where(data_mask, data)
        else:
            data_masked = np.ma.masked_where(mask, data)
    else:
        data_masked = data

    # Infill x and y
    # - Note: currently cause issue if NaN when crossing antimeridian ...
    # --> TODO: interpolation should be done in X,Y,Z
    if np.any(x_invalid):
        x = _interpolate_data(x, method="linear")  # interpolation
        x = _interpolate_data(x, method="nearest")  # nearest neighbours outside the convex hull
    if np.any(y_invalid):
        y = _interpolate_data(y, method="linear")  # interpolation
        y = _interpolate_data(y, method="nearest")  # nearest neighbours outside the convex hull
    return x, y, data_masked


def _interpolate_data(arr, method="linear"):
    # 1D coordinate (i.e. along_track/cross_track view)
    if arr.ndim == 1:
        return _interpolate_1d_coord(arr, method=method)
    # 2D coordinates (swath image)
    return _interpolate_2d_coord(arr, method=method)


def _interpolate_1d_coord(arr, method="linear"):
    # Find invalid locations
    is_invalid = ~np.isfinite(arr)

    # Find the indices of NaN values
    nan_indices = np.where(is_invalid)[0]

    # Return array if not NaN values
    if len(nan_indices) == 0:
        return arr

    # Find the indices of non-NaN values
    non_nan_indices = np.where(~is_invalid)

    # Create indices
    indices = np.arange(len(arr))

    # Points where we have valid data
    points = indices[non_nan_indices]

    # Points where data is NaN
    points_nan = indices[nan_indices]

    # Values at the non-NaN points
    values = arr[non_nan_indices]

    # Interpolate using griddata
    arr_new = arr.copy()
    arr_new[nan_indices] = griddata(points, values, points_nan, method=method)
    return arr_new


def _interpolate_2d_coord(arr, method="linear"):
    # Find invalid locations
    is_invalid = ~np.isfinite(arr)

    # Find the indices of NaN values
    nan_indices = np.where(is_invalid)

    # Return array if not NaN values
    if len(nan_indices) == 0:
        return arr

    # Find the indices of non-NaN values
    non_nan_indices = np.where(~is_invalid)

    # Create a meshgrid of indices
    x, y = np.meshgrid(range(arr.shape[1]), range(arr.shape[0]))

    # Points (X, Y) where we have valid data
    points = np.array([y[non_nan_indices], x[non_nan_indices]]).T

    # Points where data is NaN
    points_nan = np.array([y[nan_indices], x[nan_indices]]).T

    # Values at the non-NaN points
    values = arr[non_nan_indices]

    # Interpolate using griddata
    arr_new = arr.copy()
    arr_new[nan_indices] = griddata(points, values, points_nan, method=method)
    return arr_new


def _predict_forward_point(lon1, lat1, lon2, lat2, geod):
    """
    Predict a point in the forward direction at the same distance as between two given vertices.

    Parameters
    ----------
    lon1, lat1 : Longitude and latitude of the starting vertex.
    lon2, lat2 : Longitude and latitude of the ending vertex.
    geod       : Pyproj Geod object for geodesic calculations.

    Returns
    -------
    (lon3, lat3): Longitude and latitude of the predicted point in the forward direction.
    """
    # Calculate the forward azimuth and distance between the start and end vertices
    fwd_azimuth, _, distance = geod.inv(lon1, lat1, lon2, lat2)

    # Predict the point in the forward direction from the end vertex
    lon3, lat3, _ = geod.fwd(lon2, lat2, fwd_azimuth, distance)

    return lon3, lat3


def compute_lon_lat_corners_for_1d_swath(lon, lat):
    """Compute lon/lat corners from 1D swath.

    It infer the footprint size based on the spacing between footprints.
    """
    geod = Geod(ellps="WGS84")
    # Define corners between centroids
    ortho_line1_lon = []
    ortho_line1_lat = []
    ortho_line2_lon = []
    ortho_line2_lat = []

    # Process each segment of the original line
    for i in range(len(lon) - 1):
        lon1, lat1 = lon[i], lat[i]
        lon2, lat2 = lon[i + 1], lat[i + 1]

        # Calculate the forward azimuth and distance between consecutive vertices
        fwd_azimuth, _, distance = geod.inv(lon1, lat1, lon2, lat2, radians=False)

        # Calculate the half distance for the orthogonal projection
        half_distance = distance / 2

        # Orthogonal azimuths, +90 degrees and -90 degrees
        ortho_azimuth1 = (fwd_azimuth + 90) % 360
        ortho_azimuth2 = (fwd_azimuth - 90) % 360

        # Calculate the orthogonal points from the midpoint of the segment
        mid_lon, mid_lat, _ = geod.fwd((lon1 + lon2) / 2, (lat1 + lat2) / 2, fwd_azimuth, distance / 2)

        # Project the orthogonal points
        lon_ortho1, lat_ortho1, _ = geod.fwd(mid_lon, mid_lat, ortho_azimuth1, half_distance)
        lon_ortho2, lat_ortho2, _ = geod.fwd(mid_lon, mid_lat, ortho_azimuth2, half_distance)

        # Append the calculated orthogonal points to the lines
        ortho_line1_lon.append(lon_ortho1)
        ortho_line1_lat.append(lat_ortho1)
        ortho_line2_lon.append(lon_ortho2)
        ortho_line2_lat.append(lat_ortho2)

    # Extend at the extremities to define the corners
    # - Extend the start of ortho_line1
    start_ext_lon1, start_ext_lat1 = _predict_forward_point(
        ortho_line1_lon[1],
        ortho_line1_lat[1],
        ortho_line1_lon[0],
        ortho_line1_lat[0],
        geod=geod,
    )
    # - Extend the end of ortho_line1
    end_ext_lon1, end_ext_lat1 = _predict_forward_point(
        ortho_line1_lon[-2],
        ortho_line1_lat[-2],
        ortho_line1_lon[-1],
        ortho_line1_lat[-1],
        geod=geod,
    )

    # - Extend the start of ortho_line2
    start_ext_lon2, start_ext_lat2 = _predict_forward_point(
        ortho_line2_lon[1],
        ortho_line2_lat[1],
        ortho_line2_lon[0],
        ortho_line2_lat[0],
        geod=geod,
    )
    # - Extend the end of ortho_line2
    end_ext_lon2, end_ext_lat2 = _predict_forward_point(
        ortho_line2_lon[-2],
        ortho_line2_lat[-2],
        ortho_line2_lon[-1],
        ortho_line2_lat[-1],
        geod=geod,
    )

    # Define corners
    lon_corners1 = np.array([start_ext_lon1, *ortho_line1_lon, end_ext_lon1])
    lat_corners1 = np.array([start_ext_lat1, *ortho_line1_lat, end_ext_lat1])

    lat_corners2 = np.array([start_ext_lat2, *ortho_line2_lat, end_ext_lat2])
    lon_corners2 = np.array([start_ext_lon2, *ortho_line2_lon, end_ext_lon2])

    lon_corners = np.vstack((lon_corners1, lon_corners2)).T
    lat_corners = np.vstack((lat_corners1, lat_corners2)).T

    # Define lon lat corners
    return lon_corners, lat_corners


def get_lon_lat_corners(lon, lat):
    from gpm.utils.area import _get_lonlat_corners

    if lon.ndim == 1:
        lon, lat = compute_lon_lat_corners_for_1d_swath(lon, lat)
    else:
        lon, lat = _get_lonlat_corners(lon, lat)
    return lon, lat


def _mask_antimeridian_crossing_arr(arr, antimeridian_mask, rgb):
    if np.ma.is_masked(arr):
        if rgb:
            antimeridian_mask = np.broadcast_to(np.expand_dims(antimeridian_mask, axis=-1), arr.shape)
            combined_mask = np.logical_or(arr.mask, antimeridian_mask)
        else:
            combined_mask = np.logical_or(arr.mask, antimeridian_mask)
        arr = np.ma.masked_where(combined_mask, arr)
    else:
        if rgb:
            antimeridian_mask = np.broadcast_to(
                np.expand_dims(antimeridian_mask, axis=-1),
                arr.shape,
            )
        arr = np.ma.masked_where(antimeridian_mask, arr)
    return arr


def mask_antimeridian_crossing_array(arr, lon, rgb, plot_kwargs):
    """Mask the array cells crossing the antimeridian.

    Here we assume not invalid lon coordinates anymore.
    Cartopy still bugs with several projections when data cross the antimeridian.
    By default, GPM-API mask data crossing the antimeridian.
    The GPM-API configuration default can be modified with: ``gpm.config.set({"viz_hide_antimeridian_data": False})``
    """
    antimeridian_mask = get_antimeridian_mask(lon)
    is_crossing_antimeridian = np.any(antimeridian_mask)
    if is_crossing_antimeridian:
        # Sanitize cmap to avoid cartopy bug related to cmap bad color
        # - Cartopy requires the bad color to be fully transparent
        plot_kwargs = _sanitize_cartopy_plot_kwargs(plot_kwargs)
        # Mask data based on GPM-API config 'viz_hide_antimeridian_data'
        if gpm.config.get("viz_hide_antimeridian_data"):  # default is True
            arr = _mask_antimeridian_crossing_arr(arr, antimeridian_mask=antimeridian_mask, rgb=rgb)
    return arr, plot_kwargs


####--------------------------------------------------------------------------.
########################
#### Plot utilities ####
########################


def preprocess_rgb_dataarray(da, rgb):
    if rgb:
        if rgb not in da.dims:
            raise ValueError(f"The specified rgb='{rgb}' must be a dimension of the DataArray.")
        if da[rgb].size not in [3, 4]:
            raise ValueError("The RGB dimension must have size 3 or 4.")
        da = da.transpose(..., rgb)
    return da


def check_object_format(da, plot_kwargs, check_function, **function_kwargs):
    """Check object format and valid dimension names."""
    # Preprocess RGB DataArrays
    da = da.squeeze()
    da = preprocess_rgb_dataarray(da, plot_kwargs.get("rgb", False))
    # Retrieve rgb or FacetGrid column/row dimensions
    dims_dict = {key: plot_kwargs.get(key) for key in ["rgb", "col", "row"] if plot_kwargs.get(key, None)}
    # Check such dimensions are available
    for key, dim in dims_dict.items():
        if dim not in da.dims:
            raise ValueError(f"The DataArray does not have a {key}='{dim}' dimension.")
    # Subset DataArray to check if complies with specific check function
    isel_dict = {dim: 0 for dim in dims_dict.values()}
    check_function(da.isel(isel_dict), **function_kwargs)
    return da


def preprocess_figure_args(ax, fig_kwargs=None, subplot_kwargs=None, is_facetgrid=False):
    if is_facetgrid and ax is not None:
        raise ValueError("When plotting with FacetGrid, do not specify the 'ax'.")
    fig_kwargs = {} if fig_kwargs is None else fig_kwargs
    subplot_kwargs = {} if subplot_kwargs is None else subplot_kwargs
    if ax is not None:
        if len(subplot_kwargs) >= 1:
            raise ValueError("Provide `subplot_kwargs`only if ``ax``is None")
        if len(fig_kwargs) >= 1:
            raise ValueError("Provide `fig_kwargs` only if ``ax``is None")
    return fig_kwargs


def preprocess_subplot_kwargs(subplot_kwargs):
    subplot_kwargs = {} if subplot_kwargs is None else subplot_kwargs
    subplot_kwargs = subplot_kwargs.copy()
    if "projection" not in subplot_kwargs:
        subplot_kwargs["projection"] = ccrs.PlateCarree()
    return subplot_kwargs


def infer_xy_labels(da, x=None, y=None, rgb=None):
    from xarray.plot.utils import _infer_xy_labels

    # Infer dimensions
    x, y = _infer_xy_labels(da, x=x, y=y, imshow=True, rgb=rgb)  # dummy flag for rgb
    return x, y


def initialize_cartopy_plot(
    ax,
    fig_kwargs,
    subplot_kwargs,
    add_background,
):
    """Initialize figure for cartopy plot if necessary."""
    # - Initialize figure
    if ax is None:
        fig_kwargs = preprocess_figure_args(
            ax=ax,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
        )
        subplot_kwargs = preprocess_subplot_kwargs(subplot_kwargs)
        _, ax = plt.subplots(subplot_kw=subplot_kwargs, **fig_kwargs)

    # - Add cartopy background
    if add_background:
        ax = plot_cartopy_background(ax)
    return ax


def plot_cartopy_background(ax):
    """Plot cartopy background."""
    # - Add coastlines
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, facecolor=[0.9, 0.9, 0.9])
    ax.add_feature(cartopy.feature.OCEAN, alpha=0.6)
    ax.add_feature(cartopy.feature.BORDERS)  # BORDERS also draws provinces, ...
    # - Add grid lines
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.1,
        linestyle="-",
    )
    gl.top_labels = False  # gl.xlabels_top = False
    gl.right_labels = False  # gl.ylabels_right = False
    gl.xlines = True
    gl.ylines = True
    return ax


def plot_sides(sides, ax, **plot_kwargs):
    """Plot boundary sides.

    Expects a list of (lon, lat) tuples.
    """
    for side in sides:
        p = ax.plot(*side, transform=ccrs.Geodetic(), **plot_kwargs)
    return p[0]


####--------------------------------------------------------------------------.
############################
#### Colorbar utilities ####
############################


def _get_orientation_location(cbar_kwargs):
    location = cbar_kwargs.get("location", None)
    orientation = cbar_kwargs.get("orientation", None)
    # Set defaults
    if location is None and orientation is None:
        return "vertical", "right"
    # Check orientation is horizontal or vertical
    if orientation is not None and orientation not in ("horizontal", "vertical"):
        raise ValueError("Invalid orientation. Choose 'horizontal' or 'vertical'.")
    # Check location is top, left, right or bottom
    if location is not None and location not in ("top", "left", "right", "bottom"):
        raise ValueError("Invalid location. Choose 'top', 'left', 'right', or 'bottom'.")
    # Check compatible arguments
    if orientation is not None and location is not None:
        if orientation == "vertical":
            # Raise error if not right or left
            if location not in ("right", "left"):
                raise ValueError("Invalid location for vertical orientation. Choose 'right' or 'left'.")
        elif location not in ("bottom", "top"):
            raise ValueError("Invalid location for horizontal orientation. Choose 'bottom' or 'top'.")
        return orientation, location
    # Return with default location if missing
    if orientation is not None:
        if orientation == "vertical":
            return "vertical", "right"
        return "horizontal", "bottom"
    # Return with correct orientation if missing
    # if location is not None:
    if location in ("right", "left"):
        return "vertical", location
    return "horizontal", location


def plot_colorbar(p, ax, cbar_kwargs=None):
    """Add a colorbar to a matplotlib/cartopy plot.

    cbar_kwargs 'size' and 'pad' controls the size of the colorbar.
    and the padding between the plot and the colorbar.

    p: `matplotlib.image.AxesImage`
    ax:  `cartopy.mpl.geoaxes.GeoAxesSubplot`
    """
    cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
    cbar_kwargs = cbar_kwargs.copy()  # otherwise pop ticklabels outside the function
    ticklabels = cbar_kwargs.pop("ticklabels", None)
    orientation, location = _get_orientation_location(cbar_kwargs)
    # Defne colorbar axis
    divider = make_axes_locatable(ax)
    if orientation == "vertical":
        size = cbar_kwargs.get("size", "5%")
        pad = cbar_kwargs.get("pad", 0.1)
        cax = divider.append_axes(location, size=size, pad=pad, axes_class=plt.Axes)
    else:  # orientation == "horizontal":
        size = cbar_kwargs.get("size", "5%")
        pad = cbar_kwargs.get("pad", 0.25)
        cax = divider.append_axes(location, size=size, pad=pad, axes_class=plt.Axes)
    p.figure.add_axes(cax)
    # Add colorbar
    cbar = plt.colorbar(p, cax=cax, ax=ax, **cbar_kwargs)
    if ticklabels is not None:
        _ = cbar.ax.set_yticklabels(ticklabels) if orientation == "vertical" else cbar.ax.set_xticklabels(ticklabels)
    return cbar


def set_colorbar_fully_transparent(p):
    """Add a fully transparent colorbar.

    This is useful for animation where the colorbar should
    not always in all frames but the plot area must be fixed.
    """
    # Get the position of the colorbar
    cbar_pos = p.colorbar.ax.get_position()

    cbar_x, cbar_y = cbar_pos.x0, cbar_pos.y0
    cbar_width, cbar_height = cbar_pos.width, cbar_pos.height

    # Remove the colorbar
    p.colorbar.ax.set_visible(False)

    # Now plot an empty rectangle
    fig = plt.gcf()
    rect = plt.Rectangle(
        (cbar_x, cbar_y),
        cbar_width,
        cbar_height,
        transform=fig.transFigure,
        facecolor="none",
        edgecolor="none",
    )

    fig.patches.append(rect)


####--------------------------------------------------------------------------.
##########################
#### Cartopy wrappers ####
##########################


def _sanitize_cartopy_plot_kwargs(plot_kwargs):
    """Sanitize 'cmap' to avoid cartopy bug related to cmap bad color.

    Cartopy requires the bad color to be fully transparent.
    """
    cmap = plot_kwargs.get("cmap", None)
    if cmap is not None:
        bad = cmap.get_bad()
        bad[3] = 0  # enforce to 0 (transparent)
        cmap.set_bad(bad)
        plot_kwargs["cmap"] = cmap
    return plot_kwargs


def _compute_extent(x_coords, y_coords):
    """Compute the extent (x_min, x_max, y_min, y_max) from the pixel centroids in x and y coordinates.

    This function assumes that the spacing between each pixel is uniform.
    """
    # Calculate the pixel size assuming uniform spacing between pixels
    pixel_size_x = (x_coords[-1] - x_coords[0]) / (len(x_coords) - 1)
    pixel_size_y = (y_coords[-1] - y_coords[0]) / (len(y_coords) - 1)

    # Adjust min and max to get the corners of the outer pixels
    x_min, x_max = x_coords[0] - pixel_size_x / 2, x_coords[-1] + pixel_size_x / 2
    y_min, y_max = y_coords[0] - pixel_size_y / 2, y_coords[-1] + pixel_size_y / 2

    return [x_min, x_max, y_min, y_max]


def plot_cartopy_imshow(
    ax,
    da,
    x,
    y,
    interpolation="nearest",
    add_colorbar=True,
    plot_kwargs=None,
    cbar_kwargs=None,
):
    """Plot imshow with cartopy."""
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs

    # Infer x and y
    x, y = infer_xy_labels(da, x=x, y=y, rgb=plot_kwargs.get("rgb", None))
    # - Ensure image with correct dimensions orders
    da = da.transpose(y, x, ...)
    arr = np.asanyarray(da.data)

    # - Compute coordinates
    x_coords = da[x].to_numpy()
    y_coords = da[y].to_numpy()

    # - Derive extent
    extent = _compute_extent(x_coords=x_coords, y_coords=y_coords)

    # - Determine origin based on the orientation of da[y] values
    # -->  If increasing, set origin="lower"
    # -->  If decreasing, set origin="upper"
    origin = "lower" if y_coords[1] > y_coords[0] else "upper"

    # - Add variable field with cartopy
    rgb = plot_kwargs.pop("rgb", False)
    p = ax.imshow(
        arr,
        transform=ccrs.PlateCarree(),
        extent=extent,
        origin=origin,
        interpolation=interpolation,
        **plot_kwargs,
    )
    # - Set the extent
    ax.set_extent(extent)

    # - Add colorbar
    if add_colorbar and not rgb:
        _ = plot_colorbar(p=p, ax=ax, cbar_kwargs=cbar_kwargs)
    return p


def plot_cartopy_pcolormesh(
    ax,
    da,
    x,
    y,
    add_colorbar=True,
    add_swath_lines=True,
    plot_kwargs=None,
    cbar_kwargs=None,
):
    """Plot imshow with cartopy.

    x and y must represents longitude and latitudes.
    The function currently does not allow to zoom on regions across the antimeridian.
    The function mask scanning pixels which spans across the antimeridian.
    If the DataArray has a RGB dimension, plot_kwargs should contain the ``rgb``
    key with the name of the RGB dimension.

    """
    plot_kwargs = {} if plot_kwargs is None else plot_kwargs

    # Remove RGB from plot_kwargs
    rgb = plot_kwargs.pop("rgb", False)

    # Get x, y, and array to plot
    da = preprocess_rgb_dataarray(da, rgb=rgb)
    da = da.compute()
    lon = da[x].data.copy()
    lat = da[y].data.copy()
    arr = da.data

    # Check if 1D coordinate (nadir-looking / transect / cross-section case)
    is_1d_case = lon.ndim == 1

    # Infill invalid value and mask data at invalid coordinates
    # - No invalid values after this function call
    lon, lat, arr = get_valid_pcolormesh_inputs(lon, lat, arr, rgb=rgb, mask_data=True)
    if is_1d_case:
        arr = np.expand_dims(arr, axis=1)

    # Ensure arguments
    if rgb:
        add_colorbar = False

    # Compute coordinates of cell corners for pcolormesh quadrilateral mesh
    # - This enable correct masking of cells crossing the antimeridian
    lon, lat = get_lon_lat_corners(lon, lat)

    # Mask cells crossing the antimeridian
    # - with gpm.config.set({"viz_hide_antimeridian_data": False}): can be used to modify the masking behaviour
    arr, plot_kwargs = mask_antimeridian_crossing_array(arr, lon, rgb, plot_kwargs)

    # Add variable field with cartopy
    _ = plot_kwargs.setdefault("shading", "flat")
    p = ax.pcolormesh(
        lon,
        lat,
        arr,
        transform=ccrs.PlateCarree(),
        **plot_kwargs,
    )

    # Add swath lines
    if add_swath_lines and not is_1d_case:
        sides = [(lon[0, :], lat[0, :]), (lon[-1, :], lat[-1, :])]
        plot_sides(sides=sides, ax=ax, linestyle="--", color="black")

    # Add colorbar
    if add_colorbar:
        _ = plot_colorbar(p=p, ax=ax, cbar_kwargs=cbar_kwargs)
    return p


####-------------------------------------------------------------------------------.
#########################
#### Xarray wrappers ####
#########################


def _preprocess_xr_kwargs(add_colorbar, plot_kwargs, cbar_kwargs):
    if not add_colorbar:
        cbar_kwargs = None

    if "rgb" in plot_kwargs:
        cbar_kwargs = None
        add_colorbar = False
        plot_kwargs = {"rgb": plot_kwargs.get("rgb")}  # alpha currently skipped if RGB
    return add_colorbar, plot_kwargs, cbar_kwargs


def plot_xr_pcolormesh(
    ax,
    da,
    x,
    y,
    add_colorbar=True,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot pcolormesh with xarray."""
    ticklabels = cbar_kwargs.pop("ticklabels", None)
    add_colorbar, plot_kwargs, cbar_kwargs = _preprocess_xr_kwargs(
        add_colorbar=add_colorbar,
        plot_kwargs=plot_kwargs,
        cbar_kwargs=cbar_kwargs,
    )
    p = da.plot.pcolormesh(
        x=x,
        y=y,
        ax=ax,
        add_colorbar=add_colorbar,
        cbar_kwargs=cbar_kwargs,
        **plot_kwargs,
    )
    plt.title(da.name)
    if add_colorbar and ticklabels is not None:
        p.colorbar.ax.set_yticklabels(ticklabels)
    return p


def plot_xr_imshow(
    ax,
    da,
    x,
    y,
    interpolation="nearest",
    add_colorbar=True,
    cbar_kwargs=None,
    visible_colorbar=True,
    **plot_kwargs,
):
    """Plot imshow with xarray.

    The colorbar is added with xarray to enable to display multiple colorbars
    when calling this function multiple times on different fields with
    different colorbars.
    """
    ticklabels = cbar_kwargs.pop("ticklabels", None)
    add_colorbar, plot_kwargs, cbar_kwargs = _preprocess_xr_kwargs(
        add_colorbar=add_colorbar,
        plot_kwargs=plot_kwargs,
        cbar_kwargs=cbar_kwargs,
    )
    p = da.plot.imshow(
        x=x,
        y=y,
        ax=ax,
        interpolation=interpolation,
        add_colorbar=add_colorbar,
        cbar_kwargs=cbar_kwargs,
        **plot_kwargs,
    )
    plt.title(da.name)
    if add_colorbar and ticklabels is not None:
        p.colorbar.ax.set_yticklabels(ticklabels)

    # Make the colorbar fully transparent with a smart trick ;)
    # - TODO: this still cause issues when plotting 2 colorbars !
    if add_colorbar and not visible_colorbar:
        set_colorbar_fully_transparent(p)

    # Add manually the colorbar
    # p = da.plot.imshow(
    #     x=x,
    #     y=y,
    #     ax=ax,
    #     interpolation=interpolation,
    #     add_colorbar=False,
    #     **plot_kwargs,
    # )
    # plt.title(da.name)
    # if add_colorbar:
    #     _ = plot_colorbar(p=p, ax=ax, cbar_kwargs=cbar_kwargs)
    return p


####--------------------------------------------------------------------------.
####################
#### Plot Image ####
####################


def _plot_image(
    da,
    x=None,
    y=None,
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    fig_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot GPM orbit granule as in image."""
    from gpm.checks import is_grid, is_orbit
    from gpm.visualization.facetgrid import sanitize_facetgrid_plot_kwargs

    fig_kwargs = preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs)

    # - Initialize figure
    if ax is None:
        _, ax = plt.subplots(**fig_kwargs)

    # - Sanitize plot_kwargs set by by xarray FacetGrid.map_dataarray
    is_facetgrid = plot_kwargs.get("_is_facetgrid", False)
    plot_kwargs = sanitize_facetgrid_plot_kwargs(plot_kwargs)

    # - If not specified, retrieve/update plot_kwargs and cbar_kwargs as function of product name
    plot_kwargs, cbar_kwargs = get_plot_kwargs(
        name=da.name,
        user_plot_kwargs=plot_kwargs,
        user_cbar_kwargs=cbar_kwargs,
    )

    # - Plot with xarray
    p = plot_xr_imshow(
        ax=ax,
        da=da,
        x=x,
        y=y,
        interpolation=interpolation,
        add_colorbar=add_colorbar,
        cbar_kwargs=cbar_kwargs,
        **plot_kwargs,
    )

    if is_orbit(da):
        ax.set_xlabel("Along-Track")
        ax.set_ylabel("Cross-Track")
    elif is_grid(da):
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    # - Monkey patch the mappable instance to add optimize_layout
    if not is_facetgrid:
        p = add_optimize_layout_method(p)

    # - Return mappable
    return p


def _plot_image_facetgrid(
    da,
    x=None,
    y=None,
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    fig_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot 2D fields with FacetGrid."""
    from gpm.visualization.facetgrid import ImageFacetGrid

    # Check inputs
    fig_kwargs = preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs, is_facetgrid=True)

    # Retrieve GPM-API defaults cmap and cbar kwargs
    variable = da.name
    plot_kwargs, cbar_kwargs = get_plot_kwargs(
        name=variable,
        user_plot_kwargs=plot_kwargs,
        user_cbar_kwargs=cbar_kwargs,
    )

    # Disable colorbar if rgb
    if plot_kwargs.get("rgb", False):
        add_colorbar = False
        cbar_kwargs = {}

    # Create FacetGrid
    fc = ImageFacetGrid(
        data=da.compute(),
        col=plot_kwargs.pop("col", None),
        row=plot_kwargs.pop("row", None),
        col_wrap=plot_kwargs.pop("col_wrap", None),
        axes_pad=plot_kwargs.pop("axes_pad", None),
        fig_kwargs=fig_kwargs,
        cbar_kwargs=cbar_kwargs,
        add_colorbar=add_colorbar,
        aspect=plot_kwargs.pop("aspect", False),
        facet_height=plot_kwargs.pop("facet_height", 3),
        facet_aspect=plot_kwargs.pop("facet_aspect", 1),
    )

    # Plot the maps
    fc = fc.map_dataarray(
        _plot_image,
        x=x,
        y=y,
        add_colorbar=False,
        interpolation=interpolation,
        cbar_kwargs=cbar_kwargs,
        **plot_kwargs,
    )

    fc.remove_duplicated_axis_labels()

    # Add colorbar
    if add_colorbar:
        fc.add_colorbar(**cbar_kwargs)

    return fc


def plot_image(
    da,
    x=None,
    y=None,
    ax=None,
    add_colorbar=True,
    interpolation="nearest",
    fig_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot data using imshow.

    Parameters
    ----------
    da : `xr.DataArray`
        xarray DataArray.
    x : str, optional
        X dimension name.
        If ``None``, takes the second dimension.
        The default is ``None``.
    y : str, optional
        Y dimension name.
        If ``None``, takes the first dimension.
        The default is ``None``.
    ax : `cartopy.GeoAxes`, optional
        The matplotlib axes where to plot the image.
        If ``None``, a figure is initialized using the
        specified ``fig_kwargs``.
        The default is ``None``.
    add_colorbar : bool, optional
        Whether to add a colorbar. The default is ``True``.
    interpolation : str, optional
        Argument to be passed to imshow.
        The default is ``"nearest"``.
    fig_kwargs : dict, optional
        Figure options to be passed to `matplotlib.pyplot.subplots``.
        The default is ``None``.
        Only used if ``ax`` is ``None``.
    subplot_kwargs : dict, optional
        Subplot options to be passed to `matplotlib.pyplot.subplots`.
        The default is ``None``.
        Only used if ```ax``` is ``None``.
    cbar_kwargs : dict, optional
        Colorbar options. The default is ``None``.
    **plot_kwargs
        Additional arguments to be passed to the plotting function.
        Examples include ``cmap``, ``norm``, ``vmin``, ``vmax``, ``levels``, ...
        For FacetGrid plots, specify ``row``, ``col`` and ``col_wrap``.
        With ``rgb`` you can specify the name of the `xarray.DataArray` RGB dimension.


    """
    from gpm.checks import check_is_spatial_2d, is_spatial_2d

    # Plot orbit
    if not is_spatial_2d(da, strict=False):
        raise ValueError("Can not plot. It's not a spatial 2D object.")

    # Check inputs
    da = check_object_format(da, plot_kwargs=plot_kwargs, check_function=check_is_spatial_2d, strict=True)

    # Plot FacetGrid with xarray imshow
    if "col" in plot_kwargs or "row" in plot_kwargs:
        p = _plot_image_facetgrid(
            da=da,
            x=x,
            y=y,
            ax=ax,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    # Plot with xarray imshow
    else:
        p = _plot_image(
            da=da,
            x=x,
            y=y,
            ax=ax,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            fig_kwargs=fig_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    # Return mappable
    return p


####--------------------------------------------------------------------------.
##################
#### Plot map ####
##################


def plot_map(
    da,
    x=None,
    y=None,
    ax=None,
    add_colorbar=True,
    add_swath_lines=True,  # used only for GPM orbit objects
    add_background=True,
    interpolation="nearest",  # used only for GPM grid objects
    fig_kwargs=None,
    subplot_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot data on a geographic map.

    Parameters
    ----------
    da : `xr.DataArray`
        xarray DataArray.
    x : str, optional
        Longitude coordinate name.
        If ``None``, takes the second dimension.
        The default is ``None``.
    y : str, optional
        Latitude coordinate name.
        If ``None``, takes the first dimension.
        The default is ``None``.
    ax : `cartopy.GeoAxes`, optional
        The cartopy GeoAxes where to plot the map.
        If ``None``, a figure is initialized using the
        specified ``fig_kwargs`` and ``subplot_kwargs``.
        The default is ``None``.
    add_colorbar : bool, optional
        Whether to add a colorbar. The default is ``True``.
    add_swath_lines : bool, optional
        Whether to plot the swath sides with a dashed line. The default is ``True``.
        This argument only applies for ORBIT objects.
    add_background : bool, optional
        Whether to add the map background. The default is ``True``.
    interpolation : str, optional
        Argument to be passed to `matplotlib.pyplot.imshow`. Only applies for GRID objects.
        The default is ``"nearest"``.
    fig_kwargs : dict, optional
        Figure options to be passed to `matplotlib.pyplot.subplots`.
        The default is ``None``.
        Only used if ``ax`` is ``None``.
    subplot_kwargs : dict, optional
        Dictionary of keyword arguments for `matplotlib.pyplot.subplots`.
        Must contain the Cartopy CRS `'projection'` key if specified.
        The default is ``None``.
        Only used if ``ax`` is ``None``.
    cbar_kwargs : dict, optional
        Colorbar options. The default is ``None``.
    **plot_kwargs
        Additional arguments to be passed to the plotting function.
        Examples include ``cmap``, ``norm``, ``vmin``, ``vmax``, ``levels``, ...
        For FacetGrid plots, specify ``row``, ``col`` and ``col_wrap``.
        With ``rgb`` you can specify the name of the `xarray.DataArray` RGB dimension.


    """
    from gpm.checks import has_spatial_dim, is_grid, is_orbit, is_spatial_2d
    from gpm.visualization.grid import plot_grid_map
    from gpm.visualization.orbit import plot_orbit_map

    # Plot orbit
    # - allow vertical or other dimensions for FacetGrid
    # - allow to plot a swath of size 1 (i.e. nadir-looking)
    if is_orbit(da) and has_spatial_dim(da):
        p = plot_orbit_map(
            da=da,
            x=x,
            y=y,
            ax=ax,
            add_colorbar=add_colorbar,
            add_swath_lines=add_swath_lines,
            add_background=add_background,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    # Plot grid
    elif is_grid(da) and is_spatial_2d(da, strict=False):
        p = plot_grid_map(
            da=da,
            x=x,
            y=y,
            ax=ax,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            add_background=add_background,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    else:
        raise ValueError("Can not plot. It's neither a GPM GRID or GPM ORBIT spatial 2D object.")
    # Return mappable
    return p


def plot_map_mesh(
    xr_obj,
    x=None,
    y=None,
    ax=None,
    edgecolors="k",
    linewidth=0.1,
    add_background=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    **plot_kwargs,
):
    from gpm.checks import is_grid, is_orbit
    from gpm.visualization.grid import plot_grid_mesh
    from gpm.visualization.orbit import infer_orbit_xy_coords, plot_orbit_mesh

    # Plot orbit
    if is_orbit(xr_obj):
        x, y = infer_orbit_xy_coords(xr_obj, x=x, y=y)
        p = plot_orbit_mesh(
            da=xr_obj[y],
            ax=ax,
            x=x,
            y=y,
            edgecolors=edgecolors,
            linewidth=linewidth,
            add_background=add_background,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            **plot_kwargs,
        )
    elif is_grid(xr_obj):
        p = plot_grid_mesh(
            xr_obj=xr_obj,
            x=x,
            y=y,
            ax=ax,
            edgecolors=edgecolors,
            linewidth=linewidth,
            add_background=add_background,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            **plot_kwargs,
        )
    else:
        raise ValueError("Can not plot. It's neither a GPM GRID or GPM ORBIT spatial object.")
    # Return mappable
    return p


def plot_map_mesh_centroids(
    xr_obj,
    x=None,
    y=None,
    ax=None,
    c="r",
    s=1,
    add_background=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    **plot_kwargs,
):
    """Plot GPM orbit granule mesh centroids in a cartographic map."""
    from gpm.checks import is_grid, is_orbit
    from gpm.visualization.orbit import infer_orbit_xy_coords

    # Initialize figure if necessary
    ax = initialize_cartopy_plot(
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
        add_background=add_background,
    )

    # Retrieve orbits lon, lat coordinates
    if is_orbit(xr_obj):
        x, y = infer_orbit_xy_coords(xr_obj, x=x, y=y)

    # Retrieve grid centroids mesh
    if is_grid(xr_obj):
        x, y = infer_xy_labels(xr_obj, x=x, y=y)
        xr_obj = create_grid_mesh_data_array(xr_obj, x=x, y=y)

    # Extract numpy arrays
    lon = xr_obj[x].to_numpy()
    lat = xr_obj[y].to_numpy()

    # Plot centroids
    p = ax.scatter(lon, lat, transform=ccrs.PlateCarree(), c=c, s=s, **plot_kwargs)

    # Return mappable
    return p


def create_grid_mesh_data_array(xr_obj, x, y):
    """Create a 2D mesh coordinates DataArray.

    Takes as input the 1D coordinate arrays from an existing `xarray.DataArray` or `xarray.Dataset` object.

    The function creates a 2D grid (mesh) of x and y coordinates and initializes
    the data values to NaN.

    Parameters
    ----------
    xr_obj : `xarray.DataArray` or `xarray.Dataset`
        The input xarray object containing the 1D coordinate arrays.
    x : str
        The name of the x-coordinate in `xr_obj`.
    y : str
        The name of the y-coordinate in `xr_obj`.

    Returns
    -------
    da_mesh : `xarray.DataArray`
        A 2D `xarray.DataArray` with mesh coordinates for `x` and `y`, and NaN values for data points.

    Notes
    -----
    The resulting `xarray.DataArray` has dimensions named 'y' and 'x', corresponding to the
    y and x coordinates respectively.
    The coordinate values are taken directly from the input 1D coordinate arrays,
    and the data values are set to NaN.

    """
    # Extract 1D coordinate arrays
    x_coords = xr_obj[x].to_numpy()
    y_coords = xr_obj[y].to_numpy()

    # Create 2D meshgrid for x and y coordinates
    X, Y = np.meshgrid(x_coords, y_coords, indexing="xy")

    # Create a 2D array of NaN values with the same shape as the meshgrid
    dummy_values = np.full(X.shape, np.nan)

    # Create a new DataArray with 2D coordinates and NaN values
    return xr.DataArray(
        dummy_values,
        coords={x: (("y", "x"), X), y: (("y", "x"), Y)},
        dims=("y", "x"),
    )


####--------------------------------------------------------------------------.


def _plot_labels(
    xr_obj,
    label_name=None,
    max_n_labels=50,
    add_colorbar=True,
    interpolation="nearest",
    cmap="Paired",
    fig_kwargs=None,
    **plot_kwargs,
):
    """Plot labels.

    The maximum allowed number of labels to plot is 'max_n_labels'.
    """
    from ximage.labels.labels import get_label_indices, redefine_label_array
    from ximage.labels.plot_labels import get_label_colorbar_settings

    from gpm.visualization.plot import plot_image

    if isinstance(xr_obj, xr.Dataset):
        dataarray = xr_obj[label_name]
    else:
        dataarray = xr_obj[label_name] if label_name is not None else xr_obj

    dataarray = dataarray.compute()
    label_indices = get_label_indices(dataarray)
    n_labels = len(label_indices)
    if add_colorbar and n_labels > max_n_labels:
        msg = f"""The array currently contains {n_labels} labels
        and 'max_n_labels' is set to {max_n_labels}. The colorbar is not displayed!"""
        print(msg)
        add_colorbar = False
    # Relabel array from 1 to ... for plotting
    dataarray = redefine_label_array(dataarray, label_indices=label_indices)
    # Replace 0 with nan
    dataarray = dataarray.where(dataarray > 0)
    # Define appropriate colormap
    default_plot_kwargs, cbar_kwargs = get_label_colorbar_settings(label_indices, cmap=cmap)
    default_plot_kwargs.update(plot_kwargs)
    # Plot image
    return plot_image(
        dataarray,
        interpolation=interpolation,
        add_colorbar=add_colorbar,
        cbar_kwargs=cbar_kwargs,
        fig_kwargs=fig_kwargs,
        **default_plot_kwargs,
    )


def plot_labels(
    obj,  # Dataset, DataArray or generator
    label_name=None,
    max_n_labels=50,
    add_colorbar=True,
    interpolation="nearest",
    cmap="Paired",
    fig_kwargs=None,
    **plot_kwargs,
):
    if is_generator(obj):
        for _, xr_obj in obj:  # label_id, xr_obj
            p = _plot_labels(
                xr_obj=xr_obj,
                label_name=label_name,
                max_n_labels=max_n_labels,
                add_colorbar=add_colorbar,
                interpolation=interpolation,
                cmap=cmap,
                fig_kwargs=fig_kwargs,
                **plot_kwargs,
            )
            plt.show()
    else:
        p = _plot_labels(
            xr_obj=obj,
            label_name=label_name,
            max_n_labels=max_n_labels,
            add_colorbar=add_colorbar,
            interpolation=interpolation,
            cmap=cmap,
            fig_kwargs=fig_kwargs,
            **plot_kwargs,
        )
    return p


def plot_patches(
    patch_gen,
    variable=None,
    add_colorbar=True,
    interpolation="nearest",
    fig_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot patches."""
    from gpm.visualization.plot import plot_image

    # Plot patches
    for _, xr_patch in patch_gen:  # label_id, xr_obj
        if isinstance(xr_patch, xr.Dataset):
            if variable is None:
                raise ValueError("'variable' must be specified when plotting xarray.Dataset patches.")
            xr_patch = xr_patch[variable]
        try:
            plot_image(
                xr_patch,
                interpolation=interpolation,
                add_colorbar=add_colorbar,
                fig_kwargs=fig_kwargs,
                cbar_kwargs=cbar_kwargs,
                **plot_kwargs,
            )
            plt.show()
        except Exception:
            pass


####--------------------------------------------------------------------------.


def get_inset_bounds(
    ax,
    loc="upper right",
    inset_height=0.2,
    inside_figure=True,
    aspect_ratio=1,
):
    """Calculate the bounds for an inset axes in a matplotlib figure.

    This function computes the normalized figure coordinates for placing an inset axes within a figure,
    based on the specified location, size, and whether the inset should be fully inside the figure bounds.
    It is designed to be used with matplotlib figures to facilitate the addition of insets (e.g., for maps
    or zoomed plots) at predefined positions.

    Parameters
    ----------
    loc : str
        The location of the inset within the figure. Valid options are ``'lower left'``, ``'lower right'``,
        ``'upper left'``, and ``'upper right'``. The default is ``'upper right'``.
    inset_height : float
        The size of the inset height, specified as a fraction of the figure's height.
        For example, a value of 0.2 indicates that the inset's height will be 20% of the figure's height.
        The aspect ratio will govern the ``inset_width``.
    inside_figure : bool, optional
        Determines whether the inset is constrained to be fully inside the figure bounds. If  ``True`` (default),
        the inset is placed fully within the figure. If ``False``, the inset can extend beyond the figure's edges,
        allowing for a half-outside placement.
    aspect_ratio : float, optional
        The width-to-height ratio of the inset figure.
        A value greater than 1 indicates an inset figure wider than it is tall,
        and a value less than 1 indicates an inset figure taller than it is wide.
        The default value is 1.0, indicating a square inset figure.

    Returns
    -------
    inset_bounds : list of float
        The calculated bounds of the inset, in the format ``[x0, y0, width, height]``, where ``x0`` and ``y0``
        are the normalized figure coordinates of the lower left corner of the inset, and ``width`` and
        ``height`` are the normalized width and height of the inset, respectively.

    """
    # Get the bounding box of the parent axes in figure coordinates
    bbox = ax.get_position()
    parent_width = bbox.width
    parent_height = bbox.height

    # Compute the inset width percentage (relative to the parent axes)
    # - Take into account possible different aspect ratios
    inset_height_abs = inset_height * parent_height
    inset_width_abs = inset_height_abs * aspect_ratio
    inset_width = inset_width_abs / parent_width
    loc_mapping = {
        "upper right": (1 - inset_width, 1 - inset_height),
        "upper left": (0, 1 - inset_height),
        "lower right": (1 - inset_width, 0),
        "lower left": (0, 0),
    }
    inset_x, inset_y = loc_mapping[loc]

    # Adjust for insets that are allowed to be half outside of the figure
    if not inside_figure:
        inset_x += inset_width / 2 * (-1 if loc.endswith("left") else 1)
        inset_y += inset_height / 2 * (-1 if loc.startswith("lower") else 1)

    return [inset_x, inset_y, inset_width, inset_height]


def add_map_inset(ax, loc="upper left", inset_height=0.2, projection=None, inside_figure=True):
    """Adds an inset map to a matplotlib axis using Cartopy, highlighting the extent of the main plot.

    This function creates a smaller map inset within a larger map plot to show a global view or
    contextual location of the main plot's extent.

    It uses Cartopy for map projections and plotting, and it outlines the extent of the main plot
    within the inset to provide geographical context.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes` or `cartopy.mpl.geoaxes.GeoAxes`
        The main matplotlib or cartopy axis object where the geographic data is plotted.
    loc : str, optional
        The location of the inset map within the main plot.
        Options include ``'lower left'``, ``'lower right'``,
        ``'upper left'``, and ``'upper right'``. The default is ``'upper left'``.
    inset_height : float, optional
        The size of the inset height, specified as a fraction of the figure's height.
        For example, a value of 0.2 indicates that the inset's height will be 20% of the figure's height.
        The aspect ratio (of the map inset) will govern the ``inset_width``.
    inside_figure : bool, optional
        Determines whether the inset is constrained to be fully inside the figure bounds. If ``True`` (default),
        the inset is placed fully within the figure. If ``False``, the inset can extend beyond the figure's edges,
        allowing for a half-outside placement.
    projection: `cartopy.crs.Projection`, optional
        A cartopy projection. If ``None``, am Orthographic projection centered on the extent center is used.

    Returns
    -------
    ax2 : `cartopy.mpl.geoaxes.GeoAxes`
        The Cartopy GeoAxesSubplot object for the inset map.

    Notes
    -----
    The function adjusts the extent of the inset map based on the main plot's extent, adding a
    slight padding for visual clarity. It then overlays a red outline indicating the main plot's
    geographical extent.

    Examples
    --------
    >>> p = da.gpm.plot_map()
    >>> add_map_inset(ax=p.axes, loc="upper left", inset_height=0.15)

    This example creates a main plot with a specified extent and adds an upper-left inset map
    showing the global context of the main plot's extent.

    """
    from shapely import Polygon

    from gpm.utils.geospatial import extend_geographic_extent

    # Retrieve extent and bounds
    extent = ax.get_extent()
    extent = extend_geographic_extent(extent, padding=0.5)
    bounds = [extent[i] for i in [0, 2, 1, 3]]
    # Create Cartopy Polygon
    polygon = Polygon.from_bounds(*bounds)
    # Define Orthographic projection
    if projection is None:
        lon_min, lon_max, lat_min, lat_max = extent
        projection = ccrs.Orthographic(
            central_latitude=(lat_min + lat_max) / 2,
            central_longitude=(lon_min + lon_max) / 2,
        )

    # Define aspect ratio of the map inset
    aspect_ratio = float(np.diff(projection.x_limits) / np.diff(projection.y_limits).item())

    # Define inset location relative to main plot (ax) in normalized units
    # - Lower-left corner of inset Axes, and its width and height
    # - [x0, y0, width, height]
    inset_bounds = get_inset_bounds(
        ax=ax,
        loc=loc,
        inset_height=inset_height,
        inside_figure=inside_figure,
        aspect_ratio=aspect_ratio,
    )

    # ax2 = plt.axes(inset_bounds, projection=projection)
    ax2 = ax.inset_axes(
        inset_bounds,
        projection=projection,
    )

    # Add global map
    ax2.set_global()
    ax2.add_feature(cfeature.LAND)
    ax2.add_feature(cfeature.OCEAN)
    # Add extent polygon
    _ = ax2.add_geometries(
        [polygon],
        ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="red",
        linewidth=0.3,
    )
    return ax2
