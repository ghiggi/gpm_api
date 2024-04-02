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
from scipy.interpolate import griddata

import gpm


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

    If  mask_data=True, data values with invalid coordinates are masked
    and a numpy masked array is returned.
    Masked data values are not displayed in pcolormesh !
    If rgb=True, it assumes the RGB dimension is the last data dimension.

    """
    # Retrieve mask of invalid coordinates
    x_invalid = ~np.isfinite(x)
    y_invalid = ~np.isfinite(y)
    mask = np.logical_or(x_invalid, y_invalid)

    # If no invalid coordinates, return original data
    if np.all(~mask):
        return x, y, data

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
    if np.any(x_invalid):
        x = _interpolate_data(x, method="linear")  # interpolation
        x = _interpolate_data(x, method="nearest")  # nearest neighbours outside the convex hull
    if np.any(y_invalid):
        y = _interpolate_data(y, method="linear")  # interpolation
        y = _interpolate_data(y, method="nearest")  # nearest neighbours outside the convex hull
    return x, y, data_masked


def _interpolate_data(arr, method="linear"):
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


####--------------------------------------------------------------------------.


def preprocess_figure_args(ax, fig_kwargs=None, subplot_kwargs=None):
    fig_kwargs = {} if fig_kwargs is None else fig_kwargs
    subplot_kwargs = {} if subplot_kwargs is None else subplot_kwargs
    if ax is not None:
        if len(subplot_kwargs) >= 1:
            raise ValueError("Provide `subplot_kwargs`only if `ax`is None")
        if len(fig_kwargs) >= 1:
            raise ValueError("Provide `fig_kwargs` only if `ax`is None")
    return fig_kwargs


def preprocess_subplot_kwargs(subplot_kwargs):
    subplot_kwargs = {} if subplot_kwargs is None else subplot_kwargs
    subplot_kwargs = subplot_kwargs.copy()
    if "projection" not in subplot_kwargs:
        subplot_kwargs["projection"] = ccrs.PlateCarree()
    return subplot_kwargs


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


####--------------------------------------------------------------------------.


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


def plot_colorbar(p, ax, cbar_kwargs=None):
    """Add a colorbar to a matplotlib/cartopy plot.

    cbar_kwargs 'size' and 'pad' controls the size of the colorbar.
    and the padding between the plot and the colorbar.

    p: matplotlib.image.AxesImage
    ax:  cartopy.mpl.geoaxes.GeoAxesSubplot^
    """
    cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
    cbar_kwargs = cbar_kwargs.copy()  # otherwise pop ticklabels outside the function
    ticklabels = cbar_kwargs.pop("ticklabels", None)
    orientation = cbar_kwargs.get("orientation", "vertical")

    divider = make_axes_locatable(ax)

    if orientation == "vertical":
        size = cbar_kwargs.get("size", "5%")
        pad = cbar_kwargs.get("pad", 0.1)
        cax = divider.append_axes("right", size=size, pad=pad, axes_class=plt.Axes)
    elif orientation == "horizontal":
        size = cbar_kwargs.get("size", "5%")
        pad = cbar_kwargs.get("pad", 0.25)
        cax = divider.append_axes("bottom", size=size, pad=pad, axes_class=plt.Axes)
    else:
        raise ValueError("Invalid orientation. Choose 'vertical' or 'horizontal'.")

    p.figure.add_axes(cax)
    cbar = plt.colorbar(p, cax=cax, ax=ax, **cbar_kwargs)
    if ticklabels is not None:
        _ = cbar.ax.set_yticklabels(ticklabels) if orientation == "vertical" else cbar.ax.set_xticklabels(ticklabels)
    return cbar


####--------------------------------------------------------------------------.


def get_dataarray_extent(da, x="lon", y="lat"):
    # TODO: compute corners array to estimate the extent
    # - OR increase by 1° in everydirection and then wrap between -180, 180,90,90
    # Get the minimum and maximum longitude and latitude values
    lon_min, lon_max = da[x].min(), da[x].max()
    lat_min, lat_max = da[y].min(), da[y].max()
    return (lon_min, lon_max, lat_min, lat_max)


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


def _plot_cartopy_imshow(
    ax,
    da,
    x,
    y,
    interpolation="nearest",
    add_colorbar=True,
    plot_kwargs={},
    cbar_kwargs=None,
):
    """Plot imshow with cartopy."""
    # - Ensure image with correct dimensions orders
    da = da.transpose(y, x)
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
    p = ax.imshow(
        arr,
        transform=ccrs.PlateCarree(),
        extent=extent,
        origin=origin,
        interpolation=interpolation,
        **plot_kwargs,
    )
    # - Set the extent
    extent = get_dataarray_extent(da, x="lon", y="lat")
    ax.set_extent(extent)

    # - Add colorbar
    if add_colorbar:
        # --> TODO: set axis proportion in a meaningful way ...
        _ = plot_colorbar(p=p, ax=ax, cbar_kwargs=cbar_kwargs)
    return p


def _mask_antimeridian_crossing_arr(arr, antimeridian_mask, rgb):
    if np.ma.is_masked(arr):
        if rgb:
            data_mask = np.broadcast_to(np.expand_dims(antimeridian_mask, axis=-1), arr.shape)
            combined_mask = np.logical_or(data_mask, antimeridian_mask)
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


def _plot_cartopy_pcolormesh(
    ax,
    da,
    x,
    y,
    rgb=False,
    add_colorbar=True,
    add_swath_lines=True,
    plot_kwargs={},
    cbar_kwargs=None,
):
    """Plot imshow with cartopy.

    The function currently does not allow to zoom on regions across the antimeridian.
    The function mask scanning pixels which spans across the antimeridian.
    If rgb=True, expect rgb dimension to be at last position.
    x and y must represents longitude and latitudes.
    """
    # Get x, y, and array to plot
    da = da.compute()
    lon = da[x].data
    lat = da[y].data
    arr = da.data

    # If RGB, expect last dimension to have 3 channels
    if rgb and arr.shape[-1] != 3 and arr.shape[-1] != 4:
        raise ValueError("RGB array must have 3 or 4 channels in the last dimension.")

    # Infill invalid value and add mask if necessary
    lon, lat, arr = get_valid_pcolormesh_inputs(lon, lat, arr, rgb=rgb)

    # Ensure arguments
    if rgb:
        add_colorbar = False

    # Compute coordinates of cell corners for pcolormesh quadrilateral mesh
    # - This enable correct masking of cells crossing the antimeridian
    from gpm.utils.area import _get_lonlat_corners

    lon, lat = _get_lonlat_corners(lon, lat)

    # Mask cells crossing the antimeridian
    # --> Here we assume not invalid coordinates anymore
    # --> Cartopy still bugs with several projections when data cross the antimeridian
    # --> This flag can be unset with gpm.config.set({"viz_hide_antimeridian_data": False})
    if gpm.config.get("viz_hide_antimeridian_data"):
        antimeridian_mask = get_antimeridian_mask(lon)
        is_crossing_antimeridian = np.any(antimeridian_mask)
        if is_crossing_antimeridian:
            arr = _mask_antimeridian_crossing_arr(arr, antimeridian_mask=antimeridian_mask, rgb=rgb)

            # Sanitize cmap bad color to avoid cartopy bug
            # - TODO cartopy requires bad_color to be transparent ...
            cmap = plot_kwargs.get("cmap", None)
            if cmap is not None:
                bad = cmap.get_bad()
                bad[3] = 0  # enforce to 0 (transparent)
                cmap.set_bad(bad)
                plot_kwargs["cmap"] = cmap

    # Add variable field with cartopy
    p = ax.pcolormesh(
        lon,
        lat,
        arr,
        transform=ccrs.PlateCarree(),
        **plot_kwargs,
    )

    # Add swath lines
    if add_swath_lines:
        sides = [(lon[0, :], lat[0, :]), (lon[-1, :], lat[-1, :])]
        plot_sides(sides=sides, ax=ax, linestyle="--", color="black")

    # Add colorbar
    # --> TODO: set axis proportion in a meaningful way ...
    if add_colorbar:
        _ = plot_colorbar(p=p, ax=ax, cbar_kwargs=cbar_kwargs)
    return p


def _plot_mpl_imshow(
    ax,
    da,
    x,
    y,
    interpolation="nearest",
    add_colorbar=True,
    plot_kwargs={},
    cbar_kwargs=None,
):
    """Plot imshow with matplotlib."""
    # - Ensure image with correct dimensions orders
    da = da.transpose(y, x)
    arr = np.asanyarray(da.data)

    # - Add variable field with matplotlib
    p = ax.imshow(
        arr,
        origin="upper",
        interpolation=interpolation,
        **plot_kwargs,
    )
    # - Add colorbar
    if add_colorbar:
        # --> TODO: set axis proportion in a meaningful way ...
        _ = plot_colorbar(p=p, ax=ax, cbar_kwargs=cbar_kwargs)
    # - Return mappable
    return p


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


def _plot_xr_imshow(
    ax,
    da,
    x,
    y,
    interpolation="nearest",
    add_colorbar=True,
    plot_kwargs={},
    cbar_kwargs=None,
    visible_colorbar=True,
):
    """Plot imshow with xarray.

    The colorbar is added with xarray to enable to display multiple colorbars
    when calling this function multiple times on different fields with
    different colorbars.
    """
    # --> BUG with colorbar: https://github.com/pydata/xarray/issues/7014
    ticklabels = cbar_kwargs.pop("ticklabels", None)
    if not add_colorbar:
        cbar_kwargs = None
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


def _plot_xr_pcolormesh(
    ax,
    da,
    x,
    y,
    add_colorbar=True,
    plot_kwargs={},
    cbar_kwargs=None,
):
    """Plot pcolormesh with xarray."""
    ticklabels = cbar_kwargs.pop("ticklabels", None)
    if not add_colorbar:
        cbar_kwargs = None
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


####--------------------------------------------------------------------------.


def plot_map(
    da,
    x="lon",
    y="lat",
    ax=None,
    add_colorbar=True,
    add_swath_lines=True,  # used only for GPM orbit objects
    add_background=True,
    rgb=False,
    interpolation="nearest",  # used only for GPM grid objects
    fig_kwargs=None,
    subplot_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot data on a geographic map.

    Parameters
    ----------
    da : xr.DataArray
        xarray DataArray.
    x : str, optional
        Longitude coordinate name. The default is `"lon"`.
    y : str, optional
        Latitude coordinate name. The default is `"lat"`.
    ax : cartopy.GeoAxes, optional
        The cartopy GeoAxes where to plot the map.
        If `None`, a figure is initialized using the
        specified `fig_kwargs`and `subplot_kwargs`.
        The default is `None`.
    add_colorbar : bool, optional
        Whether to add a colorbar. The default is `True`.
    add_swath_lines : bool, optional
        Whether to plot the swath sides with a dashed line. The default is `True`.
        This argument only applies for ORBIT objects.
    add_background : bool, optional
        Whether to add the map background. The default is `True`.
    rgb : bool, optional
        Whether the input DataArray has a rgb dimension. The default is `False`.
    interpolation : str, optional
        Argument to be passed to imshow. Only applies for GRID objects.
        The default is `"nearest"`.
    fig_kwargs : dict, optional
        Figure options to be passed to plt.subplots.
        The default is `None`.
        Only used if `ax` is `None`.
    subplot_kwargs : dict, optional
        Dictionary of keyword arguments for Matplotlib subplots.
        Must contain the Cartopy CRS `'projection'` key if specified.
        The default is `None`.
        Only used if `ax` is `None`.
    cbar_kwargs : dict, optional
        Colorbar options. The default is `None`.
    **plot_kwargs
        Additional arguments to be passed to the plotting function.
        Examples include `cmap`, `norm`, `vmin`, `vmax`, `levels`, ...
        For FacetGrid plots, specify `row`, `col` and `col_wrap`.

    """
    from gpm.checks import is_grid, is_orbit
    from gpm.visualization.grid import plot_grid_map
    from gpm.visualization.orbit import plot_orbit_map

    # Plot orbit
    if is_orbit(da):
        p = plot_orbit_map(
            da=da,
            x=x,
            y=y,
            ax=ax,
            add_colorbar=add_colorbar,
            add_swath_lines=add_swath_lines,
            add_background=add_background,
            rgb=rgb,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    # Plot grid
    elif is_grid(da):
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
        raise ValueError("Can not plot. It's neither a GPM grid, neither a GPM orbit.")
    # Return mappable
    return p


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
    da : xr.DataArray
        xarray DataArray.
    x : str, optional
        X dimension name.
        If ``None``, takes the second dimension.
        The default is `None`.
    y : str, optional
        Y dimension name.
        If ``None``, takes the first dimension.
        The default is `None`.
    ax : cartopy.GeoAxes, optional
        The matplotlib axes where to plot the image.
        If ``None``, a figure is initialized using the
        specified `fig_kwargs`.
        The default is `None`.
    add_colorbar : bool, optional
        Whether to add a colorbar. The default is `True`.
    interpolation : str, optional
        Argument to be passed to imshow.
        The default is `"nearest"`.
    fig_kwargs : dict, optional
        Figure options to be passed to `plt.subplots`.
        The default is ``None``.
        Only used if `ax` is None.
    subplot_kwargs : dict, optional
        Subplot options to be passed to `plt.subplots`.
        The default is `None`.
        Only used if `ax` is `None`.
    cbar_kwargs : dict, optional
        Colorbar options. The default is `None`.
    **plot_kwargs
        Additional arguments to be passed to the plotting function.
        Examples include `cmap`, `norm`, `vmin`, `vmax`, `levels`, ...
        For FacetGrid plots, specify `row`, `col` and `col_wrap`.

    """
    # figsize, dpi, subplot_kw only used if ax is None
    from gpm.checks import is_grid, is_orbit
    from gpm.visualization.grid import plot_grid_image
    from gpm.visualization.orbit import plot_orbit_image

    # Plot orbit
    if is_orbit(da):
        p = plot_orbit_image(
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
    # Plot grid
    elif is_grid(da):
        p = plot_grid_image(
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
    else:
        raise ValueError("Can not plot. It's neither a GPM GRID, neither a GPM ORBIT.")
    # Return mappable
    return p


####--------------------------------------------------------------------------.


def create_grid_mesh_data_array(xr_obj, x, y):
    """Create a 2D mesh coordinates DataArray.

    Takes as input the 1D coordinate arrays from an existing xarray object (Dataset or DataArray).

    The function creates a 2D grid (mesh) of x and y coordinates and initializes
    the data values to NaN.

    Parameters
    ----------
    xr_obj : xarray.DataArray or xarray.Dataset
        The input xarray object containing the 1D coordinate arrays.
    x : str
        The name of the x-coordinate in `xr_obj`.
    y : str
        The name of the y-coordinate in `xr_obj`.

    Returns
    -------
    da_mesh : xarray.DataArray
        A 2D xarray DataArray with mesh coordinates for `x` and `y`, and NaN values for data points.

    Notes
    -----
    The resulting DataArray has dimensions named 'y' and 'x', corresponding to the y and x coordinates respectively.
    The coordinate values are taken directly from the input 1D coordinate arrays, and the data values are set to NaN.

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


def plot_map_mesh(
    xr_obj,
    x="lon",
    y="lat",
    ax=None,
    edgecolors="k",
    linewidth=0.1,
    add_background=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    **plot_kwargs,
):
    from gpm.checks import is_orbit  # is_grid

    from .grid import plot_grid_mesh
    from .orbit import plot_orbit_mesh

    # Plot orbit
    if is_orbit(xr_obj):
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
    else:  # Plot grid
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
    # Return mappable
    return p


def plot_map_mesh_centroids(
    xr_obj,
    x="lon",
    y="lat",
    ax=None,
    c="r",
    s=1,
    add_background=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    **plot_kwargs,
):
    """Plot GPM orbit granule mesh centroids in a cartographic map."""
    from gpm.checks import is_grid

    # - Initialize figure if necessary
    ax = initialize_cartopy_plot(
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
        add_background=add_background,
    )

    # - Retrieve centroids
    if is_grid(xr_obj):
        xr_obj = create_grid_mesh_data_array(xr_obj, x=x, y=y)
    lon = xr_obj[x].to_numpy()
    lat = xr_obj[y].to_numpy()

    # - Plot centroids
    return ax.scatter(lon, lat, transform=ccrs.PlateCarree(), c=c, s=s, **plot_kwargs)

    # - Return mappable


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
                raise ValueError("'variable' must be specified when plotting xr.Dataset patches.")
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
        The location of the inset within the figure. Valid options are 'lower left', 'lower right',
        'upper left', and 'upper right'. The default is 'upper right'.
    inset_height : float
        The size of the inset height, specified as a fraction of the figure's height.
        For example, a value of 0.2 indicates that the inset's height will be 20% of the figure's height.
        The aspect ratio will govern the inset_width.
    inside_figure : bool, optional
        Determines whether the inset is constrained to be fully inside the figure bounds. If `True` (default),
        the inset is placed fully within the figure. If `False`, the inset can extend beyond the figure's edges,
        allowing for a half-outside placement.
    aspect_ratio : float, optional
        The width-to-height ratio of the inset figure.
        A value greater than 1 indicates an inset figure wider than it is tall,
        and a value less than 1 indicates an inset figure taller than it is wide.
        The default value is 1.0, indicating a square inset figure.

    Returns
    -------
    inset_bounds : list of float
        The calculated bounds of the inset, in the format [x0, y0, width, height], where `x0` and `y0`
        are the normalized figure coordinates of the lower left corner of the inset, and `width` and
        `height` are the normalized width and height of the inset, respectively.

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
    ax : (matplotlib.axes.Axes, cartopy.mpl.geoaxes.GeoAxes)
        The main matplotlib or cartopy axis object where the geographic data is plotted.
    loc : str, optional
        The location of the inset map within the main plot.
        Options include 'lower left', 'lower right', 'upper left', 'upper right'.
        The default is 'upper left'.
    inset_height : float
        The size of the inset height, specified as a fraction of the figure's height.
        For example, a value of 0.2 indicates that the inset's height will be 20% of the figure's height.
        The aspect ratio (of the map inset) will govern the inset_width.
    inside_figure : bool, optional
        Determines whether the inset is constrained to be fully inside the figure bounds. If `True` (default),
        the inset is placed fully within the figure. If `False`, the inset can extend beyond the figure's edges,
        allowing for a half-outside placement.
    projection: cartopy.crs.Projection
        A cartopy projection. If ``None``, am Orthographic projection centered on the extent center is used.

    Returns
    -------
    ax2 : cartopy.mpl.geoaxes.GeoAxes
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
