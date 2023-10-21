#!/usr/bin/env python3
"""
Created on Sat Dec 10 18:42:28 2022

@author: ghiggi
"""
import inspect

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

### TODO: Add xarray + cartopy  (xr_carto) (xr_mpl)
# _plot_cartopy_xr_imshow
# _plot_cartopy_xr_pcolormesh


def is_generator(obj):
    return inspect.isgeneratorfunction(obj) or inspect.isgenerator(obj)


def _preprocess_figure_args(ax, fig_kwargs={}, subplot_kwargs={}):
    if ax is not None:
        if len(subplot_kwargs) >= 1:
            raise ValueError("Provide `subplot_kwargs`only if `ax`is None")
        if len(fig_kwargs) >= 1:
            raise ValueError("Provide `fig_kwargs` only if `ax`is None")

    # If ax is not specified, specify the figure defaults
    # if ax is None:
    # Set default figure size and dpi
    # fig_kwargs['figsize'] = (12, 10)
    # fig_kwargs['dpi'] = 100


def _preprocess_subplot_kwargs(subplot_kwargs):
    subplot_kwargs = subplot_kwargs.copy()
    if "projection" not in subplot_kwargs:
        subplot_kwargs["projection"] = ccrs.PlateCarree()
    return subplot_kwargs


def get_extent(da, x="lon", y="lat"):
    # TODO: compute corners array to estimate the extent
    # - OR increase by 1° in everydirection and then wrap between -180, 180,90,90
    # Get the minimum and maximum longitude and latitude values
    lon_min, lon_max = da[x].min(), da[x].max()
    lat_min, lat_max = da[y].min(), da[y].max()
    extent = (lon_min, lon_max, lat_min, lat_max)
    return extent


def get_antimeridian_mask(lons, buffer=True):
    """Get mask of longitude coordinates neighbors crossing the antimeridian."""
    from scipy.ndimage import binary_dilation

    # Check vertical edges
    row_idx, col_idx = np.where(np.abs(np.diff(lons, axis=0)) > 180)
    row_idx_rev, col_idx_rev = np.where(np.abs(np.diff(lons[::-1, :], axis=0)) > 180)
    row_idx_rev = lons.shape[0] - row_idx_rev - 1
    row_indices = np.append(row_idx, row_idx_rev)
    col_indices = np.append(col_idx, col_idx_rev)
    # Check horizontal
    row_idx, col_idx = np.where(np.abs(np.diff(lons, axis=1)) > 180)
    row_idx_rev, col_idx_rev = np.where(np.abs(np.diff(lons[:, ::-1], axis=1)) > 180)
    col_idx_rev = lons.shape[1] - col_idx_rev - 1
    row_indices = np.append(row_indices, np.append(row_idx, row_idx_rev))
    col_indices = np.append(col_indices, np.append(col_idx, col_idx_rev))
    # Create mask
    mask = np.zeros(lons.shape)
    mask[row_indices, col_indices] = 1
    # Buffer by 1 in all directions to ensure edges not crossing the antimeridian
    mask = binary_dilation(mask)
    return mask


def get_masked_cells_polycollection(x, y, arr, mask, plot_kwargs):
    from scipy.ndimage import binary_dilation

    from gpm_api.utils.area import _from_corners_to_bounds, _get_lonlat_corners, is_vertex_clockwise

    # - Buffer mask by 1 to derive vertices of all masked QuadMesh
    mask = binary_dilation(mask)

    # - Get index of masked quadmesh
    row_mask, col_mask = np.where(mask)

    # - Retrieve values of masked cells
    array = arr[row_mask, col_mask]

    # - Retrieve QuadMesh corners (m+1 x n+1)
    x_corners, y_corners = _get_lonlat_corners(x, y)

    # - Retrieve QuadMesh bounds (m*n x 4)
    x_bounds = _from_corners_to_bounds(x_corners)
    y_bounds = _from_corners_to_bounds(y_corners)

    # - Retrieve vertices of masked QuadMesh (n_masked, 4, 2)
    x_vertices = x_bounds[row_mask, col_mask]
    y_vertices = y_bounds[row_mask, col_mask]

    vertices = np.stack((x_vertices, y_vertices), axis=2)

    # Check that are counterclockwise oriented (check first vertex)
    # TODO: this check should be updated to use pyresample.future.spherical
    if is_vertex_clockwise(vertices[0, :, :]):
        vertices = vertices[:, ::-1, :]

    # - Define additional kwargs for PolyCollection
    plot_kwargs = plot_kwargs.copy()
    if "edgecolors" not in plot_kwargs:
        plot_kwargs["edgecolors"] = "face"  # 'none'
    if "linewidth" not in plot_kwargs:
        plot_kwargs["linewidth"] = 0
    plot_kwargs["antialiaseds"] = False  # to better plotting quality

    # - Define PolyCollection
    coll = PolyCollection(
        verts=vertices,
        array=array,
        transform=ccrs.Geodetic(),
        **plot_kwargs,
    )
    return coll


def get_valid_pcolormesh_inputs(x, y, data, rgb=False):
    """
    Fill non-finite values with neighbour valid coordinates.

    pcolormesh does not accept non-finite values in the coordinates.
    This function:
    - Infill NaN/Inf in lat/x with closest values
    - Mask the corresponding pixels in the data that must not be displayed.

    If RGB=True, the RGB channels is in the last dimension
    """
    # TODO:
    # - Instead of np.interp, can use nearest neighbors or just 0 to speed up?

    # Retrieve mask of invalid coordinates
    mask = np.logical_or(~np.isfinite(x), ~np.isfinite(y))

    # If no invalid coordinates, return original data
    if np.all(~mask):
        return x, y, data

    # Dilate mask
    # mask = dilation(mask, square(2))

    # Mask the data
    if rgb:
        data_mask = np.broadcast_to(np.expand_dims(mask, axis=-1), data.shape)
        data_masked = np.ma.masked_where(data_mask, data)
    else:
        data_masked = np.ma.masked_where(mask, data)

    # TODO: should be done in XYZ?
    x_dummy = x.copy()
    x_dummy[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
    y_dummy = y.copy()
    y_dummy[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
    return x_dummy, y_dummy, data_masked


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


def plot_colorbar(p, ax, cbar_kwargs={}, size="5%", pad=0.1):
    """Add a colorbar to a matplotlib/cartopy plot.

    p: matplotlib.image.AxesImage
    ax:  cartopy.mpl.geoaxes.GeoAxesSubplot
    """
    cbar_kwargs = cbar_kwargs.copy()  # otherwise pop ticklabels outside the function
    ticklabels = cbar_kwargs.pop("ticklabels", None)
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size=size, pad=pad, axes_class=plt.Axes)

    p.figure.add_axes(cax)
    cbar = plt.colorbar(p, cax=cax, ax=ax, **cbar_kwargs)
    if ticklabels is not None:
        _ = cbar.ax.set_yticklabels(ticklabels)
    return cbar


####--------------------------------------------------------------------------.
def _plot_cartopy_imshow(
    ax,
    da,
    x,
    y,
    interpolation="nearest",
    add_colorbar=True,
    plot_kwargs={},
    cbar_kwargs={},
):
    """Plot imshow with cartopy."""
    # TODO: allow to plot whatever projection (based on CRS) !
    # TODO: allow to plot subset of PlateeCarree !

    # - Ensure image with correct dimensions orders
    da = da.transpose(y, x)
    arr = np.asanyarray(da.data)

    # - Derive extent
    extent = [-180, 180, -90, 90]  # TODO: Derive from data !!!!

    # TODO: ensure y data is increasing --> origin = "lower"
    # TODO: ensure y data is decreasing --> origin = "upper"

    # - Add variable field with cartopy
    p = ax.imshow(
        arr,
        transform=ccrs.PlateCarree(),
        extent=extent,
        origin="lower",
        interpolation=interpolation,
        **plot_kwargs,
    )
    # - Set the extent
    extent = get_extent(da, x="lon", y="lat")
    ax.set_extent(extent)

    # - Add colorbar
    if add_colorbar:
        # --> TODO: set axis proportion in a meaningful way ...
        _ = plot_colorbar(p=p, ax=ax, cbar_kwargs=cbar_kwargs)
    return p


def _plot_rgb_pcolormesh(x, y, image, ax, **kwargs):
    """Plot xarray RGB DataArray with non uniform-coordinates.

    Matplotlib, cartopy and xarray pcolormesh currently does not support RGB(A) arrays.
    This is a temporary workaround !
    """
    if image.shape[2] not in [3, 4]:
        raise ValueError("Expecting RGB or RGB(A) arrays.")

    colorTuple = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
    im = ax.pcolormesh(
        x,
        y,
        image[:, :, 1],  # dummy to work ...
        color=colorTuple,
        **kwargs,
    )
    # im.set_array(None)
    return im


def _plot_cartopy_pcolormesh(
    ax,
    da,
    x,
    y,
    rgb=False,
    add_colorbar=True,
    plot_kwargs={},
    cbar_kwargs={},
):
    """Plot imshow with cartopy.

    The function currently does not allow to zoom on regions across the antimeridian.
    The function mask scanning pixels which spans across the antimeridian.
    If rgb=True, expect rgb dimension to be at last position.
    x and y must represents longitude and latitudes.
    """
    # - Get x, y, and array to plot
    da = da.compute()
    x = da[x].data
    y = da[y].data
    arr = da.data

    # - Infill invalid value and add mask if necessary
    x, y, arr = get_valid_pcolormesh_inputs(x, y, arr, rgb=rgb)
    # - Ensure arguments
    if rgb:
        add_colorbar = False

    # - Mask cells crossing the antimeridian
    # --> Here assume not invalid coordinates anymore
    antimeridian_mask = get_antimeridian_mask(x, buffer=True)
    is_crossing_antimeridian = np.any(antimeridian_mask)
    if is_crossing_antimeridian:
        if np.ma.is_masked(arr):
            if rgb:
                data_mask = np.broadcast_to(np.expand_dims(antimeridian_mask, axis=-1), arr.shape)
                combined_mask = np.logical_or(data_mask, antimeridian_mask)
            else:
                combined_mask = np.logical_or(arr.mask, antimeridian_mask)
            arr = np.ma.masked_where(combined_mask, arr)
        else:
            arr = np.ma.masked_where(antimeridian_mask, arr)
        # Sanitize cmap bad color to avoid cartopy bug
        if "cmap" in plot_kwargs:
            cmap = plot_kwargs["cmap"]
            bad = cmap.get_bad()
            bad[3] = 0  # enforce to 0 (transparent)
            cmap.set_bad(bad)
            plot_kwargs["cmap"] = cmap

    # - Add variable field with cartopy
    if not rgb:
        p = ax.pcolormesh(
            x,
            y,
            arr,
            transform=ccrs.PlateCarree(),
            **plot_kwargs,
        )
        # - Add PolyCollection of QuadMesh cells crossing the antimeridian
        if is_crossing_antimeridian:
            coll = get_masked_cells_polycollection(
                x, y, arr.data, mask=antimeridian_mask, plot_kwargs=plot_kwargs
            )
            p.axes.add_collection(coll)

    # - Add RGB
    else:
        p = _plot_rgb_pcolormesh(x, y, arr, ax=ax, **plot_kwargs)
        if is_crossing_antimeridian:
            plot_kwargs["facecolors"] = arr.reshape((arr.shape[0] * arr.shape[1], arr.shape[2]))
            coll = get_masked_cells_polycollection(
                x, y, arr.data, mask=antimeridian_mask, plot_kwargs=plot_kwargs
            )
            p.axes.add_collection(coll)

    # - Set the extent
    # --> To be set in projection coordinates of crs !!!
    #     lon/lat conversion to proj required !
    # extent = get_extent(da, x="lon", y="lat")
    # ax.set_extent(extent)

    # - Add colorbar
    if add_colorbar:
        # --> TODO: set axis proportion in a meaningful way ...
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
    cbar_kwargs={},
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


# def _get_colorbar_inset_axes_kwargs(p):
#     from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#     colorbar_axes = p.colorbar.ax

#     # Get the position and size of the colorbar axes in figure coordinates
#     bbox = colorbar_axes.get_position()

#     # Extract the width and height of the colorbar axes in figure coordinates
#     width = bbox.x1 - bbox.x0
#     height = bbox.y1 - bbox.y0

#     # Get the location of the colorbar axes ('upper', 'lower', 'center', etc.)
#     # This information will be used to set the 'loc' parameter of inset_axes
#     loc = 'upper right'  # Modify this according to your preference

#     # Get the transformation of the colorbar axes with respect to the image axes
#     # This information will be used to set the 'bbox_transform' parameter of inset_axes
#     bbox_transform = colorbar_axes.get_transform()

#     # Calculate the coordinates of the colorbar axes relative to the image axes
#     x0, y0 = bbox_transform.transform((bbox.x0, bbox.y0))
#     x1, y1 = bbox_transform.transform((bbox.x1, bbox.y1))
#     bbox_to_anchor = (x0, y0, x1 - x0, y1 - y0)


def set_colorbar_fully_transparent(p):
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # colorbar_axes = p.colorbar.ax

    # # Get the position and size of the colorbar axes in figure coordinates
    # bbox = colorbar_axes.get_position()

    # # Extract the width and height of the colorbar axes in figure coordinates
    # width = bbox.x1 - bbox.x0
    # height = bbox.y1 - bbox.y0

    # # Get the location of the colorbar axes ('upper', 'lower', 'center', etc.)
    # # This information will be used to set the 'loc' parameter of inset_axes
    # loc = 'upper right'  # Modify this according to your preference

    # # Get the transformation of the colorbar axes with respect to the image axes
    # # This information will be used to set the 'bbox_transform' parameter of inset_axes
    # bbox_transform = colorbar_axes.get_transform()

    # # Calculate the coordinates of the colorbar axes relative to the image axes
    # x0, y0 = bbox_transform.transform((bbox.x0, bbox.y0))
    # x1, y1 = bbox_transform.transform((bbox.x1, bbox.y1))
    # bbox_to_anchor = (x0, y0, x1 - x0, y1 - y0)

    # # Create the inset axes using the retrieved parameters
    # inset_ax = inset_axes(p.axes,
    #                       width=width,
    #                       height=height,
    #                       loc=loc,
    #                       bbox_to_anchor=bbox_to_anchor,
    #                       bbox_transform=p.axes.transAxes,
    #                       borderpad=0)

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
    cbar_kwargs={},
    xarray_colorbar=True,
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
        cbar_kwargs = {}
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
    cbar_kwargs={},
):
    """Plot pcolormesh with xarray."""
    ticklabels = cbar_kwargs.pop("ticklabels", None)
    if not add_colorbar:
        cbar_kwargs = {}
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
    fig_kwargs={},
    subplot_kwargs={},
    cbar_kwargs={},
    **plot_kwargs,
):
    from gpm_api.checks import is_grid, is_orbit
    from gpm_api.visualization.grid import plot_grid_map
    from gpm_api.visualization.orbit import plot_orbit_map

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
    fig_kwargs={},
    cbar_kwargs={},
    **plot_kwargs,
):
    # figsize, dpi, subplot_kw only used if ax is None
    from gpm_api.checks import is_grid, is_orbit
    from gpm_api.visualization.grid import plot_grid_image
    from gpm_api.visualization.orbit import plot_orbit_image

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


def plot_map_mesh(
    xr_obj,
    x="lon",
    y="lat",
    ax=None,
    edgecolors="k",
    linewidth=0.1,
    add_background=True,
    fig_kwargs={},
    subplot_kwargs={},
    **plot_kwargs,
):
    # Interpolation only for grid objects
    # figsize, dpi, subplot_kw only used if ax is None
    from gpm_api.checks import is_orbit  # is_grid

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
    fig_kwargs={},
    subplot_kwargs={},
    **plot_kwargs,
):
    """Plot GPM orbit granule mesh centroids in a cartographic map."""
    # - Check inputs
    _preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs, subplot_kwargs=subplot_kwargs)

    # - Initialize figure
    if ax is None:
        subplot_kwargs = _preprocess_subplot_kwargs(subplot_kwargs)
        fig, ax = plt.subplots(subplot_kw=subplot_kwargs, **fig_kwargs)

    # - Add cartopy background
    if add_background:
        ax = plot_cartopy_background(ax)

    # Plot centroids
    lon = xr_obj[x].data
    lat = xr_obj[y].data
    p = ax.scatter(lon, lat, transform=ccrs.PlateCarree(), c=c, s=s, **plot_kwargs)

    # - Return mappable
    return p


####--------------------------------------------------------------------------.


def _plot_labels(
    xr_obj,
    label_name=None,
    max_n_labels=50,
    add_colorbar=True,
    interpolation="nearest",
    cmap="Paired",
    fig_kwargs={},
    **plot_kwargs,
):
    """Plot labels.

    The maximum allowed number of labels to plot is 'max_n_labels'.
    """
    from ximage.labels.labels import get_label_indices, redefine_label_array
    from ximage.labels.plot_labels import get_label_colorbar_settings

    from gpm_api.visualization.plot import plot_image

    if isinstance(xr_obj, xr.Dataset):
        dataarray = xr_obj[label_name]
    else:
        if label_name is not None:
            dataarray = xr_obj[label_name]
        else:
            dataarray = xr_obj

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
    plot_kwargs, cbar_kwargs = get_label_colorbar_settings(label_indices, cmap="Paired")
    # Plot image
    p = plot_image(
        dataarray,
        interpolation=interpolation,
        add_colorbar=add_colorbar,
        cbar_kwargs=cbar_kwargs,
        fig_kwargs=fig_kwargs,
        **plot_kwargs,
    )
    return p


def plot_labels(
    obj,  # Dataset, DataArray or generator
    label_name=None,
    max_n_labels=50,
    add_colorbar=True,
    interpolation="nearest",
    cmap="Paired",
    fig_kwargs={},
    **plot_kwargs,
):
    if is_generator(obj):
        for label_id, xr_obj in obj:
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
    fig_kwargs={},
    cbar_kwargs={},
    **plot_kwargs,
):
    """Plot patches."""
    from gpm_api.visualization.plot import plot_image

    # Plot patches
    for label_id, xr_patch in patch_gen:
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
        except:
            pass
    return
