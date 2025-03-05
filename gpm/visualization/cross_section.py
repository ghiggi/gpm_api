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
"""This module contains functions to visualize cross-sections."""
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr
from pyproj import Geod

from gpm import get_plot_kwargs
from gpm.checks import check_has_cross_track_dim, check_is_cross_section, check_is_transect
from gpm.utils.checks import check_contiguous_scans
from gpm.utils.xarray import get_dimensions_without
from gpm.visualization.plot import (
    check_object_format,
    get_valid_pcolormesh_inputs,
    initialize_cartopy_plot,
    plot_xr_imshow,
    plot_xr_pcolormesh,
    preprocess_figure_args,
)


def get_cross_track_horizontal_distance(xr_obj):
    """Retrieve the horizontal_distance from the nadir.

    Requires a cross-section with cross_track as horizontal spatial dimension !
    """
    check_is_transect(xr_obj, strict=False)
    check_has_cross_track_dim(xr_obj)

    # Retrieve required DataArrays
    lons = xr_obj["lon"].data
    lats = xr_obj["lat"].data
    idx = np.where(xr_obj["gpm_cross_track_id"] == 24)[0].item()
    start_lon = xr_obj["lon"].isel(cross_track=idx).data
    start_lat = xr_obj["lat"].isel(cross_track=idx).data

    geod = Geod(ellps="WGS84")
    distances = np.array([geod.inv(start_lon, start_lat, lon, lat)[2] for lon, lat in zip(lons, lats, strict=False)])
    distances[:idx] = -distances[:idx]
    da_dist = xr.DataArray(distances, dims="cross_track")
    return da_dist


def _ensure_valid_pcolormesh_coords(da, x, y, rgb):
    # Ensure coords dimensions are aligned with data array
    da = da.transpose(*da.dims)
    # Get 2D x and y coordinates
    da = da.copy()
    da_template = da.isel({da.dims[-1]: 0}) if rgb else da
    da_x = da[x].broadcast_like(da_template)
    da_y = da[y].broadcast_like(da_template)

    # Get valid coordinates
    x_coord, y_coord, data = get_valid_pcolormesh_inputs(
        x=da_x.data,
        y=da_y.data,
        data=da.data,
        rgb=rgb,
        mask_data=True,
    )
    # Mask data
    da.data = data

    # Set back validated coordinates
    # - If x or y are dimension names without coordinates, nothing to be done
    if x in da.coords:
        if da[x].ndim == 1:
            dim_name = list(da[x].dims)[0]
            da_x.data = x_coord
            da_x_values = da_x.isel({dim: 0 for dim in get_dimensions_without(da_x, da[x].dims)}).data
            da = da.assign_coords({x: (dim_name, da_x_values)})
        else:
            # da[x].data = x_coord
            da = da.assign_coords({x: (da_x.dims, x_coord)})
    if y in da.coords:
        if da[y].ndim == 1:
            dim_name = list(da[y].dims)[0]
            da_y.data = y_coord
            da_y_values = da_y.isel({dim: 0 for dim in get_dimensions_without(da_y, da[y].dims)}).data
            da = da.assign_coords({y: (dim_name, da_y_values)})
        else:
            da = da.assign_coords({y: (da_y.dims, y_coord)})
            # da[y].data = y_coord
    return da


def _get_x_axis_options(da, x):
    # Define xlabels
    xlabel_dicts = {
        "cross_track": "Cross-Track",
        "along_track": "Along-Track",
        "horizontal_distance": "Distance from nadir [m]",
        "horizontal_distance_km": "Distance from nadir [km]",
        "lon": "Longitude [°]",
        "lat": "Latitude [°]",
    }

    # Define additional coordinates on the fly if asked
    if x in ["horizontal_distance", "horizontal_distance_km"]:
        scale_factor = 1000 if x == "horizontal_distance_km" else 1
        da_distance = get_cross_track_horizontal_distance(da) / scale_factor
        da = da.assign_coords({x: da_distance})
    # If x specified, check valid coordinate
    if x is not None:
        if x not in list(set(da.dims) | set(da.coords)):
            raise ValueError(f"'{x}' is not a DataArray coordinate. Specify a valid 'x' or compute '{x}'.")
    else:  # set default (cross_track or along_track)
        # TODO in future use gpm.x_dims
        candidate_dims = get_dimensions_without(da, da.gpm.vertical_dimension)
        if "cross_track" in candidate_dims:
            x = "cross_track"
        elif "along_track" in candidate_dims:
            x = "along_track"
        else:
            x = get_dimensions_without(da, da.gpm.vertical_dimension)[0]  # the dimension which is not vertical
    # Define xlabel
    xlabel = xlabel_dicts.get(x, x.title())
    # Return x, label and DataArray
    return x, xlabel, da


def _get_y_axis_options(da, y, origin):
    # Define ylabels
    # - Order of keys is the preferred y
    ylabel_dicts = {
        "height": "Height [m]",
        "height_km": "Height [km]",
        "range": "Range Index",  # Start at 1
        "gpm_range_id": "Range Index",  # Start at 0
        "range_distance_from_satellite": "Range Distance From Satellite [m]",
        "range_distance_from_ellipsoid": "Range Distance From Ellipsoid [m]",
        "range_distance_from_satellite_km": "Range Distance From Satellite [km]",
        "range_distance_from_ellipsoid_km": "Range Distance From Ellipsoid [km]",
    }

    # Check y and define default if None
    y = _get_default_y(y=y, da=da, possible_defaults=list(ylabel_dicts))

    # Define additional coordinates on the fly
    if y in ["range_distance_from_satellite_km", "range_distance_from_ellipsoid_km", "height_km"]:
        da = da.assign_coords({y: da[y[:-3]] / 1000})

    # Define origin for 1D y coordinate
    if origin is None:
        origin = "lower" if y in ["height", "height_km"] else "upper"  # range, gpm_range_id

    # Define ylabel
    ylabel = ylabel_dicts.get(y, y.title())

    # Return x, label and DataArray
    return y, ylabel, da, origin


def _get_default_y(y, da, possible_defaults):
    """Define default y."""
    # Define default "y" (at least "range" is available since check_is_cross_section() called before
    if y is None:
        candidate_y = list(set(da.dims) | set(da.coords))
        expected_y = np.array(possible_defaults)
        available_y = expected_y[np.isin(expected_y, candidate_y)]
        return available_y[0]
    if y in ["range_distance_from_satellite_km", "range_distance_from_ellipsoid_km", "height_km"]:
        if y[:-3] not in (da.coords):
            raise ValueError(f"'{y[:-3]}' is not a DataArray coordinate. Specify a valid 'y' or compute {y[:-3]}.")
        return y
    if y not in list(set(da.dims) | set(da.coords)):
        raise ValueError(f"'{y}' is not a DataArray coordinate. Specify a valid 'y' or compute '{y}'.")
    return y


def plot_cross_section(
    da,
    x=None,
    y=None,
    ax=None,
    add_colorbar=True,
    zoom=True,
    check_contiguity=False,
    interpolation="nearest",
    fig_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """Plot GPM cross-section.

    If RGB DataArray, all other plot_kwargs are ignored !
    """
    # TODO: With cbar_kwargs, we currently cannot use 'size' argument colorbar directly
    # because we use xr.imshow() and xr.pcolormesh()

    da = check_object_format(da, plot_kwargs=plot_kwargs, check_function=check_is_cross_section, strict=True)
    is_facetgrid = "col" in plot_kwargs or "row" in plot_kwargs

    if is_facetgrid and ax is not None:
        raise ValueError("When creating a FacetGrid plot, do not specify the 'ax'.")

    # - Check for contiguous along-track scans
    if "along_track" in da.dims and check_contiguity:
        check_contiguous_scans(da)

    # - Initialize figure
    fig_kwargs = preprocess_figure_args(ax=ax, fig_kwargs=fig_kwargs)

    if ax is None and not is_facetgrid:
        _, ax = plt.subplots(**fig_kwargs)

    # - If not specified, retrieve/update plot_kwargs and cbar_kwargs as function of product name
    plot_kwargs, cbar_kwargs = get_plot_kwargs(
        name=da.name,
        user_plot_kwargs=plot_kwargs,
        user_cbar_kwargs=cbar_kwargs,
    )

    # - Select only vertical regions with data
    if zoom:
        da = da.gpm.subset_range_with_valid_data()

    # - Check x and define x label
    x, xlabel, da = _get_x_axis_options(da, x=x)

    # - Check y and define ylabel
    y, ylabel, da, origin = _get_y_axis_options(da, y=y, origin=plot_kwargs.get("origin", None))

    # - Plot with xarray
    if da[y].ndim == 1 and da[x].ndim == 1:
        plot_kwargs["origin"] = origin
        p = plot_xr_imshow(
            ax=ax,
            da=da,
            x=x,
            y=y,
            interpolation=interpolation,
            add_colorbar=add_colorbar,
            cbar_kwargs=cbar_kwargs,
            visible_colorbar=True,
            **plot_kwargs,
        )
    else:
        # Infill invalid coordinates and add mask to data if necessary
        # - This occur when extracting L2 dataset from L1B and use y = "height/rangeDist"
        da = _ensure_valid_pcolormesh_coords(da, x=x, y=y, rgb=plot_kwargs.get("rgb", False))

        # Plot cross-section
        p = plot_xr_pcolormesh(
            ax=ax,
            da=da,
            x=x,
            y=y,
            add_colorbar=add_colorbar,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
    if not is_facetgrid:
        p.axes.set_xlabel(xlabel)
        p.axes.set_ylabel(ylabel)

    # - Return mappable
    return p


def plot_transect_line(
    xr_obj,
    ax=None,
    add_direction=True,
    add_background=True,
    add_gridlines=True,
    add_labels=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    text_kwargs=None,
    line_kwargs=None,
    **common_kwargs,
):
    # - Check is transect
    check_is_transect(xr_obj, strict=False)  # allow i.e. radar_frequency, vertical dimension etc.

    # - Set defaults
    text_kwargs = {} if text_kwargs is None else text_kwargs
    line_kwargs = {} if line_kwargs is None else line_kwargs

    # - Initialize figure if necessary
    ax = initialize_cartopy_plot(
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
        add_background=add_background,
        add_gridlines=add_gridlines,
        add_labels=add_labels,
    )

    # Retrieve start and end coordinates
    start_lonlat = (xr_obj["lon"].data[0], xr_obj["lat"].data[0])
    end_lonlat = (xr_obj["lon"].data[-1], xr_obj["lat"].data[-1])
    # lon_startend = (start_lonlat[0], end_lonlat[0])
    # lat_startend = (start_lonlat[1], end_lonlat[1])
    # # Draw line
    # p = ax.plot(lon_startend, lat_startend, transform=ccrs.Geodetic(), **line_kwargs, **common_kwargs)

    # Draw line
    p = ax.plot(xr_obj["lon"].data, xr_obj["lat"].data, transform=ccrs.Geodetic(), **line_kwargs, **common_kwargs)

    # Add transect start (Left) and End (Right) (i.e. when plotting cross-section)
    if add_direction:
        g = pyproj.Geod(ellps="WGS84")
        fwd_az, back_az, dist = g.inv(*start_lonlat, *end_lonlat, radians=False)
        lon_r, lat_r, _ = g.fwd(*start_lonlat, az=fwd_az, dist=dist + 50000)  # dist in m
        fwd_az, back_az, dist = g.inv(*end_lonlat, *start_lonlat, radians=False)
        lon_l, lat_l, _ = g.fwd(*end_lonlat, az=fwd_az, dist=dist + 50000)  # dist in m
        ax.text(lon_r, lat_r, "S", **text_kwargs, **common_kwargs)
        ax.text(lon_l, lat_l, "E", **text_kwargs, **common_kwargs)

    # - Return mappable
    return p[0]
