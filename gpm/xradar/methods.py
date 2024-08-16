import cartopy.crs as ccrs
import numpy as np
import pyproj
import xarray as xr


def resolution_at_range(xr_obj, azimuth_beamwidth, elevation_beamwidth):
    """
    Compute the horizontal and vertical resolution of each radar gate.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        xradar Dataset or DataArray.
    azimuth_beamwidth : float
        Azimuth beamwidth (Δφ) in degrees.
    elevation_beamwidth : float
        Elevation beamwidth (Δθ) in degrees.

    Returns
    -------
    horizontal_res : xr.DataArray
        Horizontal resolution of the radar gates.
    vertical_res : xr.DataArray
        Vertical resolution of the radar gates.
    """
    # Convert beamwidths from degrees to radians
    azimuth_beamwidth_rad = np.deg2rad(azimuth_beamwidth / 2)
    elevation_beamwidth_rad = np.deg2rad(elevation_beamwidth / 2)

    # Calculate the horizontal and vertical beamwidths
    da_tmp = xr_obj["DBZH"].copy() if isinstance(xr_obj, xr.Dataset) else xr_obj.copy()
    da_tmp.data[:] = 1
    horizontal_res = 2 * xr_obj["range"] * np.tan(azimuth_beamwidth_rad) * da_tmp
    vertical_res = 2 * xr_obj["range"] * np.tan(elevation_beamwidth_rad) * da_tmp
    return horizontal_res, vertical_res


def get_quadmesh(ds):
    """Return the quadrilateral mesh.

    A quadrilateral mesh is a grid of M by N adjacent quadrilaterals that are defined via a (M+1, N+1)
    grid of vertices.

    The quadrilateral mesh is accepted by `matplotlib.pyplot.pcolormesh`, `matplotlib.collections.QuadMesh`
    `matplotlib.collections.PolyQuadMesh`.

    Return
    --------
    numpy.ndarray
        Quadmesh array of shape (M+1, N+1, 2)
    """
    x = ds["x"]
    # 2D Case
    for i in reversed(range(x.ndim)):
        x = xr.plot.utils._infer_interval_breaks(x, axis=i)
    y = ds["y"]
    for i in reversed(range(y.ndim)):
        y = xr.plot.utils._infer_interval_breaks(y, axis=i)
    corners = np.stack([x, y], axis=-1)

    # 1D case
    # TODO:
    return corners


def get_vertices(ds, ccw=True):
    """Return the gates vertices in an array of shape (N, M, 4, 3).

    Once the first 2 dimension are flattened and the z dimension discarded,
    the output vertices can be passed directly to a `matplotlib.PolyCollection`.
    For plotting with cartopy, the polygon order must be "counterclockwise".
    Shapely also expects the polygon order to be "counterclockwise".
    """
    corners = get_quadmesh(ds)
    # Retrieve clockwise coordinates
    # --> lower-left, upper-left, upper-right, lower-right
    z = ds["z"].to_numpy()[..., None]  # expand dimension
    bottom_left = np.dstack([corners[0:-1, 0:-1], z])
    top_left = np.dstack([corners[0:-1, 1:], z])
    top_right = np.dstack([corners[1:, 1:], z])
    bottom_right = np.dstack([corners[1:, 0:-1], z])
    if ccw:
        list_vertices = [top_left, bottom_left, bottom_right, top_right]
    else:
        list_vertices = [top_left, top_right, bottom_right, bottom_left]

    vertices = np.stack(list_vertices, axis=-2)
    return vertices


def to_geopandas(ds, dim_order=None):
    """Convert xradar dataset to geopandas object."""
    import geopandas as gpd
    from shapely import polygons

    # Retrieve radar gates polygons on the range-azimuth plane
    poly_vertices = get_vertices(ds, ccw=True)
    poly_flat = poly_vertices[..., 0:2].reshape(-1, 4, 2)
    polygons = polygons(poly_flat)
    # Create pandas DataFrame
    df = ds.to_dataframe()
    # Copy range and azimuth also as column
    df["range"] = df.index.get_level_values("range")
    df["azimuth"] = df.index.get_level_values("azimuth")
    # Create geopandas DataFrame
    gdf = gpd.GeoDataFrame(df, crs=ds.xradar_dev.pyproj_crs, geometry=polygons)
    if dim_order is not None:
        gdf = gdf.reorder_levels(dim_order)
    return gdf


def get_radius_polygon(xr_obj, distance, crs=None):
    from shapely import Point, Polygon

    from gpm.utils.gv import reproject_coords

    if crs is None:
        crs = pyproj.CRS.from_epsg(4326)
    coords = np.array(Point(0, 0).buffer(distance).exterior.xy).T
    lon_r, lat_r, _ = reproject_coords(
        x=coords[:, 0],
        y=coords[:, 1],
        src_crs=xr_obj.xradar_dev.pyproj_crs,
        dst_crs=crs,
    )
    polygon = Polygon(np.stack((lon_r, lat_r)).T)
    return polygon


def get_extent(xr_obj, max_distance=None, crs=None):
    """Get extent , restricted to max_distance from radar location if specified.

    If the CRS is not specified, it returns extent in WGS84 CRS.
    """
    if max_distance is None:
        max_distance = xr_obj["range"].max().item()
    polygon = get_radius_polygon(xr_obj, distance=max_distance, crs=crs)
    extent = [polygon.bounds[i] for i in [0, 2, 1, 3]]
    return extent


def plot_range_distance(
    xr_obj,
    distance,
    ax=None,
    fig_kwargs=None,
    subplot_kwargs=None,
    add_background=True,
    **plot_kwargs,
):
    from gpm.visualization.plot import initialize_cartopy_plot

    # Define arguments
    plot_kwargs.setdefault("facecolor", "none")
    plot_kwargs.pop("crs", None)
    crs = ccrs.Geodetic()

    ax_provided = ax is not None

    # Initialize figure if necessary
    ax = initialize_cartopy_plot(
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
        add_background=add_background,
    )

    # Retrieve circle polygon at given radius from radar
    polygon = get_radius_polygon(xr_obj=xr_obj, distance=distance, crs=pyproj.CRS.from_wkt(crs.to_wkt()))

    # Plot circle polygon
    p = ax.add_geometries([polygon], crs=crs, **plot_kwargs)

    # Restrict extent if axis not provided
    if not ax_provided:
        extent = [polygon.bounds[i] for i in [0, 2, 1, 3]]
        ax.set_extent(extent)
    return p


def plot_map(
    da,
    x="lon",
    y="lat",
    ax=None,
    add_colorbar=True,
    add_background=True,
    fig_kwargs=None,
    subplot_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    import gpm
    from gpm.visualization.facetgrid import sanitize_facetgrid_plot_kwargs
    from gpm.visualization.plot import initialize_cartopy_plot, plot_colorbar

    # Initialize figure if necessary
    ax = initialize_cartopy_plot(
        ax=ax,
        fig_kwargs=fig_kwargs,
        subplot_kwargs=subplot_kwargs,
        add_background=add_background,
    )

    # Sanitize plot_kwargs set by by xarray FacetGrid.map_dataarray
    plot_kwargs = sanitize_facetgrid_plot_kwargs(plot_kwargs)

    # If not specified, retrieve/update plot_kwargs and cbar_kwargs as function of variable name
    variable = da.name
    plot_kwargs, cbar_kwargs = gpm.get_plot_kwargs(
        name=variable,
        user_plot_kwargs=plot_kwargs,
        user_cbar_kwargs=cbar_kwargs,
    )
    # Display variable with cartopy
    p = da.plot(
        ax=ax,
        x=x,
        y=y,
        add_colorbar=False,
        # cbar_kwargs=cbar_kwargs,
        **plot_kwargs,
    )
    # Remove title
    ax.set_title("")

    # Add colorbar
    if add_colorbar:
        _ = plot_colorbar(p=p, ax=ax, cbar_kwargs=cbar_kwargs)

    return p
