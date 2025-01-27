import xarray as xr

# import xradar as xd

# Add ds.xradar.crs accessor
#    ds_gr.xradar.get_crs() --> ds.xradar.crs
#    ds.metpy.crs
#    ds.gpm.crs


class Xradar_Dev_Base_Accessor:

    def __init__(self, xarray_obj):
        if not isinstance(xarray_obj, (xr.DataArray, xr.Dataset)):
            raise TypeError("The 'xradar_dev' accessor is available only for xarray.Dataset and xarray.DataArray.")
        self._obj = xarray_obj

    def resolution_at_range(self, azimuth_beamwidth, elevation_beamwidth):
        from gpm.gv.xradar.methods import resolution_at_range

        return resolution_at_range(
            self._obj,
            azimuth_beamwidth=azimuth_beamwidth,
            elevation_beamwidth=elevation_beamwidth,
        )

    @property
    def maximum_range_distance(self):
        from gpm.gv.xradar.methods import get_maximum_range_distance

        return get_maximum_range_distance(self._obj)

    @property
    def maximum_horizontal_distance(self):
        from gpm.gv.xradar.methods import get_maximum_horizontal_distance

        return get_maximum_horizontal_distance(self._obj)

    def extent(self, max_distance=None, crs=None):
        from gpm.gv.xradar.methods import get_extent

        return get_extent(self._obj, max_distance=max_distance, crs=crs)

    def quadmesh_corners(self):
        from gpm.gv.xradar.methods import get_quadmesh_corners

        return get_quadmesh_corners(self._obj)

    def quadmesh_vertices(self, ccw=True):
        from gpm.gv.xradar.methods import get_quadmesh_vertices

        return get_quadmesh_vertices(self._obj, ccw=ccw)

    def quadmesh_polygons(self, crs=None):
        from gpm.gv.xradar.methods import get_quadmesh_polygons

        return get_quadmesh_polygons(self._obj, crs=crs)

    @property
    def pyproj_crs(self):
        from gpm.gv.xradar.methods import _xradar_get_crs

        return _xradar_get_crs(self._obj)

    def plot_range_distance(
        self,
        distance,
        ax=None,
        add_background=True,
        fig_kwargs=None,
        subplot_kwargs=None,
        **plot_kwargs,
    ):
        from gpm.gv.xradar.methods import plot_range_distance

        return plot_range_distance(
            self._obj,
            distance=distance,
            ax=ax,
            add_background=add_background,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            **plot_kwargs,
        )


@xr.register_dataarray_accessor("xradar_dev")
class Xradar_Dev_DataArray_Accessor(Xradar_Dev_Base_Accessor):

    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    def plot_map(
        self,
        ax=None,
        x="lon",
        y="lat",
        add_colorbar=True,
        add_background=True,
        fig_kwargs=None,
        subplot_kwargs=None,
        cbar_kwargs=None,
        **plot_kwargs,
    ):
        from gpm.gv.xradar.methods import plot_map

        return plot_map(
            self._obj,
            ax=ax,
            x=x,
            y=y,
            add_colorbar=add_colorbar,
            add_background=add_background,
            fig_kwargs=fig_kwargs,
            subplot_kwargs=subplot_kwargs,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )


@xr.register_dataset_accessor("xradar_dev")
class Xradar_Dev_Dataset_Accessor(Xradar_Dev_Base_Accessor):

    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    def to_geopandas(self, dim_order=None):
        from gpm.gv.xradar.methods import to_geopandas

        return to_geopandas(self._obj, dim_order=dim_order)


@xr.register_datatree_accessor("xradar_dev")
class XradarDevDataTreeAccessor:

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @property
    def sweeps(self):
        from gpm.gv.xradar.methods import get_datatree_sweeps

        return get_datatree_sweeps(self._obj)

    @property
    def maximum_range_distance(self):
        from gpm.gv.xradar.methods import get_datatree_maximum_range_distance

        return get_datatree_maximum_range_distance(self._obj)

    @property
    def maximum_horizontal_distance(self):
        from gpm.gv.xradar.methods import get_datatree_maximum_horizontal_distance

        return get_datatree_maximum_horizontal_distance(self._obj)

    def extent(self, max_distance=None, crs=None):
        from gpm.gv.xradar.methods import get_datatree_extent

        return get_datatree_extent(self._obj, max_distance=max_distance, crs=crs)


# accessor = "xradar_dev"
# package_functions =  {"plot_map": plot_map}

# # Registered functions in this way does not accept kwargs
# xd.accessors.create_xradar_dataarray_accessor(accessor, package_functions)

# plot_map(ds_gr["DBZH"].where(mask_gr))

# ds_gr["DBZH"].xradar_dev.plot_map(
#   vmin=0,
#   vmax=40,
#   cmap="Spectral",
# )
