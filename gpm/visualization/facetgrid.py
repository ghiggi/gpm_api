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
"""This module contains the FacetGrid classes."""
import itertools
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.gridliner import Gridliner
from mpl_toolkits.axes_grid1 import ImageGrid
from xarray.plot.facetgrid import FacetGrid
from xarray.plot.utils import _infer_xy_labels, _process_cmap_cbar_kwargs, label_from_attrs

from gpm.visualization.plot import adapt_fig_size


def _remove_dim_prefix(title):
    splitted_text = title.split("=")
    if len(splitted_text) >= 2:
        title = title.split("=")[-1].lstrip()
    return title


def _remove_title_dimension_prefix(ax):
    title = ax.get_title()
    title = _remove_dim_prefix(title)
    ax.set_title(title)


def _remove_title(ax):
    ax.set_title("")


def get_cartopy_gridlines_artists(ax):
    """Retrieve the cartopy gridline artist."""
    # OLD approach: gl = [ax._gridliners[0]]
    list_gridliners = [artist for artist in ax.artists if isinstance(artist, Gridliner)]
    return list_gridliners


def sanitize_facetgrid_plot_kwargs(plot_kwargs):
    """Remove defaults values set by FacetGrid.map_dataarray."""
    plot_kwargs = plot_kwargs.copy()
    is_facetgrid = plot_kwargs.get("_is_facetgrid", False)
    if is_facetgrid:
        facet_grid_args = ["vmin", "vmax", "extend", "levels", "add_labels", "_is_facetgrid"]
        _ = [plot_kwargs.pop(arg, None) for arg in facet_grid_args]
    return plot_kwargs


class CustomFacetGrid(FacetGrid, ABC):
    def __init__(
        self,
        data,
        col: Hashable | None = None,
        row: Hashable | None = None,
        col_wrap: int | None = None,
        axes_pad: tuple[float, float] | None = None,
        aspect: bool = True,
        add_colorbar: bool = True,
        facet_height: float = 3.0,
        facet_aspect: float = 1.0,
        cbar_kwargs: dict | None = None,
        fig_kwargs: dict | None = None,
        axes_class=None,
    ) -> None:
        """Class for xarray-based FacetGrid plots.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            xarray object to be plotted.
        row, col : str
            Dimension names that define subsets of the data, which will be drawn
            on separate facets in the grid.
        col_wrap : int, optional
            "Wrap" the grid the for the column variable after this number of columns,
            adding rows if ``col_wrap`` is less than the number of facets.
        axes_pad : tuple or float, optional
            Padding or (horizontal padding, vertical padding) between axes, in
            inches. The default is ``(0.1, 0.3)`` inches.
        aspect : bool, optional
            Whether the axes aspect ratio follows the aspect ratio of the data
            limits. The default is ``True``.
        axes_class : subclass of :py:class:`matplotlib.axes.Axes`, optional
            The default is ``None``.
        add_colorbar: bool, optional
            Whether to add a colorbar to the figure.
            The default is ``True``.
        cbar_kwargs : dict, optional
            Dictionary of keyword arguments to pass to the colorbar.
            The ``pad`` argument controls the space between the image axes and the colorbar axes.
            The ``pad`` default is 0.2.
            The ``size`` argument control the colorbar size. The default value is '3%'.
            For other arguments, see :py:class:`matplotlib.figure.Figure.colorbar`.
        facet_height: float, optional
            Height (in inches) of each facet. The default is 3.
            This parameter is used only if the ``figsize`` argument is not specified in ``fig_kwargs``.
        facet_aspect:  float, optional
            Aspect ratio of each facet. The default is 1.
            The facet width is determined by ``facet_height`` * ``facet_aspect``.
            This parameter is used only if the ``figsize`` argument is not specified in ``fig_kwargs``.
        fig_kwargs : dict, optional
                Dictionary of keyword arguments to pass to the Figure.
                Typical arguments include ``figsize`` and ``dpi``.
                ``figsize`` is a tuple (width, height) of the figure in inches.
                If ``figsize`` is specified, it overrides ``facet_size`` and ``facet_aspect`` arguments.
                (see :py:class:`matplotlib.figure.Figure`).

        """
        # Handle corner case of nonunique coordinates
        rep_col = col is not None and not data[col].to_index().is_unique
        rep_row = row is not None and not data[row].to_index().is_unique
        if rep_col or rep_row:
            raise ValueError(
                "Coordinates used for faceting cannot " "contain repeated (nonunique) values.",
            )

        # single_group is the grouping variable, if there is exactly one
        if col and row:
            single_group = False
            nrow = len(data[row])
            ncol = len(data[col])
            nfacet = nrow * ncol
            if col_wrap is not None:
                warnings.warn("Ignoring col_wrap since both col and row were passed", stacklevel=1)
        elif row and not col:
            single_group = row
        elif not row and col:
            single_group = col
        else:
            raise ValueError("Pass a coordinate name as an argument for row or col")

        # Compute grid shape
        if single_group:
            nfacet = len(data[single_group])
            if col:
                # idea - could add heuristic for nice shapes like 3x4
                ncol = nfacet
            if row:
                ncol = 1
            if col_wrap is not None:
                # Overrides previous settings
                ncol = col_wrap
            nrow = int(np.ceil(nfacet / ncol))

        # Define axis spacing
        if axes_pad is None:
            axes_pad = (0.1, 0.3)

        # Define colorbar settings
        default_pad = 0.3 if (row is not None and col is not None) else 0.2
        cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
        orientation = cbar_kwargs.get("orientation", "vertical")
        cbar_pad = cbar_kwargs.get("pad", default_pad)
        cbar_size = cbar_kwargs.get("size", "3%")
        if add_colorbar:
            cbar_mode = "single"
            cbar_location = "right" if orientation == "vertical" else "bottom"
        else:
            cbar_mode = None
            cbar_location = "right"  # unused

        # Initialize figure size
        # --> facet_height=size and facet_aspect=aspect in xarray FacetGrid
        # --> We could provide this also as argument (fig_kwargs or **plot_kwargs?)
        # --> Only used in figsize not specified !
        fig_kwargs = {} if fig_kwargs is None else fig_kwargs
        figsize = fig_kwargs.pop("figsize", None)
        if figsize is None:  # xarray FacetGrid defaults
            facet_width = facet_height * facet_aspect  # Width (in inches) of each facet
            figsize = [ncol * facet_width, nrow * facet_height]  # (width, height)
            if add_colorbar:
                cbar_space = 1
                if orientation == "vertical":
                    figsize[0] = figsize[0] + cbar_space  # extra width space
                else:
                    figsize[1] = figsize[1] + cbar_space  # extra height space
            figsize = tuple(figsize)

        # Initialize figure and axes
        fig = plt.figure(figsize=figsize, **fig_kwargs)
        image_grid = ImageGrid(
            fig,
            111,
            axes_class=axes_class,
            nrows_ncols=(nrow, ncol),
            axes_pad=axes_pad,  # Padding or (horizontal padding, vertical padding) between axes, in inches
            cbar_location=cbar_location,
            cbar_mode=cbar_mode,
            cbar_pad=cbar_pad,
            cbar_size=cbar_size,
            aspect=aspect,
            # direction="row", # plot row by row
            label_mode="all",  # does not matter with cartopy plot
        )

        # Extract axes like subplots
        axs = np.array(image_grid.axes_all).reshape(nrow, ncol)

        # Delete empty axis (to avoid bad layout)
        n_subplots = nrow * ncol
        if nfacet != n_subplots:
            for i in range(nfacet, n_subplots):
                fig.delaxes(axs.flatten()[i])

        # Set up the lists of names for the row and column facet variables
        col_names = list(data[col].to_numpy()) if col else []
        row_names = list(data[row].to_numpy()) if row else []

        if single_group:
            full = [{single_group: x} for x in data[single_group].to_numpy()]
            empty = [None for x in range(nrow * ncol - len(full))]
            name_dict_list = full + empty
        else:
            rowcols = itertools.product(row_names, col_names)
            name_dict_list = [{row: r, col: c} for r, c in rowcols]

        name_dicts = np.array(name_dict_list).reshape(nrow, ncol)

        # Set up the class attributes
        # ---------------------------

        # First the public API
        self.data = data
        self.name_dicts = name_dicts
        self.fig = fig
        self.image_grid = image_grid
        self.axs = axs
        self.row_names = row_names
        self.col_names = col_names

        # guides
        self.figlegend = None
        self.quiverkey = None
        self.cbar = None

        # Next the private variables
        self._single_group = single_group
        self._nrow = nrow
        self._row_var = row
        self._ncol = ncol
        self._col_var = col
        self._col_wrap = col_wrap
        self.row_labels = [None] * nrow
        self.col_labels = [None] * ncol
        self._x_var = None
        self._y_var = None
        self._cmap_extend = None
        self._mappables = []
        self._finalized = False

    def map_dataarray(
        self,
        func,
        x=None,
        y=None,
        **kwargs,
    ):
        """
        Apply a plotting function to a 2d facet's subset of the data.

        This is more convenient and less general than ``FacetGrid.map``

        Parameters
        ----------
        func : callable
            A plotting function with the same signature as a 2d xarray
            plotting method such as xarray.plot.imshow
        x, y : str
            Names of the coordinates to plot on x, y axes
        **kwargs
            additional keyword arguments to func

        Returns
        -------
        xarray.plot.facetgrid.FacetGrid
            FacetGrid object

        """
        if kwargs.get("cbar_ax") is not None:
            raise ValueError("cbar_ax not supported by FacetGrid.")

        cmap_params, cbar_kwargs = _process_cmap_cbar_kwargs(
            func,
            self.data.to_numpy(),
            **kwargs,
        )

        self._cmap_extend = cmap_params.get("extend")

        # Order is important
        func_kwargs = {k: v for k, v in kwargs.items() if k not in {"cmap", "colors", "cbar_kwargs", "levels"}}
        func_kwargs.update(cmap_params)
        func_kwargs["add_colorbar"] = False

        # if func.__name__ != "surface":
        #     func_kwargs["add_labels"] = False

        # Get x, y labels for the first subplot
        # - Get DataArray prototype without row, col and rgb !
        da_proto = self.data.loc[self.name_dicts.flat[0]]
        if self._row_var in list(da_proto.dims):
            da_proto = da_proto.isel({self._row_var: 0})
        if self._col_var in list(da_proto.dims):
            da_proto = da_proto.isel({self._col_var: 0})
        if kwargs.get("rgb"):
            da_proto = da_proto.isel({kwargs.get("rgb"): 0})

        # Infer x - y labels
        x, y = _infer_xy_labels(
            darray=da_proto,
            x=x,
            y=y,
            imshow=True,
            # rgb=kwargs.get("rgb", None),
        )
        for d, ax in zip(self.name_dicts.flat, self.axs.flat, strict=False):
            # None is the sentinel value
            if d is not None:
                subset = self.data.loc[d]
                mappable = func(
                    subset,
                    x=x,
                    y=y,
                    ax=ax,
                    **func_kwargs,
                    _is_facetgrid=True,
                )
                self._mappables.append(mappable)

        self._finalize_grid(x, y)

        if kwargs.get("add_colorbar", True):
            self.add_colorbar(**cbar_kwargs)

        return self

    @abstractmethod
    def _remove_bottom_ticks_and_labels(self, ax):
        """Method removing axis ticks and labels on the bottom of the subplots."""
        raise NotImplementedError

    @abstractmethod
    def _remove_left_ticks_and_labels(self, ax):
        """Method removing axis ticks and labels on the left of the subplots."""
        raise NotImplementedError

    def map_to_axes(self, func, **kwargs):
        """Map a function to each axes."""
        n_rows, n_cols = self.axs.shape
        missing_bottom_plots = [not ax.has_data() for ax in self.axs[n_rows - 1]]
        idx_bottom_plots = np.where(missing_bottom_plots)[0]
        has_missing_bottom_plots = len(idx_bottom_plots) > 0
        for i in range(0, n_rows):
            for j in range(0, n_cols):
                if has_missing_bottom_plots and i == n_rows and j in idx_bottom_plots:
                    continue
                # Otherwise apply function
                func(ax=self.axs[i, j], **kwargs)

    def remove_bottom_ticks_and_labels(self):
        """Remove the bottom ticks and labels from each subplot."""
        self.map_to_axes(func=self._remove_bottom_ticks_and_labels)

    def remove_left_ticks_and_labels(self):
        """Remove the left ticks and labels from each subplot."""
        self.map_to_axes(func=self._remove_left_ticks_and_labels)

    def remove_duplicated_axis_labels(self):
        """Remove axis labels which are not located on the left or bottom of the figure."""
        n_rows, n_cols = self.axs.shape
        missing_bottom_plots = [not ax.has_data() for ax in self.axs[n_rows - 1]]
        idx_bottom_plots = np.where(missing_bottom_plots)[0]
        has_missing_bottom_plots = len(idx_bottom_plots) > 0

        # Remove bottom axis labels from all subplots except the bottom ones
        if n_rows > 1:
            for i in range(0, n_rows - 1):
                for j in range(0, n_cols):
                    if has_missing_bottom_plots and i == n_rows - 2 and j in idx_bottom_plots:
                        continue
                    self._remove_bottom_ticks_and_labels(ax=self.axs[i, j])

        # Remove left axis labels from all subplots except the left ones
        if n_cols > 1:
            for i in range(0, n_rows):
                for j in range(1, n_cols):
                    self._remove_left_ticks_and_labels(ax=self.axs[i, j])

    def add_colorbar(self, **cbar_kwargs) -> None:
        """Draw a colorbar."""
        cbar_kwargs = cbar_kwargs.copy()
        # Check for extend in cmap
        if self._cmap_extend is not None:
            cbar_kwargs.setdefault("extend", self._cmap_extend)
        # Don't pass 'extend' as kwarg if it is in the mappable
        if hasattr(self._mappables[-1], "extend"):
            cbar_kwargs.pop("extend", None)
        # If label not specified, use the dataarray name or attributes
        if "label" not in cbar_kwargs:
            assert isinstance(self.data, xr.DataArray)
            cbar_kwargs.setdefault("label", label_from_attrs(self.data))
        # Accept ticklabels as kwargs
        ticklabels = cbar_kwargs.pop("ticklabels", None)
        # Draw the colorbar
        self.cbar = self.image_grid.cbar_axes[0].colorbar(
            self._mappables[-1],
            ax=list(self.axs.flat),
            **cbar_kwargs,
        )
        # Add ticklabel
        if ticklabels is not None:
            # Retrieve ticks
            ticks = cbar_kwargs.get("ticks", None)
            if ticks is None:
                ticks = self.cbar.get_ticks()
            # Remove existing ticklabels
            self.cbar.set_ticklabels([])
            self.cbar.set_ticklabels([], minor=True)
            # Add custom ticklabels
            self.cbar.set_ticks(ticks, labels=ticklabels)
            # self.cbar.ax.set_yticklabels(ticklabels)

    def remove_title_dimension_prefix(self, row=True, col=True):
        """Remove the dimension prefix from the subplot labels."""
        if len(self.row_names) == 0 or len(self.col_names) == 0:
            self.map(lambda: _remove_title_dimension_prefix(plt.gca()))
        else:
            if col:
                _ = [ann.set_text(_remove_dim_prefix(ann.get_text())) for ann in self.col_labels]
            if row:
                _ = [ann.set_text(_remove_dim_prefix(ann.get_text())) for ann in self.row_labels]

    def remove_titles(self, row=True, col=True):
        """Remove the plot titles."""
        if len(self.row_names) == 0 or len(self.col_names) == 0:
            self.map(lambda: _remove_title(plt.gca()))
        else:
            if col:
                _ = [ann.set_text("") for ann in self.col_labels]
            if row:
                _ = [ann.set_text("") for ann in self.row_labels]

    def set_title(self, title, horizontalalignment="center", **kwargs):
        """Add a title above all sublots.

        The y argument controls the spacing to the subplots.
        Decreasing or increasing the y argument (from a default value of 1)
        reduce/increase the spacing.
        """
        self.fig.suptitle(title, horizontalalignment=horizontalalignment, **kwargs)

    def adapt_fig_size(self):
        """Adjusts the figure height of the plot based on the aspect ratio of cartopy subplots.

        This function is intended to be called after all plotting has been completed.
        It operates under the assumption that all subplots within the figure share the same aspect ratio.

        The implementation is inspired by Mathias Hauser's mplotutils set_map_layout function.
        """
        # Assumes that the first axis in the collection of axes is representative of all others.
        # This means that all subplots are expected to have the same aspect ratio and size.
        ax = np.asarray(self.axs).flat[0]
        adapt_fig_size(ax, nrow=self._nrow, ncol=self._ncol)


class CartopyFacetGrid(CustomFacetGrid):
    def __init__(
        self,
        data,
        projection,
        col: Hashable | None = None,
        row: Hashable | None = None,
        col_wrap: int | None = None,
        axes_pad: tuple[float, float] | None = None,
        add_colorbar: bool = True,
        cbar_kwargs: dict | None = None,
        fig_kwargs: dict | None = None,
        facet_height: float = 3.0,
        facet_aspect: float = 1.0,
    ) -> None:
        """Cartopy FacetGrid class.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            xarray object to be plotted.
        projection: cartopy.crs.CRS
            Cartopy projection.
        row, col : str
            Dimension names that define subsets of the data, which will be drawn
            on separate facets in the grid.
        col_wrap : int, optional
            "Wrap" the grid the for the column variable after this number of columns,
            adding rows if ``col_wrap`` is less than the number of facets.
        axes_pad : tuple or float, optional
            Padding or (horizontal padding, vertical padding) between axes, in
            inches. The default is ``(0.1, 0.3)`` inches.
        add_colorbar: bool, optional
            Whether to add a colorbar to the figure.
            The default is ``True``.
        cbar_kwargs : dict, optional
            Dictionary of keyword arguments to pass to the colorbar.
            The ``pad`` argument controls the space between the image axes and the colorbar axes.
            The ``pad`` default is 0.2.
            The ``size`` argument control the colorbar size. The default value is ``'3%'``.
            For other arguments, see :py:class:`matplotlib.figure.Figure.colorbar`.
        facet_height: float, optional
            Height (in inches) of each facet. The default is 3.
            This parameter is used only if the ``figsize`` argument is not specified in ``fig_kwargs``.
        facet_aspect:  float, optional
           Aspect ratio of each facet. The default is 1.
           The facet width is determined by ``facet_height`` * ``facet_aspect``.
           This parameter is used only if the ``figsize`` argument is not specified in ``fig_kwargs``.
        fig_kwargs : dict, optional
             Dictionary of keyword arguments to pass to the Figure.
             Typical arguments include ``figsize`` and ``dpi``.
             ``figsize`` is a tuple (width, height) of the figure in inches.
             If ``figsize`` is specified, it overrides ``facet_size`` and ``facet_aspect`` arguments.
             (see `matplotlib.figure.Figure`).

        """
        # Define Cartopy axes
        if projection is None:
            raise ValueError("Please specify a Cartopy projection.")
        axes_class = (GeoAxes, {"projection": projection})

        super().__init__(
            data=data,
            col=col,
            row=row,
            col_wrap=col_wrap,
            axes_pad=axes_pad,
            aspect=True,
            add_colorbar=add_colorbar,
            cbar_kwargs=cbar_kwargs,
            fig_kwargs=fig_kwargs,
            facet_height=facet_height,
            facet_aspect=facet_aspect,
            axes_class=axes_class,
        )

    def _finalize_grid(self, *axlabels) -> None:
        """Finalize the annotations and layout of FacetGrid."""
        if not self._finalized:
            self.set_axis_labels(*axlabels)
            self.set_titles()
            for ax, namedict in zip(self.axs.flat, self.name_dicts.flat, strict=False):
                if namedict is None:
                    ax.set_visible(False)
            self._finalized = True

    def _remove_bottom_ticks_and_labels(self, ax):
        """Remove Cartopy bottom gridlines labels."""
        if isinstance(ax, GeoAxes):
            try:
                for gl in get_cartopy_gridlines_artists(ax):
                    gl.bottom_labels = False
            except Exception:
                pass

    def _remove_left_ticks_and_labels(self, ax):
        """Remove Cartopy left gridlines labels."""
        if isinstance(ax, GeoAxes):
            try:
                for gl in get_cartopy_gridlines_artists(ax):
                    gl.left_labels = False
            except Exception:
                pass

    def optimize_layout(self):
        """Optimize the figure size and layout of the Figure.

        This function must be called only once !
        """
        self.adapt_fig_size()
        with warnings.catch_warnings(record=False):
            warnings.simplefilter("ignore", UserWarning)
            self.fig.tight_layout()

    def set_extent(self, extent):
        """Modify extent of all Cartopy subplots."""
        if extent is None:
            return
        # Modify extent
        for ax in self.axs.flat:
            if isinstance(ax, GeoAxes):
                ax.set_extent(extent)
        # Readjust map layout
        self.optimize_layout()


class ImageFacetGrid(CustomFacetGrid):
    def __init__(
        self,
        data,
        col: Hashable | None = None,
        row: Hashable | None = None,
        col_wrap: int | None = None,
        axes_pad: tuple[float, float] | None = None,
        aspect: bool = False,
        add_colorbar: bool = True,
        cbar_kwargs: dict | None = None,
        fig_kwargs: dict | None = None,
        facet_height: float = 3.0,
        facet_aspect: float = 1.0,
    ) -> None:
        """Image FacetGrid class.

        Parameters
        ----------
        data : xarray.DataArray or xarray.Dataset
            xarray object to be plotted.
        row, col : str
            Dimension names that define subsets of the data, which will be drawn
            on separate facets in the grid.
        col_wrap : int, optional
            "Wrap" the grid the for the column variable after this number of columns,
            adding rows if ``col_wrap`` is less than the number of facets.
        axes_pad : tuple or float, optional
            Padding or (horizontal padding, vertical padding) between axes, in
            inches. The default is ``(0.1, 0.3)`` inches.
        aspect : bool
            Whether the axes aspect ratio follows the aspect ratio of the data limits.
            The default is ``False``.
        add_colorbar: bool, optional
            Whether to add a colorbar to the figure.
            The default is ``True``.
        cbar_kwargs : dict, optional
            Dictionary of keyword arguments to pass to the colorbar.
            The ``pad`` argument controls the space between the image axes and the colorbar axes.
            The ``pad`` default is 0.2.
            The ``size`` argument control the colorbar size. The default value is ``'3%'``.
            For other arguments, see :py:class:`matplotlib.figure.Figure.colorbar`.
        facet_height: float, optional
            Height (in inches) of each facet. The default is 3.
            This parameter is used only if the ``figsize`` argument is not specified in ``fig_kwargs``.
        facet_aspect:  float, optional
           Aspect ratio of each facet. The default is 1.
           The facet width is determined by ``facet_height`` * ``facet_aspect``.
           This parameter is used only if the ``figsize`` argument is not specified in ``fig_kwargs``.
        fig_kwargs : dict, optional
             Dictionary of keyword arguments to pass to the Figure.
             Typical arguments include ``figsize`` and ``dpi``.
             ``figsize`` is a tuple (width, height) of the figure in inches.
             If ``figsize`` is specified, it overrides ``facet_size`` and ``facet_aspect`` arguments.
             (see :py:class:`matplotlib.figure.Figure`).

        """
        super().__init__(
            data=data,
            col=col,
            row=row,
            col_wrap=col_wrap,
            axes_pad=axes_pad,
            aspect=aspect,
            add_colorbar=add_colorbar,
            cbar_kwargs=cbar_kwargs,
            fig_kwargs=fig_kwargs,
            facet_height=facet_height,
            facet_aspect=facet_aspect,
        )

    def _finalize_grid(self, *axlabels) -> None:  # noqa
        """Finalize the annotations and layout of FacetGrid."""
        if not self._finalized:
            # Add subplots titles
            self.set_titles()
            # Make empty subplots unvisible
            for ax, namedict in zip(self.axs.flat, self.name_dicts.flat, strict=False):
                if namedict is None:
                    ax.set_visible(False)
            self._finalized = True

    def _remove_bottom_ticks_and_labels(self, ax):
        """Remove bottom ticks and labels."""
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_xlabel("")

    def _remove_left_ticks_and_labels(self, ax):
        """Remove left ticks and labels."""
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_ylabel("")
        ax.tick_params(axis="y", length=0)
