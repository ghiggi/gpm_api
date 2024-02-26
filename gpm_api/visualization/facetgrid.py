#!/usr/bin/env python3
"""
Created on Fri Feb 23 17:14:18 2024

@author: ghiggi
"""
import itertools
import warnings
from collections.abc import Hashable
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from xarray.plot.facetgrid import FacetGrid
from xarray.plot.utils import label_from_attrs


def define_subplots_adjust_kwargs(subplots_adjust_kwargs, add_colorbar, cbar_kwargs):
    if subplots_adjust_kwargs is None:
        colorbar_space = 0.25
        left_margin = 0.025
        right_margin = 0.025
        bottom_margin = 0.05
        top_margin = 0.12
        wspace = 0  # 0.1

        # Adjust the margins manually for colorbar
        if add_colorbar:
            orientation = cbar_kwargs.get("orientation", "vertical")
            if orientation == "vertical":
                right = 1 - colorbar_space
                left = left_margin
                bottom = bottom_margin
                top = 1 - top_margin
                hspace = 0  # 0.05 originally
            else:  # horizontal
                right = 1 - right_margin
                left = left_margin
                bottom = colorbar_space
                top = 1 - top_margin
                hspace = 0.1
                if "pad" not in cbar_kwargs:
                    cbar_kwargs["pad"] = 0.05
        else:
            top = 1 - top_margin
            bottom = bottom_margin
            right = 1 - right_margin
            left = left_margin
            hspace = 0.08

        subplots_adjust_kwargs = {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "hspace": hspace,
            "wspace": wspace,
        }
    return subplots_adjust_kwargs


def remove_gridline_bottom_labels(ax):
    gl = ax._gridliners[0]
    gl.bottom_labels = False


def remove_gridline_left_labels(ax):
    gl = ax._gridliners[0]
    gl.left_labels = False


def _remove_title_dimension_prefix(ax):
    title = ax.get_title()
    splitted_text = title.split("=")
    if len(splitted_text) >= 2:
        title = title.split("=")[-1].lstrip()
    ax.set_title(title)


class CartopyFacetGrid(FacetGrid):
    def __init__(
        self,
        data,
        col: Optional[Hashable] = None,
        row: Optional[Hashable] = None,
        col_wrap: Optional[int] = None,
        axes_pad: Optional[Tuple[float, float]] = None,
        add_colorbar: bool = True,
        cbar_kwargs: dict = {},
        fig_kwargs: dict = {},
        subplot_kws: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        data : DataArray or Dataset
            DataArray or Dataset to be plotted.
        row, col : str
            Dimension names that define subsets of the data, which will be drawn
            on separate facets in the grid.
        col_wrap : int, optional
            "Wrap" the grid the for the column variable after this number of columns,
            adding rows if ``col_wrap`` is less than the number of facets.
        figsize : Iterable of float or None, optional
            A tuple (width, height) of the figure in inches.
            If set, overrides ``size`` and ``aspect``.
        subplot_kws : dict, optional
            Dictionary of keyword arguments for Matplotlib subplots
            (:py:func:`matplotlib.pyplot.subplots`).

        """
        import matplotlib.pyplot as plt
        from cartopy.mpl.geoaxes import GeoAxes
        from mpl_toolkits.axes_grid1 import ImageGrid

        # Handle corner case of nonunique coordinates
        rep_col = col is not None and not data[col].to_index().is_unique
        rep_row = row is not None and not data[row].to_index().is_unique
        if rep_col or rep_row:
            raise ValueError(
                "Coordinates used for faceting cannot " "contain repeated (nonunique) values."
            )

        # single_group is the grouping variable, if there is exactly one
        if col and row:
            single_group = False
            nrow = len(data[row])
            ncol = len(data[col])
            nfacet = nrow * ncol
            if col_wrap is not None:
                warnings.warn("Ignoring col_wrap since both col and row were passed")
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

        # Set the subplot kwargs
        subplot_kws = {} if subplot_kws is None else subplot_kws

        # Define axes
        projection = subplot_kws["projection"]
        axes_class = (GeoAxes, dict(projection=projection))

        # Define axis spacing
        if axes_pad is None:
            axes_pad = (0.1, 0.3)

        # Define colorbar settings
        orientation = cbar_kwargs.get("orientation", "vertical")
        cbar_pad = cbar_kwargs.get("pad", 0.2)
        cbar_size = cbar_kwargs.get("size", "3%")
        if add_colorbar:
            cbar_mode = "single"
            if orientation == "vertical":
                cbar_location = "right"
            else:
                cbar_location = "bottom"
        else:
            cbar_mode = None
            cbar_location = "right"  # unused

        # Initialize figure and axes
        fig = plt.figure(**fig_kwargs)
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
            # direction="row", # plot row by row
            # label_mode="L",  # does not matter with cartopy plot
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

    def _finalize_grid(self, *axlabels) -> None:
        """Finalize the annotations and layout of FacetGrid."""
        if not self._finalized:
            self.set_axis_labels(*axlabels)
            self.set_titles()
            for ax, namedict in zip(self.axs.flat, self.name_dicts.flat):
                if namedict is None:
                    ax.set_visible(False)
            self._finalized = True

    def add_colorbar(self, **kwargs) -> None:
        """Draw a colorbar."""
        kwargs = kwargs.copy()
        # Check for extend in cmap
        if self._cmap_extend is not None:
            kwargs.setdefault("extend", self._cmap_extend)
        # Don't pass 'extend' as kwarg if it is in the mappable
        if hasattr(self._mappables[-1], "extend"):
            kwargs.pop("extend", None)
        # If label not specified, use the datarray name or attributes
        if "label" not in kwargs:
            assert isinstance(self.data, xr.DataArray)
            kwargs.setdefault("label", label_from_attrs(self.data))
        # Accept ticklabels as kwargs
        ticklabels = kwargs.pop("ticklabels", None)
        # Display the colorbar
        self.cbar = self.image_grid.cbar_axes[0].colorbar(
            self._mappables[-1], ax=list(self.axs.flat), **kwargs
        )
        # Add the ticklabels
        if ticklabels is not None:
            self.cbar.ax.set_yticklabels(ticklabels)

    def remove_duplicated_gridline_labels(self):
        """Remove gridlines labels except at the border."""
        n_rows, n_cols = self.axs.shape

        missing_bottom_plots = [not ax.has_data() for ax in self.axs[n_rows - 1]]
        idx_bottom_plots = np.where(missing_bottom_plots)[0]
        has_missing_bottom_plots = len(idx_bottom_plots) > 0

        # Remove bottom labels from all subplots except the bottom ones
        if n_rows > 1:
            for i in range(0, n_rows - 1):
                for j in range(0, n_cols):
                    if has_missing_bottom_plots and i == n_rows - 2 and j in idx_bottom_plots:
                        pass
                    else:
                        try:
                            remove_gridline_bottom_labels(ax=self.axs[i, j])
                        except Exception:
                            pass
        if n_cols > 1:
            for i in range(0, n_rows):
                for j in range(1, n_cols):
                    try:
                        remove_gridline_left_labels(ax=self.axs[i, j])
                    except Exception:
                        pass

    def set_extent(self, extent):
        """Set extent to all subplots."""
        if extent is None:
            return None

        for ax in self.axs.flat:
            try:
                ax.set_extent(extent)
            except Exception:
                pass

        # Readjust map layout
        self.optimize_layout()

    def adapt_fig_size(self):
        """
        Adjusts the figure height of the plot based on the aspect ratio of cartopy subplots.

        This function is intended to be called after all plotting has been completed.
        It operates under the assumption that all subplots within the figure share the same aspect ratio.

        The implementation is inspired by Mathias Hauser's mplotutils set_map_layout function.
        """
        # Retrieve the current size of the figure in inches.
        width, original_height = self.fig.get_size_inches()

        # Assumes that the first axis in the collection of axes is representative of all others.
        # This means that all subplots are expected to have the same aspect ratio and size.
        ax = np.asarray(self.axs).flat[0]

        # Access the figure object from the axis to manipulate its properties.
        f = ax.get_figure()

        # A call to draw the canvas is required to make sure the geometry of the figure is up-to-date.
        # This ensures that subsequent calculations for adjusting the layout are based on the latest state.
        f.canvas.draw()

        # Extract subplot parameters to understand the figure's layout.
        # These parameters include the margins of the figure and the spaces between subplots.
        bottom = f.subplotpars.bottom
        top = f.subplotpars.top
        left = f.subplotpars.left
        right = f.subplotpars.right
        hspace = f.subplotpars.hspace  # vertical space between subplots
        wspace = f.subplotpars.wspace  # horizontal space between subplots

        # Calculate the aspect ratio of the data in the subplot.
        # This ratio is used to adjust the height of the figure to match the aspect ratio of the data.
        aspect = ax.get_data_ratio()

        # Determine the number of rows and columns of subplots in the figure.
        # This information is crucial for calculating the new height of the figure.
        # nrow, ncol, __, __ = ax.get_subplotspec().get_geometry()
        nrow = self._nrow
        ncol = self._ncol

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
        f.set_figwidth(width)
        f.set_figheight(height)

    def remove_title_dimension_prefix(self):
        self.map(lambda: _remove_title_dimension_prefix(plt.gca()))

    def set_title(self, title, horizontalalignment="center", **kwargs):
        self.fig.suptitle(title, horizontalalignment=horizontalalignment, **kwargs)

    def optimize_layout(self):
        self.adapt_fig_size()
        with warnings.catch_warnings(record=False):
            warnings.simplefilter("ignore", UserWarning)
            self.fig.tight_layout()
