#!/usr/bin/env python3
"""
Created on Fri Feb 23 17:14:18 2024

@author: ghiggi
"""
from collections.abc import Hashable, Iterable
from typing import Any

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
        col: Hashable | None = None,
        row: Hashable | None = None,
        col_wrap: int | None = None,
        sharex: bool = False,
        sharey: bool = False,
        figsize: Iterable[float] | None = None,
        aspect: float = 1,
        size: float = 3,
        subplot_kws: dict[str, Any] | None = None,
    ) -> None:
        # Initialize the base FacetGrid
        super().__init__(
            data,
            col=col,
            row=row,
            col_wrap=col_wrap,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            aspect=aspect,
            size=size,
            subplot_kws=subplot_kws,
        )

    def _finalize_grid(self, *axlabels) -> None:
        """Finalize the annotations and layout of FacetGrid."""
        if not self._finalized:
            self.set_axis_labels(*axlabels)
            self.set_titles()
            # self.fig.tight_layout() # THIS MUST BE MASKED TO AVOID STILL MODIFY AXIS !

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
        self.cbar = self.fig.colorbar(self._mappables[-1], ax=list(self.axs.flat), **kwargs)
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

    def set_map_layout(self):
        """
        Adjusts the figure height of the plot based on the aspect ratio of cartopy subplots.

        This function is intended to be called after all plotting has been completed. It operates under the assumption
        that all subplots within the figure share the same aspect ratio.

        Adjust the margins with fc.fig.subplots_adjust(wspace,...) before calling
        this function.

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
        nrow, ncol, __, __ = ax.get_subplotspec().get_geometry()

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
