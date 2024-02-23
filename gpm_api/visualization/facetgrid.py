#!/usr/bin/env python3
"""
Created on Fri Feb 23 17:14:18 2024

@author: ghiggi
"""
from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
import xarray as xr
from xarray.plot.facetgrid import FacetGrid
from xarray.plot.utils import label_from_attrs


def remove_gridline_bottom_labels(ax):
    gl = ax._gridliners[0]
    gl.bottom_labels = False


def remove_gridline_left_labels(ax):
    gl = ax._gridliners[0]
    gl.left_labels = False


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
        # TODO: improve to avoid tight_layout issues !
        kwargs = kwargs.copy()
        if self._cmap_extend is not None:
            kwargs.setdefault("extend", self._cmap_extend)
        # dont pass extend as kwarg if it is in the mappable
        if hasattr(self._mappables[-1], "extend"):
            kwargs.pop("extend", None)
        if "label" not in kwargs:
            assert isinstance(self.data, xr.DataArray)
            kwargs.setdefault("label", label_from_attrs(self.data))
        ticklabels = kwargs.pop("ticklabels", None)
        self.cbar = self.fig.colorbar(self._mappables[-1], ax=list(self.axs.flat), **kwargs)
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
        for ax in self.axs.flat:
            try:
                ax.set_extent(extent)
            except Exception:
                pass
