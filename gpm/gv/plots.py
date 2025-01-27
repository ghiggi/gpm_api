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
"""This module contains plotting utility for SR/GR validation."""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import gpm
from gpm.visualization import plot_colorbar


def compare_maps(
    gdf,
    sr_column,
    gr_column,
    sr_label=None,
    gr_label=None,
    sr_title=None,
    gr_title=None,
    cmap="Spectral_r",
    vmin=None,
    vmax=None,
    grid_color="grey",
    grid_linewidth=0.25,
    unified_color_scale=True,
    shared_colorbar=False,
    subplot_kwargs=None,
    fig_kwargs=None,
    cbar_kwargs=None,
    **plot_kwargs,
):
    """
    Compare and plot side-by-side maps of SR and GR data from a GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame containing the matched SR/GR data.
    sr_column : str
        The column name for SR data.
    gr_column : str
        The column name for GR data.
    sr_label : str, optional
        Label for the SR sensor colorbar.
    gr_label : str, optional
        Label for the GR sensor colorbar.
    cmap : str, optional
        Colormap to use for both plots. The default is "Spectral_r".
    vmin : float, optional
        Minimum value for color scale.
        Defaults to the minimum value across SR and GR data if ``unified_color_scale`` is ``True``.
    vmax : float, optional
        Maximum value for color scale.
        Defaults to the maximum value across SR and GR data if ``unified_color_scale`` is ``True``.
    unified_color_scale: str, optional
        If True and ``vmin`` or ``vmax`` are not specified,
        it automatically set a common vmin/vmax value.
        The default is ``True``.
    shared_colorbar : bool, optional
        If True, a single colorbar is shared across both plots.
    grid_color : str, optional
        Color of the grid lines. The default is "grey".
    grid_linewidth : float, optional
        Linewidth of the grid lines. The default is 0.25.
    cbar_kwargs : dict, optional
        Keyword arguments for customizing the colorbar passed to :py:class:`matplotlib.pyplot.colorbar`.
    fig_kwargs : dict, optional
        Keyword arguments for customizing the figure passed to :py:class:`matplotlib.pyplot.subplots`.
        The default is ``None`` and ``figsize`` is set to (14,7).
    subplot_kwargs : dict, optional
        Keyword arguments for :py:class:`matplotlib.pyplot.subplots`.
        Can include a `projection` key to set a Cartopy CRS.

    Returns
    -------
    None
        Displays SR and GR fields side-by-side.

    """
    # Create default kwargs dictionary
    fig_kwargs = {} if fig_kwargs is None else fig_kwargs
    subplot_kwargs = {} if subplot_kwargs is None else subplot_kwargs
    cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs

    # Define default colorbar label
    if sr_label is None:
        sr_label = sr_column
    if gr_label is None:
        gr_label = gr_column

    # Default common vmin/vmax if not specified
    min_value = np.nanmin([gdf[sr_column].min(), gdf[gr_column].min()])
    max_value = np.nanmax([gdf[sr_column].max(), gdf[gr_column].max()])
    if unified_color_scale:
        if vmax is None:
            vmax = max_value
        if vmin is None:
            vmin = min_value

    # Determine the extent for the plots
    extent_xy = gdf.total_bounds[[0, 2, 1, 3]]

    # Define cbar_kwargs
    cbar_kwargs1 = cbar_kwargs.copy()
    cbar_kwargs1["label"] = sr_label
    cbar_kwargs2 = cbar_kwargs.copy()
    cbar_kwargs2["label"] = gr_label

    # Create the figure
    if "figsize" not in fig_kwargs:
        fig_kwargs["figsize"] = (14, 7)
    fig, axes = plt.subplots(1, 2, subplot_kw=subplot_kwargs, **fig_kwargs)

    # Plot SR data
    _ = plot_gdf_map(
        ax=axes[0],
        gdf=gdf,
        column=sr_column,
        title=sr_title,
        extent_xy=extent_xy,
        # Gridline settings
        grid_linewidth=grid_linewidth,
        grid_color=grid_color,
        # Colorbar settings
        add_colorbar=not shared_colorbar,
        cbar_kwargs=cbar_kwargs1,
        # Plot settings
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **plot_kwargs,
    )

    # Plot GR data
    _ = plot_gdf_map(
        ax=axes[1],
        gdf=gdf,
        column=gr_column,
        title=gr_title,
        extent_xy=extent_xy,
        # Gridline settings
        grid_linewidth=grid_linewidth,
        grid_color=grid_color,
        # Colorbar settings
        add_colorbar=True,
        cbar_kwargs=cbar_kwargs2,
        # Plot settings
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **plot_kwargs,
    )

    return fig


def plot_gdf_map(
    gdf,
    column,
    extent_xy=None,
    title=None,
    grid_color="grey",
    grid_linewidth=0.25,
    add_colorbar=True,
    cbar_kwargs=None,
    **plot_kwargs,
):
    # Set default extent_xy
    if extent_xy is None:
        extent_xy = gdf.total_bounds[[0, 2, 1, 3]]
    # Set default title
    if title is None:
        title = column

    # Retrieve default plot kwargs
    plot_kwargs, cbar_kwargs = gpm.get_plot_kwargs(
        name=column,
        user_cbar_kwargs=cbar_kwargs,
        user_plot_kwargs=plot_kwargs,
    )

    # Plot data
    p = gdf.plot(
        column=column,
        legend=False,
        **plot_kwargs,
    )
    p.axes.set_xlim(extent_xy[0:2])
    p.axes.set_ylim(extent_xy[2:4])
    p.axes.set_title(title)

    # Convert x and y axis tick labels to kilometers
    x_formatter = mticker.FuncFormatter(lambda x, pos: f"{x/1000:.0f}")  # noqa
    y_formatter = mticker.FuncFormatter(lambda y, pos: f"{y/1000:.0f}")  # noqa
    p.axes.xaxis.set_major_formatter(x_formatter)
    p.axes.yaxis.set_major_formatter(y_formatter)

    # Set axis labels
    p.axes.set_xlabel("x (km)", fontsize=12)
    p.axes.set_ylabel("y (km)", fontsize=12)

    # Set aspect ratio
    p.axes.set_aspect("equal")

    # Set grid line
    p.axes.grid(lw=grid_linewidth, color=grid_color)

    # Add colorbar
    if add_colorbar:
        plot_colorbar(p=p.collections[0], ax=p.axes, **cbar_kwargs)
    return p


def reflectivity_scatterplot(
    df,
    gr_z_column,
    sr_z_column,
    hue_column,
    ax=None,
    gr_range=None,
    sr_range=None,
    add_colorbar=True,
    marker="+",
    title="GR / SR Offset",
    cbar_kwargs=None,
    **plot_kwargs,
):
    """
    Plots a scatterplot comparing GR and SR reflectivity columns.

    Parameters
    ----------
    df : pandas.DataFrame, geopandas.GeoDataFrame
        The dataframe containing the reflectivity data.
    gr_z_column : str
        The column name for the GR (ground radar) reflectivity data on the x-axis.
    sr_z_column : str
        The column name for the SR (spaceborne radar) reflectivity data on the y-axis.
    hue_column : str
        The column name used for color mapping in the scatter plot.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If ``None``, a new figure and axis are created.
    gr_range : list, optional
        The limits for the x-axis as [min, max]. If ``None``, limits are determined from data.
    sr_range : list, optional
        The limits for the y-axis as [min, max]. If ``None``, limits are determined from data.
    add_colorbar : bool, optional
        If True, adds a colorbar to the plot. The default is True.
    marker : str, optional
        The marker style for the scatter plot. The default is ``"+"``.
    title : str, optional
        The title of the scatter plot. The default is "GR / SR Offset".
    cbar_kwargs : dict, optional
        Additional keyword arguments for the colorbar.
    **plot_kwargs : dict
        Additional keyword arguments passed to ``ax.scatter``.

    Returns
    -------
    matplotlib.collections.PathCollection
        The scatter plot object.

    """
    cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs

    # Initialize plot if necessary
    if ax is None:
        # Create a new figure and axis
        _, ax = plt.subplots()

    # Define plot limits if not specified
    if gr_range is None:
        gr_range = [df[gr_z_column].min() - 2, df[gr_z_column].max() + 2]
    if sr_range is None:
        sr_range = [df[sr_z_column].min() - 2, df[sr_z_column].max() + 2]

    # Retrieve plot_kwargs, cbar_kwargs
    plot_kwargs, cbar_kwargs = gpm.get_plot_kwargs(
        name=hue_column,
        user_plot_kwargs=plot_kwargs,
        user_cbar_kwargs=cbar_kwargs,
    )
    # Display scatterplot
    p = ax.scatter(
        df[gr_z_column],
        df[sr_z_column],
        c=df[hue_column],
        marker=marker,
        **plot_kwargs,
    )
    # Add colorbar
    if add_colorbar:
        plot_colorbar(p=p, ax=p.axes, **cbar_kwargs)
    # Add 1:1 line
    ax.plot([-10, 70], [-10, 70], linestyle="solid", color="black")
    # Restrict limits
    ax.set_xlim(*gr_range)
    ax.set_ylim(*sr_range)
    # Add labels
    ax.set_xlabel("GR reflectivity (dBZ)")
    ax.set_ylabel("SR reflectivity (dBZ)")
    # Add title
    ax.set_title(title)
    # Set aspect ratio
    p.axes.set_aspect("auto")
    return p


def reflectivity_scatterplots(
    df,
    gr_z_column,
    sr_z_column,
    hue_columns,
    ncols=2,
    fig_kwargs=None,
    gr_range=None,
    sr_range=None,
    marker="+",
    cbar_kwargs=None,
    **plot_kwargs,
):
    """
    Plots multiple scatter plots comparing GR and SR reflectivity for each specified hue variable.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the reflectivity data.
    gr_z_column : str
        The column name for the GR (ground radar) reflectivity data on the x-axis.
    sr_z_column : str
        The column name for the SR (spaceborne radar) reflectivity data on the y-axis.
    hue_variables : list of str
        A list of column names to be used as hue variables for individual scatter plots.
    ncols : int, optional
        Number of columns in the subplot grid. The default is 2.
    gr_range : list, optional
        The limits for the x-axis as [min, max]. If None, limits are determined from data.
    sr_range : list, optional
        The limits for the y-axis as [min, max]. If None, limits are determined from data.
    marker : str, optional
        The marker style for the scatter plot. The default is "+".
    fig_kwargs : dict, optional
        Keyword arguments for customizing the figure passed to :py:class:`matplotlib.pyplot.subplots`.
        If ``figsize`` is not specified, defaults to (ncols * 6, nrows * 5).
    cbar_kwargs : dict, optional
        Additional keyword arguments common to all colorbars.
    **plot_kwargs : dict
        Additional keyword arguments common to all scatter plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing all the scatter plots.

    """
    # Deal with default kwargs
    fig_kwargs = {} if fig_kwargs is None else fig_kwargs
    cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
    # Deal with case with 1 hue column
    if isinstance(hue_columns, list) and len(hue_columns) == 1:
        hue_columns = hue_columns[0]
    if isinstance(hue_columns, str):
        p = reflectivity_scatterplot(
            df=df,
            gr_z_column=gr_z_column,
            sr_z_column=sr_z_column,
            hue_column=hue_columns,
            ax=plot_kwargs.pop("ax", None),
            gr_range=gr_range,
            sr_range=sr_range,
            marker=marker,
            cbar_kwargs=cbar_kwargs,
            **plot_kwargs,
        )
        return p
    # Identify number of rows required
    nrows = (len(hue_columns) + 1) // ncols  # Calculate the number of rows needed
    # Specify default fig size if not specified
    if "figsize" not in fig_kwargs:
        fig_kwargs["figsize"] = (ncols * 6, nrows * 5)
    # Define plot limits if not specified
    if gr_range is None:
        gr_range = [df[gr_z_column].min() - 2, df[gr_z_column].max() + 2]
    if sr_range is None:
        sr_range = [df[sr_z_column].min() - 2, df[sr_z_column].max() + 2]
    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, **fig_kwargs)
    axes = axes.flatten()  # Flatten axes array to make iteration easier
    for i, hue_column in enumerate(hue_columns):
        ax = axes[i]
        reflectivity_scatterplot(
            df=df,
            gr_z_column=gr_z_column,
            sr_z_column=sr_z_column,
            hue_column=hue_column,
            ax=ax,
            gr_range=gr_range,
            sr_range=sr_range,
            marker=marker,
            title=hue_column,
            cbar_kwargs=cbar_kwargs.copy(),
            **plot_kwargs.copy(),
        )
        # Only set x-labels for bottom row and y-labels for left column
        if i % ncols != 0:
            ax.set_ylabel("")
            ax.set_yticks([])
            ax.set_yticklabels([])
        if i < (nrows - 1) * ncols:
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.set_xticklabels([])

    # Turn off axes for any empty subplots (in case the grid is larger than the number of plots)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    return fig


def reflectivity_distribution(
    df,
    gr_z_column,
    sr_z_column,
    ax=None,
    bin_width=2,
):
    """
    Plots overlaid histograms of GR and SR reflectivity distributions.

    Parameters
    ----------
    df : pandas.DataFrame, geopandas.GeoDataFrame
        The dataframe containing the reflectivity data.
    gr_z_column : str
        The column name for the GR (ground radar) reflectivity data.
    sr_z_column : str
        The column name for the SR (spaceborne radar) reflectivity data.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If ``None``, a new figure and axis are created.
    bin_width : float, optional
        The width of the histogram bins. The default is 2 dBZ.

    Returns
    -------
    tuple
        The histogram plot objects for GR and SR.

    """
    # Initialize plot if necessary
    if ax is None:
        # Create a new figure and axis
        _, ax = plt.subplots()

    # Default vmin/vmax
    vmin = np.nanmin([df[gr_z_column].min(), df[sr_z_column].min()])
    vmax = np.nanmax([df[gr_z_column].max(), df[sr_z_column].max()])

    # Plot histograms
    p = ax.hist(
        df[gr_z_column],
        bins=np.arange(vmin, vmax, bin_width),
        edgecolor="None",
        label="GR",
    )
    p = ax.hist(
        df[sr_z_column],
        bins=np.arange(vmin, vmax, bin_width),
        edgecolor="red",
        facecolor="None",
        label="SR",
    )
    ax.set_xlabel("Reflectivity (dBZ)")
    ax.legend()
    return p


def calibration_summary(
    df,
    gr_z_column,
    sr_z_column,
    # Scatterplot options
    hue_column,
    gr_range=None,
    sr_range=None,
    marker="+",
    cbar_kwargs=None,
    # Histogram options
    bin_width=2,
    **plot_kwargs,
):
    """
    Creates a summary plot for comparison between GR and SR reflectivity data.

    The function draw a scatter plot and overlaid histograms of GR and SR reflectivity values.

    Parameters
    ----------
    df : pandas.DataFrame, geopandas.GeoDataFrame
        The dataframe containing the reflectivity data.
    gr_z_column : str
        The column name for the GR (ground radar) reflectivity data.
    sr_z_column : str
        The column name for the SR (spaceborne radar) reflectivity data.
    hue_column : str
        The column name used for color mapping in the scatter plot.
    gr_range : list, optional
        The limits for the x-axis in the scatter plot as [min, max]. If ``None``, limits are determined from data.
    sr_range : list, optional
        The limits for the y-axis in the scatter plot as [min, max]. If ``None``, limits are determined from data.
    marker : str, optional
        The marker style for the scatter plot. The default is ``"+"``.
    cbar_kwargs : dict, optional
        Additional keyword arguments for the colorbar.
    bin_width : float, optional
        The width of the histogram bins. The default is 2 dBZ.
    **plot_kwargs : dict
        Additional keyword arguments passed to the scatter plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the calibration summary plots.

    """
    cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
    # Compute bias
    dz = df[gr_z_column] - df[sr_z_column]

    # Compute bias statistics
    bias = np.nanmean(dz).round(2)
    rob_bias = np.nanmedian(dz).round(2)

    # - Histograms
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, aspect="equal")
    _ = reflectivity_scatterplot(
        df=df,
        gr_z_column=gr_z_column,
        sr_z_column=sr_z_column,
        hue_column=hue_column,
        gr_range=gr_range,
        sr_range=sr_range,
        marker=marker,
        ax=ax1,
        cbar_kwargs=cbar_kwargs,
        **plot_kwargs,
    )
    ax1.set_title(f"GR-SR (Robust) Bias: ({rob_bias}) {bias} dBZ")
    ax2 = fig.add_subplot(122)
    _ = reflectivity_distribution(
        df=df,
        gr_z_column=gr_z_column,
        sr_z_column=sr_z_column,
        ax=ax2,
        bin_width=bin_width,
    )

    fig.suptitle("GR / SR Calibration Summary")
    fig.tight_layout()
    return fig
