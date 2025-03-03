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
"""This module contains plotting functions for exploratory data visualization."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import date2num


def plot_boxplot(
    df_stats,
    ax=None,
    label=None,
    showfliers=False,
    showwhisker=False,
    showmeans=False,
    positions=None,
    widths=0.6,
    add_median_points=False,
    add_median_line=False,
    median_points_kwargs=None,
    median_line_kwargs=None,
    boxprops=None,
    whiskerprops=None,
    medianprops=None,
    **kwargs,
):
    """
    Draw a box and whisker plot from pre-computed statistics.

    The box extends from the first quartile *q25* to the third
    quartile *q75* of the data, with a line at the median (*median*).
    The whiskers extend from *whislow* to *whishi*.
    Flier points are markers past the end of the whiskers.
    See https://en.wikipedia.org/wiki/Box_plot for reference.

    .. code-block:: none

               whislow    q25    med    q75    whishi
                           |-----:-----|
           o      |--------|     :     |--------|    o  o
                           |-----:-----|
         flier                                      fliers

    .. note::
        This is a low-level drawing function for when you already
        have the statistical parameters. If you want a boxplot based
        on a dataset, use `matplotlib.Axes.boxplot` instead.

    Parameters
    ----------
    df_stats : pandas.DataFrame
       The dataframe index controls the boxplot position along the xaxis,
       unless 'positions' is specified.
       Required columns are:
       - 'mean':
       - 'median':
       - 'q10':
       - 'q25':
       - 'q75':
       - 'q90':
       - 'iqr':  Needed if ``showwhisker=True``.
       - 'min':  Needed if ``showfliers=True``.
       - 'max':  Needed if ``showfliers=True``.
       - 'n':

       A DatetimeIndex or a 'time' column enable to display the x axis
       with the desired date (irregular vs uniformly spaced)

    label: str, optional
       Column of the dataframe to be used as tick label for the boxplot

    positions : array-like, optional
       The positions of the boxes. If not specified, the ticks and limits
       are automatically set as function of the dataframe index.

    widths : float or array-like, optional
       The widths of the boxes.  The default is
       ``clip(0.15*(distance between extreme positions), 0.15, 0.5)``.

    capwidths : float or array-like
       Either a scalar or a vector and sets the width of each cap.
       The default is ``0.5*(width of the box)``, see *widths*.

    orientation : str, optional
       Either 'vertical' or 'horizontal'.
       If 'horizontal', plots the boxes horizontally. Otherwise, plots the boxes vertically.
       The default is 'vertical'.

    patch_artist : bool, optional
       If `False` produces boxes with the `.Line2D` artist.
       If `True` produces boxes with the `~matplotlib.patches.Patch` artist.
       The default is False.

    shownotches, showmeans, showcaps, showbox, showfliers : bool
       Whether to draw the CI notches, the mean value (both default to
       False), the caps, the box, and the fliers (all three default to
       True).

    boxprops, whiskerprops, capprops, flierprops, medianprops, meanprops : dict, optional
       Artist properties for the boxes, whiskers, caps, fliers, medians, and
       means.

    manage_ticks : bool, optional
       If True (the default), the tick locations and labels will be adjusted to match the
       boxplot positions.

    zorder : float
       The zorder of the resulting boxplot.

    Returns
    -------
    matplotlib.Axes

    See Also
    --------
    boxplot : Draw a boxplot from data instead of pre-computed statistics.
    boxplot_stat: https://github.com/matplotlib/matplotlib/blob/b5ac96a8980fdb9e59c9fb649e0714d776e26701/lib/matplotlib/cbook/__init__.py#L1103
    """
    # Ensure sorted index
    df_stats = df_stats.sort_index()

    # Define default properties
    medianprops = {"color": "black"} if medianprops is None else medianprops
    whiskerprops = {} if whiskerprops is None else whiskerprops

    # Compute IQR if not already a column
    if "iqr" not in df_stats:
        df_stats["iqr"] = df_stats["q75"] - df_stats["q25"]

    # Prepare data for bxp
    box_data = []
    for i in range(len(df_stats)):
        df_row = df_stats.iloc[i]
        box_dict = {
            "q1": df_row["q25"].item(),
            "med": df_row["median"].item(),
            "q3": df_row["q75"].item(),
            # "whislo": df_row["q10"].item(),
            # "whishi": df_row["q90"].item(),
            "whislo": np.maximum(df_row["q25"].item() - 1.5 * df_row["iqr"].item(), df_row["min"].item()),
            "whishi": np.minimum(df_row["q75"].item() + 1.5 * df_row["iqr"].item(), df_row["max"].item()),
            "fliers": [df_row["min"].item(), df_row["max"].item()],
            "mean": df_row["mean"].item(),
        }
        # Add label
        if label is not None:
            box_dict["label"] = str(df_row[label])
        box_data.append(box_dict)

    # Disable show_whisker
    if not showwhisker:
        whiskerprops["alpha"] = 0

    # Define positions
    is_datetime_index = False
    if positions is None:
        is_datetime_index = isinstance(df_stats.index, pd.DatetimeIndex)
        if is_datetime_index:
            positions = np.asarray(df_stats.index)
            positions = date2num(positions)
            # position_index = df_stats.index.astype(int)
            # positions = position_index - position_index.min()
            # positions = positions/positions.max()*len(positions)
        else:
            positions = range(len(df_stats))

    # Create the boxplot with bxp
    if ax is None:
        fig, ax = plt.subplots()
    bplot = ax.bxp(
        box_data,
        positions=positions,
        widths=widths,
        showmeans=showmeans,
        showfliers=showfliers,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        **kwargs,
    )

    # Add median points
    if add_median_points:
        median_points_kwargs = {} if median_points_kwargs is None else median_points_kwargs
        ax.scatter(positions, df_stats["median"], **median_points_kwargs)

    # Add line between median points
    if add_median_line:
        median_line_kwargs = {} if median_line_kwargs is None else median_line_kwargs
        ax.plot(positions, df_stats["median"], **median_line_kwargs)
    return bplot
