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
"""GPM-API Package."""

import contextlib
import os
from importlib.metadata import PackageNotFoundError, version

# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import pycolorbar
from pycolorbar import get_plot_kwargs  # noqa

import gpm.accessor  # noqa
from gpm._config import config  # noqa
from gpm.configs import (  # noqa
    define_configs,
    read_configs,
)
from gpm.dataset.dataset import open_dataset  # noqa
from gpm.dataset.datatree import open_datatree  # noqa
from gpm.dataset.granule import open_granule  # noqa
from gpm.io.download import download_archive as download  # noqa
from gpm.io.download import (  # noqa
    download_daily_data,
    download_files,
    download_monthly_data,
)
from gpm.io.find import find_filepaths as find_files  # noqa
from gpm.io.products import (  # noqa
    available_product_categories,
    available_product_levels,
    available_products,
    available_satellites,
    available_scan_modes,
    available_sensors,
    available_versions,
    get_product_end_time,
    get_product_start_time,
)
from gpm.utils.checks import (  # noqa
    check_contiguous_scans,
    check_missing_granules,
    check_regular_time,
    check_valid_geolocation,
)
from gpm.visualization.plot import plot_labels, plot_patches  # noqa

_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Register colormaps and colorbars
_colorbar_registered = False
if not _colorbar_registered:
    pycolorbar.register_colormaps(os.path.join(pycolorbar.etc_directory, "colormaps"))
    pycolorbar.register_colormaps(
        os.path.join(_root_path, "gpm", "etc", "colormaps"),
        verbose=False,
    )
    pycolorbar.register_colorbars(os.path.join(_root_path, "gpm", "etc", "colorbars"))

colormaps = pycolorbar.colormaps
colorbars = pycolorbar.colorbars


# Get version
with contextlib.suppress(PackageNotFoundError):
    __version__ = version("gpm_api")
