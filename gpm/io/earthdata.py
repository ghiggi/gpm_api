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
"""Search and download NASA Earthdata Cloud granules with earthaccess."""

import datetime
import os
import shutil
import warnings
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit

import earthaccess
from requests.exceptions import ConnectionError, Timeout

from gpm.io.checks import check_product_type, check_product_version
from gpm.io.filter import filter_filepaths
from gpm.io.ges_disc import get_ges_disc_product_directory_tree
from gpm.io.info import get_info_from_filepath
from gpm.io.products import get_nasa_collection_shortname
from gpm.utils.warnings import GPM_Warning


@contextmanager
def _handle_earthdata_connection_errors():
    """Report connection failures during Earthdata login, searches or downloads."""
    try:
        yield
    except (ConnectionError, Timeout) as error:
        msg = (
            "EarthData Cloud is not available right now. Check your internet connection or try again later. "
            f"Error is: {type(error).__name__}: {error}"
        )
        raise RuntimeError(msg) from error


def _login_earthdata():
    """Log in once per session, reusing an existing Earthaccess token."""
    if earthaccess.get_edl_token():
        return
    with _handle_earthdata_connection_errors():
        earthaccess.login()


def define_earthdata_filepath(product, product_type, date, version, filename):
    """Define an Earthdata Cloud filepath from a filename without searching CMR.

    Earthdata Cloud uses the GES DISC product directory tree with a different host.
    """
    product_type = check_product_type(product_type)
    if product_type == "NRT" and "IMERG" not in product:
        raise ValueError("EARTHDATA storage supports NRT products only for IMERG.")
    directory = get_ges_disc_product_directory_tree(product=product, date=date, version=version)
    return f"https://data.gesdisc.earthdata.nasa.gov/data/{directory}/{filename}"


def get_granule_url(granule):
    """Return the primary data link of an Earthdata granule.

    Keep access to ``earthaccess.results.DataGranule`` data links here so that
    changes to the upstream results API only require updating this function.
    """
    links = granule.data_links()
    if not links:
        raise ValueError("The Earthdata granule has no data links.")
    return links[0]


def get_granule_filename(granule):
    """Return the filename from the primary data link of an Earthdata granule."""
    return unquote(os.path.basename(urlsplit(get_granule_url(granule)).path))


def search_earthdata(product, version, start_time, end_time, product_type="RS", bounding_box=None):
    """Find full granules matching the existing GPM filename filters.

    ``bounding_box`` selects intersecting granules; it does not crop files.
    Authentication uses the credentials already configured in ``.netrc``.
    """
    product_type = check_product_type(product_type)
    if product_type == "NRT" and "IMERG" not in product:
        raise ValueError("EARTHDATA storage supports NRT products only for IMERG.")
    short_name = get_nasa_collection_shortname(product)
    if short_name is None:
        raise ValueError(f"No NASA collection short name is defined for {product}.")
    version = check_product_version(version, product)
    _login_earthdata()
    kwargs = {} if bounding_box is None else {"bounding_box": bounding_box}
    with _handle_earthdata_connection_errors():
        results = earthaccess.search_data(
            short_name=short_name,
            version=f"{version:02d}",
            temporal=(start_time, end_time),
            cloud_hosted=True,
            **kwargs,
        )
    # Several CMR records may refer to the same granule. Preserve one per filename.
    granules = {}
    for granule in results:
        try:
            filename = get_granule_filename(granule)
        except ValueError:
            continue
        granules.setdefault(filename, granule)
    filenames = filter_filepaths(
        list(granules),
        product=product,
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
    )
    return [granules[filename] for filename in filenames]


def get_earthdata_daily_filepaths(product, product_type, date, version, bounding_box=None):
    """Search a single day and return links for granules starting on that day.

    Restricting granules by their start date avoids duplicate cross-midnight
    granules when searching consecutive days.
    """
    day_start = datetime.datetime.combine(date, datetime.time())
    day_end = day_start + datetime.timedelta(days=1)
    granules = search_earthdata(
        product,
        version,
        day_start,
        day_end,
        product_type,
        bounding_box=bounding_box,
    )
    return [
        get_granule_url(granule)
        for granule in granules
        if get_info_from_filepath(get_granule_filename(granule))["start_time"].date() == day_start.date()
    ]


def download_earthdata(remote_filepaths, local_filepaths, n_threads=4, progress_bar=True):
    """Download remote URLs into a temporary directory, then move files into the archive.

    Files are matched by name because earthaccess need not return them in request order.
    Temporary files are cleaned up even when earthaccess raises an exception.
    """
    if not remote_filepaths:
        return []
    _login_earthdata()
    with TemporaryDirectory(prefix="gpm-earth-data-") as tmp_dir, _handle_earthdata_connection_errors():
        downloaded = earthaccess.download(
            remote_filepaths,
            local_path=tmp_dir,
            provider="GES_DISC",
            force=True,
            threads=max(1, n_threads),
            show_progress=progress_bar,
        )
        downloaded = {os.path.basename(path): path for path in downloaded if path is not None and os.path.isfile(path)}
        missing = [path for path in local_filepaths if os.path.basename(path) not in downloaded]
        if missing:
            msg = f"GPM-API could not download these files from Earthdata Cloud: {missing}"
            warnings.warn(msg, GPM_Warning, stacklevel=2)
        status = []
        moved = set()
        for local_filepath in local_filepaths:
            if os.path.basename(local_filepath) not in downloaded:
                status.append(0)
                continue
            if local_filepath not in moved:
                source = downloaded[os.path.basename(local_filepath)]
                os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
                shutil.move(source, local_filepath)
                moved.add(local_filepath)
            status.append(1)
    return status
