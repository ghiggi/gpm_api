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
"""Command line script to download monthly GPM data."""
import sys
import warnings

import click

warnings.filterwarnings("ignore")
sys.tracebacklimit = 0  # avoid full traceback error if occur


# -------------------------------------------------------------------------.
# Click Command Line Interface decorator
@click.command()
@click.argument("product", type=str)
@click.argument("year", type=int)
@click.argument("month", type=int)
@click.option("--product_type", type=str, show_default=True, default="RS")
@click.option("--storage", type=str, show_default=True, default="PPS")
@click.option("--version", type=int, show_default=True, default=None)
@click.option("--n_threads", type=int, default=4)
@click.option("--transfer_tool", type=str, default="CURL")
@click.option("--progress_bar", type=bool, default=False)
@click.option("--force_download", type=bool, default=False)
@click.option("--check_integrity", type=bool, default=True)
@click.option("--remove_corrupted", type=bool, default=True)
@click.option("--verbose", type=bool, default=True)
@click.option("--retry", type=int, default=1)
def download_gpm_monthly_data(
    product,
    year,
    month,
    product_type="RS",
    storage="PPS",
    version=None,
    n_threads=4,
    transfer_tool="CURL",
    progress_bar=False,
    force_download=False,
    check_integrity=True,
    remove_corrupted=True,
    verbose=True,
    retry=1,
):
    """Download the GPM product for a specific month."""
    from gpm.io.download import download_monthly_data

    _ = download_monthly_data(
        product=product,
        year=year,
        month=month,
        product_type=product_type,
        version=version,
        storage=storage,
        n_threads=n_threads,
        transfer_tool=transfer_tool,
        progress_bar=progress_bar,
        force_download=force_download,
        check_integrity=check_integrity,
        remove_corrupted=remove_corrupted,
        verbose=verbose,
        retry=retry,
    )
