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
"""Command line script to download GPM files by filename."""
import sys
import warnings

import click

warnings.filterwarnings("ignore")
sys.tracebacklimit = 0  # avoid full traceback error if occur


# -------------------------------------------------------------------------.
# Click Command Line Interface decorator
@click.command()
@click.argument("filenames", type=str, nargs=-1, metavar="filename")
@click.option("--product_type", type=str, show_default=True, default="RS")
@click.option("--storage", type=str, show_default=True, default="PPS")
@click.option("--n_threads", type=int, default=4)
@click.option("--transfer_tool", type=str, default="CURL")
@click.option("--progress_bar", type=bool, default=False)
@click.option("--force_download", type=bool, default=False)
@click.option("--remove_corrupted", type=bool, default=True)
@click.option("--verbose", type=bool, default=True)
@click.option("--retry", type=int, default=1)
def download_gpm_files(
    filenames,
    product_type="RS",
    storage="PPS",
    n_threads=4,
    transfer_tool="CURL",
    progress_bar=False,
    force_download=False,
    remove_corrupted=True,
    verbose=True,
    retry=1,
):
    """Download the specified GPM files."""
    from gpm.io.download import download_files

    filenames = list(filenames)  # ensure is a list
    download_files(
        filepaths=filenames,
        product_type=product_type,
        storage=storage,
        n_threads=n_threads,
        transfer_tool=transfer_tool,
        force_download=force_download,
        remove_corrupted=remove_corrupted,
        progress_bar=progress_bar,
        verbose=verbose,
        retry=retry,
    )
