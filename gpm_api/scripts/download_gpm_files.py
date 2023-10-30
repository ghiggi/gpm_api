#!/usr/bin/env python3
"""
Created on Mon Mar 13 12:11:27 2023

@author: ghiggi
"""
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
@click.option("--storage", type=str, show_default=True, default="pps")
@click.option("--n_threads", type=int, default=4)
@click.option("--transfer_tool", type=str, default="curl")
@click.option("--progress_bar", type=bool, default=False)
@click.option("--force_download", type=bool, default=False)
@click.option("--remove_corrupted", type=bool, default=True)
@click.option("--verbose", type=bool, default=True)
@click.option("--retry", type=int, default=1)
def download_gpm_files(
    filenames,
    product_type="RS",
    storage="pps",
    n_threads=4,
    transfer_tool="curl",
    progress_bar=False,
    force_download=False,
    remove_corrupted=True,
    verbose=True,
    retry=1,
):
    """Download the specified GPM files."""
    from gpm_api.io.download import download_files

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

    return


if __name__ == "__main__":
    download_gpm_files()
