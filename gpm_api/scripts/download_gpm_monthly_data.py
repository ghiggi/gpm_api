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
@click.argument("product", type=str)
@click.argument("year", type=int)
@click.argument("month", type=int)
@click.option("--product_type", type=str, show_default=True, default="RS")
@click.option("--storage", type=str, show_default=True, default="pps")
@click.option("--version", type=int, show_default=True, default=None)
@click.option("--n_threads", type=int, default=4)
@click.option("--transfer_tool", type=str, default="curl")
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
    storage="pps",
    version=None,
    n_threads=4,
    transfer_tool="curl",
    progress_bar=False,
    force_download=False,
    check_integrity=True,
    remove_corrupted=True,
    verbose=True,
    retry=1,
):
    """Download the GPM product for a specific month."""
    from gpm_api.io.download import download_monthly_data

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

    return


if __name__ == "__main__":
    download_gpm_monthly_data()
