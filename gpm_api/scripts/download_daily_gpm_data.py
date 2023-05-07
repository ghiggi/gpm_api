#!/usr/bin/env python3
"""
Created on Mon Mar 13 11:48:22 2023

@author: ghiggi
"""
import sys
import warnings

import click

from gpm_api.io import GPM_VERSION  # CURRENT GPM VERSION

warnings.filterwarnings("ignore")
sys.tracebacklimit = 0  # avoid full traceback error if occur

# -------------------------------------------------------------------------.
# Click Command Line Interface decorator
@click.command()
@click.argument("product", type=str)
@click.argument("year", type=int)
@click.argument("month", type=int)
@click.argument("day", type=int)
@click.option("--product_type", type=str, show_default=True, default="RS")
@click.option("--version", type=int, show_default=True, default=GPM_VERSION)
@click.option("--n_threads", type=int, default=2)
@click.option("--transfer_tool", type=str, default="curl")
@click.option("--progress_bar", type=bool, default=False)
@click.option("--force_download", type=bool, default=False)
@click.option("--check_integrity", type=bool, default=True)
@click.option("--remove_corrupted", type=bool, default=True)
@click.option("--verbose", type=bool, default=True)
@click.option("--retry", type=int, default=1)
@click.option("--base_dir", type=str, default=None)
@click.option("--username", type=str, default=None)
@click.option("--password", type=str, default=None)
def download_daily_gpm_data(
    product,
    year,
    month,
    day,
    product_type="RS",
    version=GPM_VERSION,
    n_threads=2,
    transfer_tool="curl",
    progress_bar=False,
    force_download=False,
    check_integrity=True,
    remove_corrupted=True,
    verbose=True,
    retry=1,
    base_dir=None,
    username=None,
    password=None,
):
    """Download the GPM product for a specific date."""
    from gpm_api.utils.archive import download_daily_data

    _ = download_daily_data(
        product=product,
        year=year,
        month=month,
        day=day,
        product_type=product_type,
        version=version,
        n_threads=n_threads,
        transfer_tool=transfer_tool,
        progress_bar=progress_bar,
        force_download=force_download,
        check_integrity=check_integrity,
        remove_corrupted=remove_corrupted,
        verbose=verbose,
        retry=retry,
        base_dir=base_dir,
        username=username,
        password=password,
    )

    return


if __name__ == "__main__":
    download_daily_gpm_data()
