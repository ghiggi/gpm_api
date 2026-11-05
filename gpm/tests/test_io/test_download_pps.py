"""Live PPS download tests requiring PPS credentials."""

import os

import pytest
import xarray as xr

import gpm


@pytest.mark.integration
def test_download_pps(tmp_path):
    """Download a real IMERG granule using PPS credentials from config or CI."""
    username = os.environ.get("PPS_USERNAME") or gpm.config.get("username_pps")
    password = os.environ.get("PPS_PASSWORD") or gpm.config.get("password_pps")
    if not username or not password:
        pytest.skip("Configure username_pps/password_pps in gpm.config or set PPS_USERNAME and PPS_PASSWORD.")

    base_dir = tmp_path / "GPM"
    with gpm.config.set({"base_dir": str(base_dir), "username_pps": username, "password_pps": password}):
        corrupted = gpm.download(
            product="IMERG-FR",
            product_type="RS",
            storage="PPS",
            version=7,
            start_time="2017-01-01 20:00",
            end_time="2017-01-01 20:30",
            n_threads=1,
            progress_bar=False,
            verbose=False,
        )

    filepath = (
        base_dir
        / "RS"
        / "V07"
        / "IMERG"
        / "IMERG-FR"
        / "2017"
        / "01"
        / "01"
        / "3B-HHR.MS.MRG.3IMERG.20170101-S200000-E202959.1200.V07B.HDF5"
    )
    assert corrupted == []
    assert filepath.is_file()
    assert filepath.stat().st_size > 0
    with xr.open_dataset(filepath, engine="netcdf4", group="Grid") as ds:
        assert ds.data_vars
        ds.isel(dict.fromkeys(ds.sizes, 0)).load()
