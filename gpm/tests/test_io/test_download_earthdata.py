"""Live Earthdata Cloud download tests requiring Earthdata credentials."""

import os

import pytest
import xarray as xr
from earthaccess.auth import netrc_path

import gpm


@pytest.mark.integration
def test_download_earthdata(tmp_path):
    """Download a real IMERG granule and verify its archive path and contents."""
    has_environment_credentials = os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD")
    if not has_environment_credentials and not netrc_path().is_file():
        pytest.skip("Configure Earthdata credentials in .netrc or set EARTHDATA_USERNAME and EARTHDATA_PASSWORD.")

    base_dir = tmp_path / "GPM"
    with gpm.config.set({"base_dir": str(base_dir)}):
        corrupted = gpm.download(
            product="IMERG-FR",
            product_type="RS",
            storage="EARTHDATA",
            version=7,
            start_time="2017-01-01 20:00",
            end_time="2017-01-01 20:30",
            n_threads=1,
            progress_bar=False,
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
        # Read a grid cell from each variable to verify data access, beyond metadata.
        ds.isel(dict.fromkeys(ds.sizes, 0)).load()
