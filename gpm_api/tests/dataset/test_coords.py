import pytest
import xarray as xr
from gpm_api.dataset import coords


def test_set_coords_attrs(sample_dataset: xr.Dataset) -> None:
    res = coords.set_coords_attrs(sample_dataset)

    assert res == sample_dataset


def test_get_coords_attrs_dict(sample_dataset: xr.Dataset) -> None:
    res = coords.get_coords_attrs_dict(sample_dataset)

    for key in [
        "lat",
        "lon",
        "time",
        "gpm_id",
        "gpm_granule_id",
        "gpm_cross_track_id",
        "gpm_along_track_id",
    ]:
        assert key in res.keys(), f"{key} not in dictionary"
