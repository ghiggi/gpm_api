from gpm_api.dataset import crs
from pyproj import CRS
import pytest
from pytest_mock import MockerFixture
import xarray as xr


def test_get_pyproj_crs_cf_fict_private() -> None:
    """Test that a dictionary is returned with the spatial_ref key"""
    res = crs._get_pyproj_crs_cf_dict(CRS(4326))  # WGS84

    assert isinstance(res, dict), "Dictionary not returned"
    assert "spatial_ref" in res.keys(), "spatial_ref key not in dictionary"


def test_get_proj_coord_unit_private() -> None:
    """Test that the coordinate unit is returned when given projected CRS"""

    # Projected CRS
    projected_crs = CRS(32661)  # WGS84 / UTM zone 61N, metre

    # Test both dimensions
    for dimension in [0, 1]:
        res = crs._get_proj_coord_unit(projected_crs, dim=dimension)
        assert res == "metre"

    # Projected WGS 84 in feet
    projected_crs = CRS(8035)  # WGS 84 / UPS North (E,N), US Survey foot
    # Test both dimensions
    for dimension in [0, 1]:
        res = crs._get_proj_coord_unit(projected_crs, dim=dimension)
        assert res is not None
        assert res != "metre"  # Result should be "{unitfactor} metre"
        assert "metre" in res  # Return should still contain metre as string
        assert len(res.split(" ")) == 2  # Should be two parts
        assert float(res.split(" ")[0]) == pytest.approx(
            1200 / 3937
        )  # Survey foot is 1200/3937 metres


# def test_get_obj_private(sample_dataset: xr.Dataset) -> None:
#     """Test that the dataset is copied when given a dataset"""

#     crs._get_obj(sample_dataset, dim=0)
