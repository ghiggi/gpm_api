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
"""This module test the GPM-API Dataset CRS."""

import pytest
from pyproj import CRS

from gpm.dataset import crs


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
