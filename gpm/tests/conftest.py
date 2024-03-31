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
"""This module defines pytest fixtures available across all test modules."""
import datetime
import ntpath as ntp
import posixpath as pxp
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import xarray as xr
from pytest_mock import MockerFixture

from gpm.io.products import get_info_dict
from gpm.tests.utils.fake_datasets import get_grid_dataarray, get_orbit_dataarray
from gpm.utils import geospatial


@pytest.fixture
def product_types() -> List[str]:
    """Return a list of all product types from the info dict"""
    from gpm.io.products import get_available_product_types

    return get_available_product_types()


@pytest.fixture
def product_categories() -> List[str]:
    """Return a list of product categories from the info dict"""
    from gpm.io.products import get_available_product_categories

    return get_available_product_categories()


@pytest.fixture
def product_levels() -> List[str]:
    """Return a list of product levels from the info dict"""
    from gpm.io.products import get_available_product_levels

    return get_available_product_levels(full=False)  #  ["1A", "1B", "1C", "2A", "2B", "3B"]


@pytest.fixture
def full_product_levels() -> List[str]:
    """Return a list of full product levels from the info dict"""
    from gpm.io.products import get_available_product_levels

    return get_available_product_levels(
        full=True
    )  # ["1A", "1B", "1C", "2A", "2A-CLIM", "2A-ENV", "2B", "3B-HHR""]


@pytest.fixture
def sensors() -> List[str]:
    """Return a list of sensors from the info dict"""
    from gpm.io.products import get_available_sensors

    return get_available_sensors()  # ['AMSR2', 'AMSRE', 'AMSUB', 'ATMS', 'DPR', ...]


@pytest.fixture
def satellites() -> List[str]:
    """Return a list of satellites from the info dict"""
    from gpm.io.products import get_available_satellites

    return get_available_satellites()  # ['GCOMW1', 'GPM', 'METOPA', 'METOPB',  'METOPC', ...]


@pytest.fixture
def versions() -> List[int]:
    """Return a list of versions"""
    from gpm.io.products import get_available_versions

    return get_available_versions()


@pytest.fixture
def products() -> List[str]:
    """Return a list of all products regardless of type"""
    from gpm.io.products import get_available_products

    return get_available_products()


@pytest.fixture
def product_info() -> Dict[str, dict]:
    """Return a dictionary of product info"""

    return get_info_dict()


@pytest.fixture
def remote_filepaths() -> Dict[str, Dict[str, Any]]:
    """Return a list of probable GPM server paths"""

    # Not validated to be real paths but follow the structure
    return {
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S170044-E183317.036092.V07A.HDF5": {  # noqa
            "year": 2020,
            "month": 7,
            "day": 5,
            "product": "2A-DPR",
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 17, 0, 44),
            "end_time": datetime.datetime(2020, 7, 5, 18, 33, 17),
            "version": 7,
            "granule_id": 36092,
        },
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S183318-E200550.036093.V07A.HDF5": {  # noqa
            "year": 2020,
            "month": 7,
            "day": 5,
            "product": "2A-DPR",
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 18, 33, 18),
            "end_time": datetime.datetime(2020, 7, 5, 20, 5, 50),
            "version": 7,
            "granule_id": 36093,
        },
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S200551-E213823.036094.V07A.HDF5": {  # noqa
            "year": 2020,
            "month": 7,
            "day": 5,
            "product": "2A-DPR",
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 20, 5, 51),
            "end_time": datetime.datetime(2020, 7, 5, 21, 38, 23),
            "version": 7,
            "granule_id": 36094,
        },
        # Over two days
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S231057-E004329.036096.V07A.HDF5": {  # noqa
            "year": 2020,
            "month": 7,
            "day": 5,
            "product": "2A-DPR",
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 23, 10, 57),
            "end_time": datetime.datetime(2020, 7, 6, 0, 43, 29),
            "version": 7,
            "granule_id": 36096,
        },
        # NRT
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S170044-E183317.V07A.HDF5": {  # noqa
            "year": 2020,
            "month": 7,
            "day": 5,
            "product": "2A-DPR",
            "product_category": "radar",
            "product_type": "NRT",
            "start_time": datetime.datetime(2020, 7, 5, 17, 0, 44),
            "end_time": datetime.datetime(2020, 7, 5, 18, 33, 17),
            "version": 7,
        },
        # JAXA
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/1B/GPMCOR_KAR_2007050002_0135_036081_1BS_DAB_07A.h5": {  # noqa
            "year": 2020,
            "month": 7,
            "day": 5,
            "product": "1B-Ka",
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 0, 2, 0),
            "end_time": datetime.datetime(2020, 7, 5, 1, 35, 0),
            "version": 7,
            "granule_id": 36081,
        },
        # JAXA over two days
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/1B/GPMCOR_KUR_2007052310_0043_036096_1BS_DUB_07A.h5": {  # noqa
            "year": 2020,
            "month": 7,
            "day": 5,
            "product": "1B-Ku",
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 23, 10, 0),
            "end_time": datetime.datetime(2020, 7, 6, 0, 43, 0),
            "version": 7,
            "granule_id": 36096,
        },
        # JAXA NRT
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/1B/GPMCOR_KAR_2007050002_0135_036081_1BR_DAB_07A.h5": {  # noqa
            "year": 2020,
            "month": 7,
            "day": 5,
            "product": "1B-Ka",
            "product_category": "radar",
            "product_type": "NRT",
            "start_time": datetime.datetime(2020, 7, 5, 0, 2, 0),
            "end_time": datetime.datetime(2020, 7, 5, 1, 35, 0),
            "version": 7,
        },
        # Include non-ftps folders
        "ftp://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S213824-E231056.036095.V07A.HDF5": {  # noqa
            "year": 2020,
            "month": 7,
            "day": 5,
            "product": "2A-DPR",
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 21, 38, 24),
            "end_time": datetime.datetime(2020, 7, 5, 23, 10, 56),
            "version": 7,
            "granule_id": 36095,
        },
        "ftp://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S231057-E004329.036096.V07A.HDF5": {  # noqa
            "year": 2020,
            "month": 7,
            "day": 5,
            "product": "2A-DPR",
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 23, 10, 57),
            "end_time": datetime.datetime(2020, 7, 6, 0, 43, 29),
            "version": 7,
            "granule_id": 36096,
        },
        "ftp://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S004330-E021602.036097.V07A.HDF5": {  # noqa
            "year": 2020,
            "month": 7,
            "day": 5,
            "product": "2A-DPR",
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 0, 43, 30),
            "end_time": datetime.datetime(2020, 7, 5, 2, 16, 2),
            "version": 7,
            "granule_id": 36097,
        },
        "ftp://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2019/07/05/radar/2A.GPM.DPR.V9-20211125.20190705-S004330-E021602.036097.V07A.HDF5": {  # noqa
            "year": 2019,
            "month": 7,
            "day": 5,
            "product": "2A-DPR",
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2019, 7, 5, 0, 43, 30),
            "end_time": datetime.datetime(2019, 7, 5, 2, 16, 2),
            "version": 7,
            "granule_id": 36097,
        },
        # TODO: Add more products with varying attributes ...
    }


@pytest.fixture
def local_filepaths() -> List[Tuple[str, ...]]:
    """Returns a list of probable local filepath structures as a list"""

    return [
        (
            "data",
            "GPM",
            "RS",
            "V05",
            "PMW",
            "1B-TMI",
            "2014",
            "07",
            "01",
            "1B.TRMM.TMI.Tb2017.20140701-S045751-E063013.094690.V05A.HDF5",
        ),
        (
            "data",
            "GPM",
            "RS",
            "V07",
            "PMW",
            "1B-TMI",
            "2014",
            "07",
            "01",
            "1B.TRMM.TMI.Tb2021.20140701-S063014-E080236.094691.V07A.HDF5",
        ),
        (
            "data",
            "GPM",
            "RS",
            "V07",
            "PMW",
            "1C-ATMS-NPP",
            "2018",
            "07",
            "01",
            "1C.NPP.ATMS.XCAL2019-V.20180701-S075948-E094117.034588.V07A.HDF5",
        ),
        (
            "data",
            "GPM",
            "RS",
            "V07",
            "RADAR",
            "2A-TRMM-SLH",
            "2014",
            "07",
            "01",
            "2A.TRMM.PR.TRMM-SLH.20140701-S080237-E093500.094692.V07A.HDF5",
        ),
        (
            "data",
            "GPM",
            "RS",
            "V07",
            "RADAR",
            "2A-ENV-PR",
            "2014",
            "07",
            "01",
            "2A-ENV.TRMM.PR.V9-20220125.20140701-S063014-E080236.094691.V07A.HDF5",
        ),
        (
            "data",
            "GPM",
            "RS",
            "V07",
            "RADAR",
            "1B-PR",
            "2014",
            "07",
            "01",
            "1B.TRMM.PR.V9-20210630.20140701-S080237-E093500.094692.V07A.HDF5",
        ),
        (
            "data",
            "GPM",
            "RS",
            "V07",
            "RADAR",
            "1B-Ku",
            "2020",
            "10",
            "28",
            "GPMCOR_KUR_2010280754_0927_037875_1BS_DUB_07A.h5",
        ),
        (
            "data",
            "GPM",
            "RS",
            "V07",
            "RADAR",
            "2A-DPR",
            "2022",
            "07",
            "06",
            "2A.GPM.DPR.V9-20211125.20220706-S043937-E061210.047456.V07A.HDF5",
        ),
        (
            "data",
            "GPM",
            "RS",
            "V07",
            "CMB",
            "2B-GPM-CORRA",
            "2016",
            "03",
            "09",
            "2B.GPM.DPRGMI.CORRA2022.20160309-S091322-E104552.011525.V07A.HDF5",
        ),
        (
            "data",
            "GPM",
            "RS",
            "V06",
            "IMERG",
            "IMERG-FR",
            "2020",
            "02",
            "01",
            "3B-HHR.MS.MRG.3IMERG.20200201-S180000-E182959.1080.V06B.HDF5",
        ),
    ]


@pytest.fixture
def local_filepaths_unix(local_filepaths) -> List[str]:
    """Return the local filepath list as unix paths"""

    return [pxp.join(*path) for path in local_filepaths]


@pytest.fixture
def local_filepaths_windows(local_filepaths) -> List[str]:
    """Return the local filepath list as windows paths"""

    return [ntp.join(*path) for path in local_filepaths]


@pytest.fixture
def set_is_orbit_to_true(
    mocker: MockerFixture,
) -> None:
    mocker.patch("gpm.checks.is_orbit", return_value=True)
    mocker.patch("gpm.checks.is_grid", return_value=False)
    mocker.patch("gpm.utils.checks.is_orbit", return_value=True)
    mocker.patch("gpm.utils.checks.is_grid", return_value=False)


@pytest.fixture
def set_is_grid_to_true(
    mocker: MockerFixture,
) -> None:
    mocker.patch("gpm.checks.is_grid", return_value=True)
    mocker.patch("gpm.checks.is_orbit", return_value=False)
    mocker.patch("gpm.utils.checks.is_grid", return_value=True)
    mocker.patch("gpm.utils.checks.is_orbit", return_value=False)


ExtentDictionary = Dict[str, Tuple[float, float, float, float]]


@pytest.fixture
def country_extent_dictionary() -> ExtentDictionary:
    return geospatial.read_countries_extent_dictionary()


@pytest.fixture
def continent_extent_dictionary() -> ExtentDictionary:
    return geospatial.read_continents_extent_dictionary()


@pytest.fixture
def prevent_pyplot_show(
    mocker: MockerFixture,
) -> None:
    """Prevent the show method of the pyplot module to be called"""

    mocker.patch("matplotlib.pyplot.show")


#### Orbit Data Array


@pytest.fixture(scope="function")
def orbit_dataarray() -> xr.DataArray:
    """Create orbit data array near 0 longitude and latitude"""

    return get_orbit_dataarray(
        start_lon=0,
        start_lat=0,
        end_lon=20,
        end_lat=15,
        width=1e6,
        n_along_track=20,
        n_cross_track=5,
    )


@pytest.fixture(scope="function")
def orbit_antimeridian_dataarray() -> xr.DataArray:
    """Create orbit data array going over the antimeridian"""

    return get_orbit_dataarray(
        start_lon=160,
        start_lat=0,
        end_lon=-170,
        end_lat=15,
        width=1e6,
        n_along_track=20,
        n_cross_track=5,
    )


@pytest.fixture(scope="function")
def orbit_pole_dataarray() -> xr.DataArray:
    """Create orbit data array going over the south pole"""

    return get_orbit_dataarray(
        start_lon=-30,
        start_lat=-70,
        end_lon=150,
        end_lat=-75,
        width=1e6,
        n_along_track=20,
        n_cross_track=5,
    )


@pytest.fixture(scope="function")
def orbit_spatial_3d_dataarray(orbit_dataarray: xr.DataArray) -> xr.DataArray:
    """Return a 3D orbit data array"""

    # Add a vertical dimension with shape larger than 1 to prevent squeezing
    return orbit_dataarray.expand_dims(dim={"height": 2})


@pytest.fixture
def orbit_transect_dataarray(orbit_dataarray: xr.DataArray) -> xr.DataArray:
    """Return a transect orbit data array"""

    orbit_dataarray = orbit_dataarray.expand_dims(dim={"height": 2})
    return orbit_dataarray.isel(along_track=0)


#### Orbit Data Array with NaN values


@pytest.fixture(scope="function")
def orbit_data_nan_cross_track_dataarray(orbit_dataarray) -> xr.DataArray:
    """Create orbit data array near 0 longitude and latitude with NaN data in outer cross-track."""

    padding_size = 2

    data = orbit_dataarray.data
    data[0:padding_size, :] = float("nan")
    data[-padding_size:, :] = float("nan")

    return orbit_dataarray


@pytest.fixture(scope="function")
def orbit_data_nan_along_track_dataarray(orbit_dataarray) -> xr.DataArray:
    """Create orbit data array near 0 longitude and latitude with NaN data at along-track edges."""

    padding_size = 2

    data = orbit_dataarray.data
    data[:, 0:padding_size] = float("nan")
    data[:, -padding_size:] = float("nan")

    return orbit_dataarray


#### Orbit Data Array with NaN coordinates


@pytest.fixture(scope="function")
def orbit_nan_slice_along_track_dataarray(orbit_dataarray) -> xr.DataArray:
    """Create orbit data array with missing coordinates over some along-track indices"""

    along_track_index = 5
    missing_size = 2

    lon = orbit_dataarray["lon"]
    lon[:, along_track_index : along_track_index + missing_size] = float("nan")

    return orbit_dataarray


@pytest.fixture(scope="function")
def orbit_nan_outer_cross_track_dataarray(orbit_dataarray) -> xr.DataArray:
    """Create orbit data array with all NaN coordinates in outer cross-track indices"""

    padding_size = 1

    lon = orbit_dataarray["lon"]
    lon[0:padding_size, :] = float("nan")
    lon[-padding_size:, :] = float("nan")

    lat = orbit_dataarray["lat"]
    lat[0:padding_size, :] = float("nan")
    lat[-padding_size:, :] = float("nan")

    return orbit_dataarray


@pytest.fixture(scope="function")
def orbit_nan_inner_cross_track_dataarray(orbit_dataarray) -> xr.DataArray:
    """Create orbit data array with all NaN coordinates in some inner cross-track indices"""
    lon = orbit_dataarray["lon"]
    lon[1, :] = float("nan")
    lon[-2, :] = float("nan")

    lat = orbit_dataarray["lat"]
    lat[1, :] = float("nan")
    lat[-2:, :] = float("nan")

    return orbit_dataarray


#### Grid Data Array


@pytest.fixture(scope="function")
def grid_dataarray() -> xr.DataArray:
    """Create grid data array near 0 longitude and latitude"""

    return get_grid_dataarray(
        start_lon=-5,
        start_lat=-5,
        end_lon=20,
        end_lat=15,
        n_lon=20,
        n_lat=15,
    )


@pytest.fixture(scope="function")
def grid_nan_lon_dataarray(grid_dataarray) -> xr.DataArray:
    """Create grid data array near 0 longitude and latitude with some NaN longitudes"""

    lon_index = 5
    missing_size = 2

    lon = grid_dataarray["lon"].data.copy()
    lon[lon_index : lon_index + missing_size] = float("nan")
    grid_dataarray["lon"] = lon

    return grid_dataarray


@pytest.fixture(scope="function")
def grid_spatial_3d_dataarray(grid_dataarray: xr.DataArray) -> xr.DataArray:
    """Return a 3D grid data array"""

    # Add a vertical dimension with shape larger than 1 to prevent squeezing
    return grid_dataarray.expand_dims(dim={"height": 2})


@pytest.fixture
def grid_transect_dataarray(grid_dataarray: xr.DataArray) -> xr.DataArray:
    """Return a transect grid data array"""

    grid_dataarray = grid_dataarray.expand_dims(dim={"height": 2})
    return grid_dataarray.isel(lat=0)


#### Datasets


@pytest.fixture
def dataset_collection(
    orbit_dataarray: xr.DataArray,
    grid_dataarray: xr.DataArray,
    orbit_spatial_3d_dataarray: xr.DataArray,
    grid_spatial_3d_dataarray: xr.DataArray,
    orbit_transect_dataarray: xr.DataArray,
    grid_transect_dataarray: xr.DataArray,
) -> xr.Dataset:
    """Return a dataset with a variety of data arrays"""

    da_frequency = xr.DataArray(np.zeros((0, 0)), dims=["other", "radar_frequency"])

    return xr.Dataset(
        {
            "variable_0": orbit_dataarray,
            "variable_1": grid_dataarray,
            "variable_2": orbit_spatial_3d_dataarray,
            "variable_3": grid_spatial_3d_dataarray,
            "variable_4": orbit_transect_dataarray,
            "variable_5": grid_transect_dataarray,
            "variable_6": da_frequency,
            "variable_7": xr.DataArray(),
        }
    )
