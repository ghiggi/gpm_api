import pytest
import datetime
from typing import Any, List, Dict, Tuple, Iterable
from gpm_api.io.products import get_info_dict, available_products
from gpm_api.utils import geospatial
import posixpath as pxp
import ntpath as ntp
import gpm_api.configs
import os
from pytest_mock import MockerFixture
from unittest.mock import patch
import xarray as xr
from gpm_api.tests.utils.fake_datasets import get_orbit_dataarray, get_grid_dataarray


@pytest.fixture(scope="session", autouse=True)
def mock_configuration() -> Iterable[Dict[str, str]]:
    """Patch the user configuration for entire session

    Doing this will retrieve the configuration from pytest memory and not
    alter the local configuration in ~/.config_gpm_api.yml
    """

    mocked_configuration = {
        "username_pps": "testuser",
        "password_pps": "testuser",
        "username_earthdata": "testuser",
        "password_earthdata": "testuser",
        "gpm_base_dir": os.path.join(
            os.getcwd(),
            "gpm_api",
            "tests",
            "resources",
        ),
    }

    with patch.object(
        gpm_api.configs,
        "read_gpm_api_configs",
        return_value=mocked_configuration,
    ):
        yield mocked_configuration


@pytest.fixture
def product_types() -> List[str]:
    """Return a list of all product types from the info dict"""
    product_types = []
    for product, info_dict in get_info_dict().items():
        product_types += info_dict["product_types"]

    product_types = list(set(product_types))  # Dedup list

    return product_types


@pytest.fixture
def product_categories() -> List[str]:
    """Return a list of product categories from the info dict"""

    return list(set([info_dict["product_category"] for info_dict in get_info_dict().values()]))


@pytest.fixture
def product_levels() -> List[str]:
    """Return a list of product levels from the info dict"""
    from gpm_api.io.products import get_available_product_levels

    return get_available_product_levels(full=False)  #  ["1A", "1B", "1C", "2A", "2B", "3B"]


@pytest.fixture
def versions() -> List[int]:
    """Return a list of versions"""
    from gpm_api.io.products import get_available_versions

    return get_available_versions()


@pytest.fixture
def products() -> List[str]:
    """Return a list of all products regardless of type"""
    from gpm_api.io.products import get_available_products

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
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S170044-E183317.036092.V07A.HDF5": {
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
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S183318-E200550.036093.V07A.HDF5": {
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
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S200551-E213823.036094.V07A.HDF5": {
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
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S231057-E004329.036096.V07A.HDF5": {
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
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S170044-E183317.V07A.HDF5": {
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
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/1B/GPMCOR_KAR_2007050002_0135_036081_1BS_DAB_07A.h5": {
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
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/1B/GPMCOR_KUR_2007052310_0043_036096_1BS_DUB_07A.h5": {
            "year": 2020,
            "month": 7,
            "day": 5,
            "product": "1B-Ka",
            "product_category": "radar",
            "product_type": "RS",
            "start_time": datetime.datetime(2020, 7, 5, 23, 10, 0),
            "end_time": datetime.datetime(2020, 7, 6, 0, 43, 0),
            "version": 7,
            "granule_id": 36096,
        },
        # JAXA NRT
        "ftps://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/1B/GPMCOR_KAR_2007050002_0135_036081_1BR_DAB_07A.h5": {
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
        "ftp://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S213824-E231056.036095.V07A.HDF5": {
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
        "ftp://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S231057-E004329.036096.V07A.HDF5": {
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
        "ftp://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2020/07/05/radar/2A.GPM.DPR.V9-20211125.20200705-S004330-E021602.036097.V07A.HDF5": {
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
        "ftp://arthurhouftps.pps.eosdis.nasa.gov/gpmdata/2019/07/05/radar/2A.GPM.DPR.V9-20211125.20190705-S004330-E021602.036097.V07A.HDF5": {
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
    mocker.patch("gpm_api.checks.is_orbit", return_value=True)
    mocker.patch("gpm_api.checks.is_grid", return_value=False)
    mocker.patch("gpm_api.utils.checks.is_orbit", return_value=True)
    mocker.patch("gpm_api.utils.checks.is_grid", return_value=False)


@pytest.fixture
def set_is_grid_to_true(
    mocker: MockerFixture,
) -> None:
    mocker.patch("gpm_api.checks.is_grid", return_value=True)
    mocker.patch("gpm_api.checks.is_orbit", return_value=False)
    mocker.patch("gpm_api.utils.checks.is_grid", return_value=True)
    mocker.patch("gpm_api.utils.checks.is_orbit", return_value=False)


ExtentDictionary = Dict[str, Tuple[float, float, float, float]]


@pytest.fixture
def country_extent_dictionary() -> ExtentDictionary:
    return geospatial._get_country_extent_dictionary()


@pytest.fixture
def continent_extent_dictionary() -> ExtentDictionary:
    return geospatial._get_continent_extent_dictionary()


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
def orbit_nan_cross_track_dataarray(orbit_dataarray) -> xr.DataArray:
    """Create orbit data array near 0 longitude and latitude with NaN cross-track edges"""

    padding_size = 2

    data = orbit_dataarray.data
    data[0:padding_size, :] = float("nan")
    data[-padding_size:, :] = float("nan")

    return orbit_dataarray


@pytest.fixture(scope="function")
def orbit_nan_along_track_dataarray(orbit_dataarray) -> xr.DataArray:
    """Create orbit data array near 0 longitude and latitude with NaN along-track edges"""

    padding_size = 2

    data = orbit_dataarray.data
    data[:, 0:padding_size] = float("nan")
    data[:, -padding_size:] = float("nan")

    return orbit_dataarray


@pytest.fixture(scope="function")
def orbit_nan_lon_dataarray(orbit_dataarray) -> xr.DataArray:
    """Create orbit data array near 0 longitude and latitude with some NaN longitudes (along-track)"""

    along_track_index = 5
    padding_size = 2

    lon = orbit_dataarray["lon"]
    lon[:, along_track_index : along_track_index + padding_size] = float("nan")

    return orbit_dataarray


@pytest.fixture(scope="function")
def orbit_nan_lat_dataarray(orbit_dataarray) -> xr.DataArray:
    """Create orbit data array near 0 longitude and latitude with some NaN latitudes (along-track)"""

    along_track_index = 10
    padding_size = 2

    lat = orbit_dataarray["lat"]
    lat[:, along_track_index : along_track_index + padding_size] = float("nan")

    return orbit_dataarray


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
def grid_antimeridian_dataarray() -> xr.DataArray:
    """Create grid data array going over the antimeridian"""

    return get_grid_dataarray(
        start_lon=160,
        start_lat=-5,
        end_lon=-170,
        end_lat=15,
        n_lon=20,
        n_lat=15,
    )
