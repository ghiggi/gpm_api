# Test created to apply to gpm_api.io.checks.check_bbox() but function is now
# deprecated, potential to use for gpm_api.utils.geospatial.extent calls


# def test_check_bbox() -> None:
#     ''' Test validity of a given bbox

#     bbox format: [lon_0, lon_1, lat_0, lat_1]
#     '''

#     # Test within range of latitude and longitude
#     res = checks.check_bbox([0, 0, 0, 0])
#     assert res == [0, 0, 0, 0], (
#         "Function returned {res}, expected [0, 0, 0, 0]"
#     )

#     # Test a series of bboxes testing the outside range of lat and lon
#     # but having at least valid ranges in lat lon
#     for bbox in [
#         [-180, 180, -91, 90],
#         [-180, 180, -90, 91],
#         [-181, 180, -90, 90],
#         [-180, 181, -90, 90],
#         # Now with signs flipped
#         [180, -180, 90, -91],
#         [180, -180, 91, -90],
#         [181, -180, 90, -90],
#         [180, -181, 90, -90],
#     ]:
#         with pytest.raises(ValueError):
#             print(
#                 f"Testing bbox within bounds of lat0, lat1, lon0, lon1: {bbox}"
#             )
#             checks.check_bbox(bbox)

#     # Test a bbox that isn't a list
#     with pytest.raises(ValueError):
#         checks.check_bbox(123)

#     # Test a bbox that isn't a list of length 4
#     with pytest.raises(ValueError):
#         checks.check_bbox([0, 0, 0])


import pytest
from typing import Dict, Tuple

from gpm_api.utils import geospatial

ExtentDictionary = Dict[str, Tuple[float, float, float, float]]


# @pytest.fixture
# def mock_country_extent_dictionary(
#     mocker: MockFixture,
# ) -> CountryExtentDictionary:
#     """Mock country extent dictionary"""

#     country_extent_dictionary = {
#         "Afghanistan": (60.5284298033, 75.1580277851, 29.318572496, 38.4862816432),
#         "Albania": (19.3044861183, 21.0200403175, 39.624997667, 42.6882473822),
#         "Algeria": (-8.68439978681, 11.9995056495, 19.0573642034, 37.1183806422),
#     }

#     # Mock gpm_api.utils.geospatial._get_country_extent_dictionary
#     mocker.patch(
#         "gpm_api.utils.geospatial._get_country_extent_dictionary",
#         return_value=country_extent_dictionary,
#     )

#     return country_extent_dictionary


def test_get_country_extent(
    country_extent_dictionary: ExtentDictionary,
) -> None:
    """Test get_country_extent"""

    # Test valid country
    country = "Afghanistan"
    e = country_extent_dictionary[country]
    expected_extent = (e[0] - 0.2, e[1] + 0.2, e[2] - 0.2, e[3] + 0.2)
    returned_extent = geospatial.get_country_extent(country)
    assert returned_extent == expected_extent

    # Test invalid country
    country = "Invalid"
    with pytest.raises(ValueError):
        geospatial.get_country_extent(country)

    # Test typo in country name
    country = "Afganistan"
    with pytest.raises(ValueError) as exception_info:
        geospatial.get_country_extent(country)

    assert "Afghanistan" in str(exception_info.value.args[0])

    # Test invalid country type
    country = 123
    with pytest.raises(TypeError):
        geospatial.get_country_extent(country)


def test_get_continent_extent(
    continent_extent_dictionary: ExtentDictionary,
) -> None:
    """Test get_continent_extent"""

    # Test valid continent
    continent = "Africa"
    expected_extent = continent_extent_dictionary[continent]
    returned_extent = geospatial.get_continent_extent(continent)
    assert returned_extent == expected_extent

    # Test invalid continent
    continent = "Invalid"
    with pytest.raises(ValueError):
        geospatial.get_continent_extent(continent)

    # Test typo in continent name
    continent = "Arica"
    with pytest.raises(ValueError) as exception_info:
        geospatial.get_continent_extent(continent)

    assert "Africa" in str(exception_info.value.args[0])

    # Test invalid continent type
    continent = 123
    with pytest.raises(TypeError):
        geospatial.get_continent_extent(continent)
