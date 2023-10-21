import pytest

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
