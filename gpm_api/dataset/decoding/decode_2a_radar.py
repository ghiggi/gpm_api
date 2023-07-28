#!/usr/bin/env python3
"""
Created on Fri Jul 28 11:33:59 2023

@author: ghiggi
"""
from gpm_api.dataset.decoding.utils import get_data_array, remap_numeric_array


def decode_landSurfaceType(xr_obj, method="landSurfaceType"):
    """Decode the 2A-<RADAR> variable landSurfaceType."""
    xr_obj = get_data_array(xr_obj, variable="landSurfaceType")
    xr_obj = xr_obj.where(xr_obj >= 0)  # < 0 set to np.nan
    xr_obj = xr_obj / 100
    xr_obj = xr_obj.astype(int)
    xr_obj.attrs["flag_values"] = [0, 1, 2, 3]
    xr_obj.attrs["flag_meanings"] = ["Ocean", "Land", "Coast", "Inland Water"]
    xr_obj.attrs["description"] = "Land Surface type"
    return xr_obj


def decode_phase(xr_obj):
    """Decode the 2A-<RADAR> variable phase."""
    xr_obj = get_data_array(xr_obj, variable="phase")
    xr_obj = xr_obj / 100
    xr_obj = xr_obj.astype(int)
    xr_obj = xr_obj.where(xr_obj >= 0)  # < 0 set to np.nan
    xr_obj.attrs["flag_values"] = [0, 1, 2]
    xr_obj.attrs["flag_meanings"] = ["solid", "mixed_phase", "liquid"]
    xr_obj.attrs["description"] = "Precipitation phase state"
    return xr_obj


def decode_phaseNearSurface(xr_obj):
    """Decode the 2A-<RADAR> variable phaseNearSurface."""
    xr_obj = get_data_array(xr_obj, variable="phaseNearSurface")
    xr_obj = xr_obj / 100
    xr_obj = xr_obj.astype(int)
    xr_obj = xr_obj.where(xr_obj >= 0)  # < 0 set to np.nan
    xr_obj.attrs["flag_values"] = [0, 1, 2]
    xr_obj.attrs["flag_meanings"] = ["solid", "mixed_phase", "liquid"]
    xr_obj.attrs["description"] = "Precipitation phase state near the surface"
    return xr_obj


def decode_flagPrecip(xr_obj):
    """Decode the 2A-<RADAR> variable flagPrecip."""
    xr_obj = get_data_array(xr_obj, variable="flagPrecip")
    if xr_obj.attrs["gpm_api_product"] == "2A-DPR":
        xr_obj.attrs["flag_values"] = [0, 1, 10, 11]
        # TODO: 2, 10, 12, 20, 21, 22 values also present
        xr_obj.attrs["flag_meanings"] = [
            "not detected by both Ku and Ka",
            "detected by Ka only",
            "detected by Ku only",
            "detected by both Ku and Ka",
        ]
    else:
        xr_obj.attrs["flag_values"] = [0, 1]
        xr_obj.attrs["flag_meanings"] = ["not detected", "detected"]
    xr_obj.attrs["description"] = "Flag for precipitation detection"
    return xr_obj


def decode_typePrecip(xr_obj, method="major_rain_type"):
    """Decode the 2A-<RADAR> variable typePrecip."""
    xr_obj = get_data_array(xr_obj, variable="typePrecip")
    available_methods = ["major_rain_type"]
    if method not in available_methods:
        raise NotImplementedError(f"Implemented methods are {available_methods}")
    # Decode typePrecip
    # if method == "major_rain_type"
    xr_obj = xr_obj / 10000000
    xr_obj = xr_obj.astype(int)
    xr_obj.attrs["flag_values"] = [0, 1, 2, 3]
    xr_obj.attrs["flag_meanings"] = ["no rain", "stratiform", "convective", "other"]
    xr_obj.attrs["description"] = "Precipitation type"
    return xr_obj


def decode_qualityTypePrecip(xr_obj):
    """Decode the 2A-<RADAR> variable qualityTypePrecip."""
    xr_obj = get_data_array(xr_obj, variable="qualityTypePrecip")
    xr_obj = xr_obj.where(xr_obj > 0, 0)
    xr_obj.attrs["flag_values"] = [0, 1]
    xr_obj.attrs["flag_meanings"] = ["no rain", "good"]
    xr_obj.attrs["description"] = "Quality of the precipitation type"
    return xr_obj


def decode_flagShallowRain(xr_obj):
    """Decode the 2A-<RADAR> variable flagShallowRain."""
    xr_obj = get_data_array(xr_obj, variable="flagShallowRain")
    xr_obj = xr_obj.where(xr_obj > -1112, 0)
    remapping_dict = {-1111: 0, 0: 1, 10: 2, 11: 3, 20: 4, 21: 5}
    xr_obj.data = remap_numeric_array(xr_obj.data, remapping_dict)  # TODO
    xr_obj.attrs["flag_values"] = list(remapping_dict.values())
    xr_obj.attrs["flag_meanings"] = [
        "no rain",
        "no shallow rain",
        "Shallow isolated (maybe)",
        "Shallow isolated (certain)",
        "Shallow non-isolated (maybe)",
        "Shallow non-isolated (certain)",
    ]
    xr_obj.attrs["description"] = "Type of shallow rain"
    return xr_obj


def decode_flagHeavyIcePrecip(xr_obj):
    """Decode the 2A-<RADAR> variable flagHeavyIcePrecip."""
    xr_obj = get_data_array(xr_obj, variable="flagHeavyIcePrecip")
    xr_obj = xr_obj.where(xr_obj >= 1)  # make 0 nan
    xr_obj.attrs["flag_values"] = [4, 8, 12, 16, 24, 32, 40]
    xr_obj.attrs["flag_meanings"] = [""] * 6  # TODO
    xr_obj.attrs[
        "description"
    ] = """Flag for detection of strong or severe precipitation accompanied
    by solid ice hydrometeors above the -10 degree C isotherm"""
    return xr_obj


def decode_flagAnvil(xr_obj):
    """Decode the 2A-<RADAR> variable flagAnvil."""
    xr_obj = get_data_array(xr_obj, variable="flagAnvil")
    xr_obj = xr_obj.where(xr_obj >= 1)  # make 0 nan
    xr_obj.attrs["flag_values"] = [0, 1, 2]  # TODO: 2 is unknown
    xr_obj.attrs["flag_meanings"] = ["not detected", "detected", "unknown"]
    xr_obj.attrs["description"] = "Flago for anvil detection by the Ku-band radar"
    return xr_obj


def decode_flagBB(xr_obj):
    """Decode the 2A-<RADAR> variable flagBB."""
    xr_obj = get_data_array(xr_obj, variable="flagBB")
    xr_obj = xr_obj.where(xr_obj >= 0)  # make -1111 (no rain) nan
    if xr_obj.attrs["gpm_api_product"] == "2A-DPR":
        xr_obj.attrs["flag_values"] = [0, 1, 2, 3]
        xr_obj.attrs["flag_meanings"] = [
            "not detected",
            "detected by Ku and DFRm",
            "detected by Ku only",
            "detected by DFRm only",
        ]
    else:
        xr_obj.attrs["flag_values"] = [0, 1]
        xr_obj.attrs["flag_meanings"] = ["not detected", "detected"]

    xr_obj.attrs["description"] = "Flag for bright band detection"
    return xr_obj


def decode_flagSurfaceSnowfall(xr_obj):
    """Decode the 2A-<RADAR> variable flagSurfaceSnowfall."""
    xr_obj = get_data_array(xr_obj, variable="flagSurfaceSnowfall")
    xr_obj.attrs["flag_values"] = [0, 1]
    xr_obj.attrs["flag_meanings"] = ["no snow-cover", "snow-cover"]
    xr_obj.attrs["description"] = "Flag for snow-cover"
    return xr_obj


def decode_flagHail(xr_obj):
    """Decode the 2A-<RADAR> variable flagHail."""
    xr_obj = get_data_array(xr_obj, variable="flagHail")
    xr_obj.attrs["flag_values"] = [0, 1]
    xr_obj.attrs["flag_meanings"] = ["not detected", "detected"]
    xr_obj.attrs["description"] = "Flag for hail detection"
    return xr_obj


def decode_flagGraupelHail(xr_obj):
    """Decode the 2A-<RADAR> variable flagGraupelHail."""
    xr_obj = get_data_array(xr_obj, variable="flagGraupelHail")
    xr_obj.attrs["flag_values"] = [0, 1]
    xr_obj.attrs["flag_meanings"] = ["not detected", "detected"]
    xr_obj.attrs["description"] = "Flag for graupel hail detection "
    return xr_obj


def _get_decoding_function(variable):
    function_name = f"decode_{variable}"
    decoding_function = globals().get(function_name)
    if decoding_function is None or not callable(decoding_function):
        raise ValueError(f"No decoding function found for variable '{variable}'")
    return decoding_function


def decode_product(ds):
    """Decode 2A-<RADAR> products."""
    # Define variables to decode with _decode_<variable> functions
    variables = [
        "flagBB",
        # "flagShallowRain",
        "flagAnvil",
        "flagHeavyIcePrecip",
        "flagHail",
        "flagGraupelHail",
        "flagSurfaceSnowfall",
        # "TypePrecip",
        "qualityTypePrecip",
        "flagPrecip",
        "phaseNearSurface",
        "phase",
        "landSurfaceType",
    ]
    # Decode such variables if present in the xarray object
    for variable in variables:
        if variable in ds and not ds[variable].attrs.get("gpm_api_decoded", False):
            ds[variable] = _get_decoding_function(variable)(ds[variable])
            ds[variable].attrs["gpm_api_decoded"] = True

    # Preprocess other variables
    # --> Split 3D field in 2D fields
    if "precipWaterIntegrated" in ds and not ds[variable].attrs.get("gpm_api_decoded", False):
        print(ds["precipWaterIntegrated"])
        ds["precipWaterIntegrated_Liquid"] = ds["precipWaterIntegrated"].isel({"LS": 0})
        ds["precipWaterIntegrated_Solid"] = ds["precipWaterIntegrated"].isel({"LS": 1})
        ds = ds.drop_vars(names="precipWaterIntegrated")
        ds["precipWaterIntegrated"] = (
            ds["precipWaterIntegrated_Liquid"] + ds["precipWaterIntegrated_Solid"]
        )
        ds["precipWaterIntegrated_Liquid"].attrs["gpm_api_decoded"] = True
        ds["precipWaterIntegrated_Solid"].attrs["gpm_api_decoded"] = True
        ds["precipWaterIntegrated"].attrs["gpm_api_decoded"] = True
    return ds
