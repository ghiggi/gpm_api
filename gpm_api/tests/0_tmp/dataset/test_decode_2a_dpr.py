#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:33:12 2023

@author: ghiggi
"""
import gpm_api
import datetime
import numpy as np
from gpm_api.dataset.decoding.decode_2a_radar import (
    decode_landSurfaceType,
    decode_phase,
    decode_phaseNearSurface,
    decode_flagPrecip,
    decode_typePrecip,
    decode_qualityTypePrecip,
    decode_flagShallowRain,
    decode_flagHeavyIcePrecip,
    decode_flagAnvil,
    decode_flagBB,
    decode_flagSurfaceSnowfall,
    decode_flagGraupelHail,
    decode_flagHail,
    decode_product,
)

#### Define analysis time period
start_time = datetime.datetime.strptime("2016-03-09 10:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2016-03-09 11:00:00", "%Y-%m-%d %H:%M:%S")


product = "2A-DPR"
version = 7
product_type = "RS"

ds = gpm_api.open_dataset(
    product=product,
    start_time=start_time,
    end_time=end_time,
    # Optional
    version=version,
    product_type=product_type,
    chunks={},
    decode_cf=True,  # does the decoding
    prefix_group=False,
)

ds["flagPrecip"].attrs  # see presence of flag_values, flag_meanings

ds = gpm_api.open_dataset(
    product=product,
    start_time=start_time,
    end_time=end_time,
    # Optional
    version=version,
    product_type=product_type,
    chunks={},
    decode_cf=False,  # TODO: test below with False !
    prefix_group=False,
)

ds = decode_product(ds)  # do not decode if already decoded ;)


### TODO: DEBUG
# xr_obj = ds["flagShallowRain"]
# remapping_dict = {-1111: 0, 0: 1, 10: 2, 11: 3, 20: 4, 21: 5}
# xr_obj = xr_obj.where(xr_obj > -1112, 0)
# xr_obj.data = remap_numeric_array(xr_obj.data, remapping_dict) TODO


###---------------------------------------------------------------------------.
ds["typePrecip"].attrs
np.unique(ds["typePrecip"].data.compute(), return_counts=True)
ds["typePrecip"] = decode_typePrecip(ds, method="major_rain_type")
ds["typePrecip"].attrs
np.unique(ds["typePrecip"].data.compute(), return_counts=True)


###---------------------------------------------------------------------------.
ds["flagPrecip"].attrs
np.unique(ds["flagPrecip"].data.compute(), return_counts=True)
ds["flagPrecip"] = decode_flagPrecip(ds)
ds["flagPrecip"].attrs
np.unique(ds["flagPrecip"].data.compute(), return_counts=True)

ds["qualityTypePrecip"].attrs
np.unique(ds["qualityTypePrecip"].data.compute(), return_counts=True)
ds["qualityTypePrecip"] = decode_qualityTypePrecip(ds)
ds["qualityTypePrecip"].attrs
np.unique(ds["qualityTypePrecip"].data.compute(), return_counts=True)

ds["flagShallowRain"].attrs
np.unique(ds["flagShallowRain"].data.compute(), return_counts=True)
ds["flagShallowRain"] = decode_flagShallowRain(ds)
ds["flagShallowRain"].attrs
np.unique(ds["flagShallowRain"].data.compute(), return_counts=True)

ds["flagHeavyIcePrecip"].attrs
np.unique(ds["flagHeavyIcePrecip"].data.compute(), return_counts=True)
ds["flagHeavyIcePrecip"] = decode_flagHeavyIcePrecip(ds)
ds["flagHeavyIcePrecip"].attrs
np.unique(ds["flagHeavyIcePrecip"].data.compute(), return_counts=True)

ds["flagAnvil"].attrs
np.unique(ds["flagAnvil"].data.compute(), return_counts=True)
ds["flagAnvil"] = decode_flagAnvil(ds)
ds["flagAnvil"].attrs
np.unique(ds["flagAnvil"].data.compute(), return_counts=True)

ds["flagBB"].attrs
np.unique(ds["flagBB"].data.compute(), return_counts=True)
ds["flagBB"] = decode_flagBB(ds)
ds["flagBB"].attrs
np.unique(ds["flagBB"].data.compute(), return_counts=True)

ds["landSurfaceType"].attrs
np.unique(ds["landSurfaceType"].data.compute(), return_counts=True)
ds["landSurfaceType"] = decode_landSurfaceType(ds)
ds["landSurfaceType"].attrs
np.unique(ds["landSurfaceType"].data.compute(), return_counts=True)

ds["phase"].attrs
np.unique(ds["phase"].data.compute(), return_counts=True)
ds["phase"] = decode_phase(ds)
ds["phase"].attrs
np.unique(ds["phase"].data.compute(), return_counts=True)

ds["phaseNearSurface"].attrs
np.unique(ds["phaseNearSurface"].data.compute(), return_counts=True)
ds["phaseNearSurface"] = decode_phaseNearSurface(ds)
ds["phaseNearSurface"].attrs
np.unique(ds["phaseNearSurface"].data.compute(), return_counts=True)

ds["flagSurfaceSnowfall"].attrs
np.unique(ds["flagSurfaceSnowfall"].data.compute(), return_counts=True)
ds["flagSurfaceSnowfall"] = decode_flagSurfaceSnowfall(ds)
ds["flagSurfaceSnowfall"].attrs
np.unique(ds["flagSurfaceSnowfall"].data.compute(), return_counts=True)

ds["flagHail"].attrs
np.unique(ds["flagHail"].data.compute(), return_counts=True)
ds["flagHail"] = decode_flagHail(ds)
ds["flagHail"].attrs
np.unique(ds["flagHail"].data.compute(), return_counts=True)

ds["flagGraupelHail"].attrs
np.unique(ds["flagGraupelHail"].data.compute(), return_counts=True)
ds["flagGraupelHail"] = decode_flagGraupelHail(ds)
ds["flagGraupelHail"].attrs
np.unique(ds["flagGraupelHail"].data.compute(), return_counts=True)
