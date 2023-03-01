#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:23:49 2023

@author: ghiggi
"""
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import dask
import gpm_api
import datetime
import pandas as pd
import xarray as xr
from dask.distributed import get_client
from dateutil.relativedelta import relativedelta
from gpm_api.utils.archive import print_elapsed_time
import matplotlib.pyplot as plt 

### GPM-GEO ####
from gpm_geo.gpm import get_gpm_extended_swath_coords
from gpm_geo.checks import check_date, check_satellite
from gpm_geo.production.io import define_dataset_filepath
from gpm_geo.production.encoding import set_coords_encoding
from gpm_geo.production.attrs import set_coords_attrs, set_grid_global_attrs
from gpm_geo.production.logger import create_logger, log_error, log_info 
from gpm_geo.production.utils import clean_memory
from pyresample_dev.spherical import SPolygon
from pyresample_dev.utils_swath import * # temporary hack

from gpm_geo.production.grid import clean_memory, get_valid_grid_slices

# import warnings
# warnings.filterwarnings("ignore")

## DEBUG
gpm_base_dir = "/ltenas8/data/GPM"
geo_base_dir = "/ltenas8/data/GPM_GEO"
gpm_geo_base_dir = "/ltenas8/data/GPM_GEO"
satellite = "GOES16" # "GOES17"
 
date = datetime.datetime(2020, 8, 19)
print(date)

###----------------------------------------------------------------------.
#### Checks inputs 
date = check_date(date)
satellite = check_satellite(satellite)

####----------------------------------------------------------------------.
#### Define GEO AOI 
AOI_MAX_RESOLUTION = 2   # KM 
aoi_fname = f"AOI_MAX_{AOI_MAX_RESOLUTION}km_resolution.shp"
aoi_fpath = os.path.join(gpm_geo_base_dir, satellite, "AOI", aoi_fname)
area_to_cover = SPolygon.from_file(aoi_fpath)
area_to_cover = area_to_cover.subsample(n_vertices=40) 
area_to_cover.plot()

####----------------------------------------------------------------------.
#### Define start_time and end_time bounds to generate daily GPM-GEO GRIDS
date_time = datetime.datetime(date.year, date.month, date.day)
start_time = date_time - datetime.timedelta(hours=3)
end_time = date_time + datetime.timedelta(days=1, hours=3)

####----------------------------------------------------------------------.
#### Load GPM Swath
# - Specify product and 
product_type = "RS"
product = "2A-DPR"
variables = ["precipRateNearSurface",
             "dataQuality", 
             "SCorientation"]

# - Specify version and scan_mode
version = 7      
scan_mode = "FS"

# - Load xr.Dataset 
ds_gpm = gpm_api.open_dataset(
    base_dir=gpm_base_dir,
    product=product,
    scan_mode=scan_mode,   
    product_type=product_type, 
    version=version,
    variables=variables,
    start_time=start_time,
    end_time=end_time,
    chunks="auto", # otherwise concatenating datasets is very slow !
    decode_cf=False, 
    prefix_group=False,
    verbose=False, 
)

ds_gpm = ds_gpm.compute()
ds_gpm.close()
        
# Check GPM daily orbit data quality 
# - Check valid geolocation and not missing granules !
gpm_api.check_valid_geolocation(ds_gpm) # TODO: ADAPT verbose=True)
 
# Generate GRID only over non-problematic swath portions 
list_valid_slices = get_valid_grid_slices(ds_gpm)