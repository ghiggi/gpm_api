#!/usr/bin/env python3
"""
Created on Wed Jul 19 15:38:07 2023

@author: ghiggi
"""
import datetime
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ximage  # noqa

import gpm_api
from gpm_api.visualization import plot_labels

####--------------------------------------------------------------------------.
#### Define matplotlib settings
matplotlib.rcParams["axes.facecolor"] = [0.9, 0.9, 0.9]
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["axes.titlesize"] = 11
matplotlib.rcParams["xtick.labelsize"] = 10
matplotlib.rcParams["ytick.labelsize"] = 10
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["legend.facecolor"] = "w"
matplotlib.rcParams["savefig.transparent"] = False

####--------------------------------------------------------------------------.
#### Define analysis time period
start_time = datetime.datetime.strptime("2016-03-09 10:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2016-03-09 11:00:00", "%Y-%m-%d %H:%M:%S")

products = ["2A-DPR", "2B-GPM-CORRA", "1C-GMI", "2A-GMI"]

version = 7
product_type = "RS"

# gpm_api.download(
#     product=product,
#     start_time=start_time,
#     end_time=end_time,
#     version=version,
#     progress_bar=True,
#     n_threads=5,  # 8
#     transfer_tool="curl",
# )

####--------------------------------------------------------------------------.
#### Open datasets
dict_product = {}
for product in products:
    ds = gpm_api.open_dataset(
        product=product,
        start_time=start_time,
        end_time=end_time,
        # Optional
        version=version,
        product_type=product_type,
        chunks={},
        decode_cf=True,
        prefix_group=False,
    )
    dict_product[product] = ds


ds_dpr = dict_product["2A-DPR"]
ds_corra = dict_product["2B-GPM-CORRA"]
ds_gpmi_2a = dict_product["2A-GMI"]
ds_gpmi_1c = dict_product["1C-GMI"]

# list(ds_gpmi_2a.data_vars)
# list(ds_gpmi_1c.data_vars) # 'Tc'

##----------------------------------------------------------------------------.
#### Identify precipitation area sorted by maximum intensity
# --> label = 0 is rain below min_value_threshold
da_precip = ds_dpr["precipRateNearSurface"]

label_name = "label_precip_max_intensity"
da_precip = da_precip.ximage.label(
    min_value_threshold=1,
    min_area_threshold=5,
    label_name=label_name,
    sort_by="maximum",
    footprint=3,
)
plot_labels(da_precip[label_name])


# Identify patch slices around precipitating areas
n_labels = 3  # select three precipitating areas
patch_size = 49
centered_on = "label_bbox"
padding = 0
highlight_label_id = False

patches_isel_dicts = da_precip.ximage.label_patches_isel_dicts(
    label_name=label_name,
    patch_size=patch_size,
    # Output options
    n_labels=n_labels,
    # Patch extraction Options
    centered_on=centered_on,
    padding=padding,
)

pprint(patches_isel_dicts)

# Select label
label_id = 1
for label_id in patches_isel_dicts.keys():
    label_patch_isel_dict = patches_isel_dicts[label_id][0]
    da_patch = da_precip.isel(label_patch_isel_dict)
    da_patch.gpm_api.plot_map()


####--------------------------------------------------------------------------.
#### Define label_id (AOI) for the rest of the example
# Define label_id
label_id = 1
# Retrieve isel dictionary
label_patch_isel_dict = patches_isel_dicts[label_id][0]
# Retrieve patch DataArray
da_patch = da_precip.isel(label_patch_isel_dict)
# Retrieve AOI extent
extent = da_patch.gpm_api.extent(padding=0)

####--------------------------------------------------------------------------.
#### Subset all product to the AOI
dict_product_patch = {}
for product, ds in dict_product.items():
    dict_product_patch[product] = ds.gpm_api.crop(extent)

####--------------------------------------------------------------------------.
#### Compare 2A-GMI, 2B-GPM-CORRA and 2A-DPR surface precipitation
product_var_dict = {
    "2A-GMI": [
        "surfacePrecipitation",
        "mostLikelyPrecipitation",
        # 'precip1stTertial',
        # 'precip2ndTertial',
        "convectivePrecipitation",
        "frozenPrecipitation",
    ],
    "2A-DPR": [
        "precipRateAve24",  # Average of precipitation rate for 2 to 4km height
        "precipRateNearSurface",
        "precipRateESurface",
        "precipRateESurface2",
    ],
    "2B-GPM-CORRA": [
        "nearSurfPrecipTotRate",
        "nearSurfPrecipLiqRate",
        "estimSurfPrecipTotRate",
        "estimSurfPrecipLiqRate",
        # "OEestimSurfPrecipTotRate",
        # "OEestimSurfPrecipLiqRate",
    ],
}

for product, list_vars in product_var_dict.items():
    for var in list_vars:
        print(product, var)
        name = product + " - " + var
        da = dict_product_patch[product][var]
        p = da.gpm_api.plot_map()
        p.axes.set_extent(extent)
        p.axes.set_title(name)
        plt.show()

####--------------------------------------------------------------------------.
#### Compare 1-C-GMI measured and 2B-GPM-CORRA simulated brightness temperatures
product_var_dict = {
    "1C-GMI": [
        "Tc",
    ],
    "2B-GPM-CORRA": [
        "simulatedBrightTemp",
    ],
}

for product, list_vars in product_var_dict.items():
    for var in list_vars:
        print(product, var)
        da = dict_product_patch[product][var]
        for i in range(da["pmw_frequency"].shape[0]):
            tmp_da = da.isel(pmw_frequency=i)
            frequency = tmp_da["pmw_frequency"].item()
            title = f"{product} {frequency}"
            p = tmp_da.gpm_api.plot_map(vmin=100, vmax=280)
            p.axes.set_extent(extent)
            p.axes.set_title(title)
            plt.show()

####--------------------------------------------------------------------------.
#### Compare 2A-GMI and 2B-GPM-CORRA rain, cloud, ice water paths

## Compute 2B-GPM-CORRA cloudWaterPath and iceWaterPath from the profile
da_lwc = dict_product_patch["2B-GPM-CORRA"]["cloudLiqWaterCont"].compute()
da_iwc = dict_product_patch["2B-GPM-CORRA"]["cloudIceWaterCont"].compute()  # all NaN !

da_clwp = da_lwc.gpm_api.integrate_profile_concentration(
    name="cloudLiquidWaterPath", scale_factor=1000, units="kg/m²"
)
da_ciwp = da_iwc.gpm_api.integrate_profile_concentration(
    name="cloudIceWaterPath", scale_factor=1000, units="kg/m²"
)
np.unique(da_clwp)
np.unique(da_ciwp)  # all NaN !

da_clwp = da_clwp.compute()
da_clwp.gpm_api.plot_map()

da_ciwp = da_ciwp.compute()
da_ciwp.gpm_api.plot_map()

# Assign values
dict_product_patch["2B-GPM-CORRA"]["cloudLiquidWaterPath"] = da_clwp
dict_product_patch["2B-GPM-CORRA"]["cloudIceWaterPath"] = da_ciwp

### Define products to display
product_var_dict = {
    "2A-GMI": [
        "rainWaterPath",  # kg/m^2 [0-3000]
        "iceWaterPath",  # kg/m^2 [0-3000]
        "cloudWaterPath",  # kg/m^2 [0-3000]
    ],
    "2B-GPM-CORRA": [
        "cloudLiquidWaterPath",  # manually computed !
        "cloudIceWaterPath",  # manually computed !
    ],
}

for product, list_vars in product_var_dict.items():
    for var in list_vars:
        print(product, var)
        da = dict_product_patch[product][var]
        title = product + " - " + var
        p = da.gpm_api.plot_map()
        p.axes.set_extent(extent)
        p.axes.set_title(title)
        plt.show()


# da_clwp = dict_product_patch["2A-GMI"]["cloudWaterPath"]
# da_clwp.gpm_api.plot_map()

# da_iwp = dict_product_patch["2A-GMI"]["iceWaterPath"]
# da_iwp.gpm_api.plot_map()

# da_rlwp = dict_product_patch["2A-GMI"]["rainWaterPath"]
# da_rlwp.gpm_api.plot_map()

# da_lwp = da_clwp + da_rlwp
# da_lwp.name = "liquidWaterPath"
# da_lwp.gpm_api.plot_map()


####--------------------------------------------------------------------------.
