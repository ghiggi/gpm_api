#!/usr/bin/env python3
"""
Created on Fri Jul 21 14:57:46 2023

@author: ghiggi
"""
import datetime

import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import ximage  # noqa

import gpm

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

products_scan_mode_dict = {
    "2A-DPR": ["FS"],
    "1C-GMI": ["S1", "S2"],
    "2A-GMI": ["S1"],
}

version = 7
product_type = "RS"

# gpm.download(
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
for product, scan_modes in products_scan_mode_dict.items():
    for scan_mode in scan_modes:
        name = product + "-" + scan_mode
        ds = gpm.open_dataset(
            product=product,
            start_time=start_time,
            end_time=end_time,
            # Optional
            version=version,
            product_type=product_type,
            scan_mode=scan_mode,
            chunks={},
            decode_cf=True,
            prefix_group=False,
        )
        dict_product[name] = ds


ds_dpr = dict_product["2A-DPR-FS"]
ds_gmi_2a_s1 = dict_product["2A-GMI-S1"]
ds_gmi_1c_s1 = dict_product["1C-GMI-S1"]
ds_gmi_1c_s2 = dict_product["1C-GMI-S2"]
# ds_gpmi_1c_s1["pmw_frequency"]
# ds_gpmi_1c_s2["pmw_frequency"]

##----------------------------------------------------------------------------.
#### Resample data from one swath to the other
ds_gmi_1c_s1_r = ds_gmi_1c_s1.gpm.remap_on(ds_dpr)
ds_gmi_1c_s2_r = ds_gmi_1c_s2.gpm.remap_on(ds_dpr)
ds_gmi_1c = xr.concat([ds_gmi_1c_s1_r, ds_gmi_1c_s2_r], dim="pmw_frequency")

gmi_2a_variables = [
    "surfacePrecipitation",
    "mostLikelyPrecipitation",
    "cloudWaterPath",  # kg/m2 .
    "rainWaterPath",  # kg/m2 .
    "iceWaterPath",  # kg/m2 .
]
ds_gmi_2a_r = ds_gmi_2a_s1[gmi_2a_variables].gpm.remap_on(ds_dpr)

##----------------------------------------------------------------------------.
# Select DataArray
variable = "precipRateNearSurface"
da_precip = ds_dpr[variable].load()


# Label the object to identify precipitation areas
label_name = "precip_label_area"
da_precip = da_precip.ximage.label(min_value_threshold=0.1, label_name=label_name)

# Define the patch generator
labels_id = None
labels_id = [1, 2, 3]
patch_size = 2  # min patch size
centered_on = "label_bbox"
padding = 0
highlight_label_id = False

patch_gen = da_precip.ximage.label_patches(
    label_name=label_name,
    patch_size=patch_size,
    # Output options
    labels_id=labels_id,
    highlight_label_id=highlight_label_id,
    # Patch extraction Options
    centered_on=centered_on,
    padding=padding,
)

patch_idx = 0
list_label_patches = list(patch_gen)
label_id, da_dpr_patch = list_label_patches[patch_idx]

# Retrieve AOI extent
extent = da_dpr_patch.gpm.extent(padding=0)
ds_gmi_1c_patch = ds_gmi_1c.gpm.crop(extent)
ds_gmi_2a_r_patch = ds_gmi_2a_r.gpm.crop(extent)

ds_gmi_1c_patch["Tc"] = ds_gmi_1c_patch["Tc"].compute()
ds_gmi_2a_r_patch = ds_gmi_2a_r_patch.compute()

# Plot DPR
da_dpr_patch.gpm.plot_image()

# Plot GMI-2A remapped data
for var in ds_gmi_2a_r_patch.data_vars:
    da = ds_gmi_2a_r_patch[var]
    title = f"{product} {var}"
    p = da.gpm.plot_image()
    p.axes.set_title(title)
    plt.show()

# Plot GMI-1C remapped data
for var in ["Tc"]:  # ["Tc","Quality"]:
    for i in range(0, ds_gmi_1c["pmw_frequency"].shape[0]):
        da = ds_gmi_1c_patch[var].isel(pmw_frequency=i)
        frequency = da["pmw_frequency"].item()
        title = f"{product} {var} {frequency}"
        p = da.gpm.plot_image()
        p.axes.set_title(title)
        plt.show()


# ds_gmi_1c["Tc"].isel({"pmw_frequency": 0}).gpm.plot_image()
# ds_gmi_1c["Tc"].isel({"pmw_frequency": 0}).gpm.plot_map()

# -----------------------------------------------------------------------------.
