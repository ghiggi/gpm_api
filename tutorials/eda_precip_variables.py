#!/usr/bin/env python3
"""
Created on Sun Aug 14 13:52:46 2022

@author: ghiggi
"""
import datetime
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import gpm_api

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

products = [
    "2A-DPR",
    "2A-GMI",
    "2B-GPM-CORRA",
]

product_var_dict = {
    "2A-DPR": [
        "precipRateNearSurface",
        "precipRateESurface",
        "precipRateESurface2",
    ],
    "2B-GPM-CORRA": [
        "nearSurfPrecipTotRate",
        "estimSurfPrecipTotRate",
    ],
    "2A-GMI": ["surfacePrecipitation"],
}

version = 7
product_type = "RS"

####--------------------------------------------------------------------------.
#### Download products
for product in products:
    print(product)
    gpm_api.download(
        product=product,
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
        force_download=False,
        transfer_tool="curl",
        progress_bar=True,
        verbose=True,
        n_threads=2,
    )


####--------------------------------------------------------------------------.
#### Open datasets
dict_product = {}
for product, variables in product_var_dict.items():
    ds = gpm_api.open_dataset(
        product=product,
        start_time=start_time,
        end_time=end_time,
        # Optional
        variables=variables,
        version=version,
        product_type=product_type,
        chunks={},
        decode_cf=True,
        prefix_group=False,
    )
    dict_product[product] = ds


np.set_printoptions(suppress=True)

# -----------------------------------------------------------------------------.
# Check difference between products DPR NearSurface and EstimatedSurface precipitation estimates
diff = (
    dict_product["2A-DPR"]["precipRateNearSurface"] - dict_product["2A-DPR"]["precipRateESurface"]
)

# Check difference between products CORRA NearSurface and EstimatedSurface precipitation estimates
diff = (
    dict_product["2B-GPM-CORRA"]["nearSurfPrecipTotRate"]
    - dict_product["2B-GPM-CORRA"]["estimSurfPrecipTotRate"]
)

# Check difference between products DPR Estimated NearSurface precipitation estimates
diff = dict_product["2A-DPR"]["precipRateESurface"] - dict_product["2A-DPR"]["precipRateESurface2"]

# Check difference between products DPR and CORRA NearSurface precipitation estimates
# --> 2B-GPM-CORRA has much lower rain rates !!!
diff = (
    dict_product["2A-DPR"]["precipRateNearSurface"]
    - dict_product["2B-GPM-CORRA"]["nearSurfPrecipTotRate"]
)

# Check difference between products DPR and CORRA Estimated NearSurface precipitation estimates
# --> 2B-GPM-CORRA has much lower rain rates !!!
diff = (
    dict_product["2A-DPR"]["precipRateESurface"]
    - dict_product["2B-GPM-CORRA"]["estimSurfPrecipTotRate"]
)


# Analyze difference values
diff = diff.compute()
values, counts = np.unique(diff.data.flatten().round(1), return_counts=True)
print(values)
print(counts)
plt.hist(diff.data.flatten(), bins=100, range=[min(values), max(values)])
plt.yscale("log")  # Set log scale on the x-axis
plt.xlabel("DIfference [in mm/h]")
plt.ylabel("Frequency")

# -----------------------------------------------------------------------------.
# Get variable min, max statistics !
dict_min = {}
dict_max = {}

for product, ds in dict_product.items():
    ds = ds.compute()
    for var in list(ds.data_vars):
        ds[var] = ds[var].where(ds[var] > 0.00000001)
        name = product + "-" + var
        dict_max[name] = np.nanmax(ds[var].data).item()
        dict_min[name] = np.nanmin(ds[var].data).item()

pprint(dict_max)
pprint(dict_min)

# CORRA and GMI max are irrealistic !!!

# -----------------------------------------------------------------------------.
#### Empirical counts
labels = []
for rounding in [0, 1, 2]:
    for product, ds in dict_product.items():
        ds = ds.compute()
        for var in list(ds.data_vars):
            data = ds[var].data.flatten()
            name = product + "-" + var
            labels.append(name)
            values, counts = np.unique(data.round(rounding), return_counts=True)
            plt.plot(values, counts)
            plt.yscale("log")  # Set log scale on the x-axis
            plt.xscale("log")  # Set log scale on the x-axis
    plt.legend(labels)
    plt.show()
