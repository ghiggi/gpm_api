import datetime
import os
import time

import numpy as np
import pandas as pd

from gpm_api.io import GPM_PMW_2A_GPROF_RS_products

# from distributed import Client, LocalCluster
# # Set Dask Client
# client = Client(processes=False)
# client = Client(processes=True)  # LTESRV1 DEFAULT: 6 PROCESSES AND 4 THREADS EACH
# cluster = LocalCluster(n_workers = 12,        # n processes
#                        threads_per_worker=2, # n_workers*threads_per_worker
#                        processes=True,
#                        memory_limit='512GB')
# client = Client(cluster)


BASE_DIR = "/home/ghiggi"
OVERPASS_TABLE_DIR = "/home/ghiggi/Overpass/Tables"
username = "gionata.ghiggi@epfl.ch"

#### Define analysis time period
start_time = datetime.datetime.strptime("2020-08-01 12:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.datetime.strptime("2020-08-10 12:00:00", "%Y-%m-%d %H:%M:%S")

#### Define PMW products
products = GPM_PMW_2A_GPROF_RS_products()
product_type = "RS"

####--------------------------------------------------------------------------.
#### Define statistics to compute
def _overpass_stats(df):
    """Compute overpass statistics."""
    timesteps = df["time"]
    timesteps = sorted(timesteps)
    n_overpass = len(timesteps)
    if n_overpass >= 2:
        intervals = np.diff(timesteps)
        dict_stats = {
            "n_overpass": np.array([n_overpass]),
            "min": np.array([np.min(intervals)]),
            "mean": np.array([np.mean(intervals)]),
            "median": np.array([np.median(intervals)]),
            "max": np.array([np.max(intervals)]),
        }
    else:
        dict_stats = {
            "n_overpass": np.array([n_overpass]),
            "min": np.array([np.nan]),
            "mean": np.array([np.nan]),
            "median": np.array([np.nan]),
            "max": np.array([np.nan]),
        }
    df_stats = pd.DataFrame(dict_stats)
    return df_stats


####--------------------------------------------------------------------------.
t_i = time.time()
#### Compute overpass intervals
# Read overpass tables
print("-Read overpass tables")
list_df = []
for product in products:
    fpath = os.path.join(OVERPASS_TABLE_DIR, product + ".parquet")
    if os.path.exists(fpath):
        df = pd.read_parquet(fpath)
        list_df.append(df)

df = pd.concat(list_df)


df = df.drop(columns=["satellite"])

# df = df1
# df = df.iloc[0:10000]

# Compute overpass statistics
print("- Compute overpass statistics")
df_grouped = df.groupby(["lon", "lat"], as_index=True)
df_stats = df_grouped.apply(_overpass_stats)

t_f = time.time()
print((t_f - t_i) / 60, "minutes")

# Cast to seconds to avoid problem with parquet
print("-Cast timedelta to seconds ")
df_stats["min"] = df_stats["min"].astype("m8[s]")
df_stats["mean"] = df_stats["mean"].astype("m8[s]")
df_stats["median"] = df_stats["median"].astype("m8[s]")
df_stats["max"] = df_stats["max"].astype("m8[s]")

# Save overpass statistics to disk (Parquet)
print("- Write statistics")
fpath = os.path.join(OVERPASS_TABLE_DIR, "Overpass_stats.parquet")
df_stats.to_parquet(fpath, engine="pyarrow")

# Create xarray object
ds_stats = df_stats.to_xarray()
ds_stats = ds_stats.assign_coords(
    {
        "lat": ds_stats["lat"] / 10,
        "lon": ds_stats["lon"] / 10,
    }
)
ds_stats = ds_stats.squeeze()

# Save to netCDF
nc_fpath = os.path.join(OVERPASS_TABLE_DIR, "Overpass_stats.nc")
ds_stats.to_netcdf(nc_fpath)

# ds_stats['min'].plot.imshow(x="lon", y="lat")
###--------------------------------------------------------------------------.
# # Dask-based (sort and split by longitude)
# fpath =

# df = df.sort_values(['lon'], ascending=[True])
# df1 = dd.from_pandas(df, npartitions=360*10)
# # df1.divisions

# df1 = df1.repartition(partition_size='200MB')

# fpath = os.path.join(OVERPASS_TABLE_DIR, "PixelBasedParquet1")
# dd.to_parquet(df1, fpath, partition_on=['lon','lat'],
#               engine='pyarrow',
#               compression='snappy')


# df_stats = df1.groupby(['lon','lat']).apply(_overpass_stats)
# df_stats = df_stats.compute()
# df_stats = df_stats.sort_index()

# # Save overpass statistics to disk (Parquet)
# fpath = os.path.join(OVERPASS_TABLE_DIR, "Overpass_stats.parquet")
# df_stats.to_parquet(fpath)

# # Create xarray object
# ds_stats = df_stats.to_xarray()
# ds_stats = ds_stats.assign_coords({'lat': ds_stats['lat']/10,
#                                    'lon': ds_stats['lon']/10,
#                                    })
# ds_stats = ds_stats.drop(['level_2'])

# # Save to netCDF
# nc_fpath = os.path.join(OVERPASS_TABLE_DIR, "Overpass_stats.nc")
# ds_stats.to_netcdf(nc_fpath)

# ###--------------------------------------------------------------------------.
# # Deve
# idx = np.logical_and(np.isin(df['lon'], np.arange(1500, 1600)), df['lat'] == -608)
# idx = np.logical_and(df['lon'] >= 0, df['lat'] == -608)
# df1 = df[idx]
# df_stats1 = df1.groupby(['lon','lat']).apply(_overpass_stats)

# # Save parquet by longitude
# # list_df = [x for _, x in DF.groupby('chr', as_index=False)

# df = df.repartition(partition_size='200MB')
# dd.to_parquet(df, fpath, partition_on=['lon','lat'],
#               engine='pyarrow',
#               compression='snappy')


# df = dask.dataframe.read_parquet(fpath,
#     engine='pyarrow',
#     nthreads=8,
# )

# https://arrow.apache.org/docs/python/parquet.html
# https://docs.dask.org/en/stable/dataframe-parquet.html
# https://docs.dask.org/en/stable/generated/dask.dataframe.to_parquet.html

# ### With pyarrow
# import pyarrow as pa
# import pyarrow.parquet as pq
# fpath = os.path.join(OVERPASS_TABLE_DIR, "PixelBasedParquet")
# table = pa.Table.from_pandas(df)
# pq.write_to_dataset(table, root_path=fpath,
#                     partition_cols=['lat','lon'])


# idx2 = np.logical_and(df['lon'] == 1503, df['lat'] == -608)
# df2 = df[idx2]
# df2 = df2.sort_values('time')
