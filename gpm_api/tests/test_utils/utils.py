import datetime
import numpy as np
from typing import Union
import xarray as xr

from gpm_api.utils import checks as gpm_checks


def create_fake_datetime_array_from_hours_list(hours: Union[list, np.ndarray]) -> np.ndarray:
    """Convert list of integers and NaNs into a np.datetime64 array"""

    datetimes = []
    for hour in hours:
        if np.isnan(hour):
            datetimes.append(np.datetime64("NaT"))
        else:
            datetimes.append(
                np.datetime64(
                    datetime.datetime(2020, 12, 31, 0, 0, 0) + datetime.timedelta(hours=int(hour))
                )
            )

    return np.array(datetimes)


def get_time_range(start_hour: int, end_hour: int) -> np.ndarray:
    return create_fake_datetime_array_from_hours_list(np.arange(start_hour, end_hour))


def create_dataset_with_coordinate(coord_name: str, coord_values: np.ndarray) -> xr.Dataset:
    """Create a dataset with a single coordinate"""

    ds = xr.Dataset()
    ds[coord_name] = coord_values
    return ds


def create_orbit_time_array(time_template: Union[list, np.ndarray]) -> np.ndarray:
    """Create a time array with ORBIT_TIME_TOLERANCE as unit"""

    start_time = np.datetime64("2020-12-31T00:00:00")
    time = np.array([start_time + gpm_checks.ORBIT_TIME_TOLERANCE * t for t in time_template])
    return time
