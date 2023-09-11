import numpy as np
import pytest
import h5py
import os
from gpm_api import open_dataset
from gpm_api.dataset.granule import _open_granule

import xarray as xr


class SaneEqualityArray(np.ndarray):
    """Wrapper class for numpy array allowing deep equality tests on objects containing numpy arrays.

    From https://stackoverflow.com/a/14276901
    """

    def __new__(cls, array):
        """Create a new SaneEqualityArray from array only (instead of shape + type + array)."""

        if isinstance(array, list):  # No need to wrap regular lists
            return array

        return np.asarray(array).view(cls)

    def __eq__(self, other):
        # Only use equal_nan for floats dtypes
        equal_nan = np.issubdtype(self.dtype, np.floating)
        return (
            isinstance(other, np.ndarray)
            and self.shape == other.shape
            and np.array_equal(self, other, equal_nan=equal_nan)
        )


@pytest.fixture()
def sample_dataset() -> xr.Dataset:
    """Return a sample dataset to use for testing

    Dataset is a 2A DPR file from 2022-07-05 that has been generated with
    test_granule_creation.py to maintain structure but remove data

    """

    # os.path.join(
    #     os.getcwd(),
    #     "gpm_api",
    #     "tests",
    #     "resources",
    #     "GPM",
    #     "RS",
    #     "V07",
    #     "PMW",
    #     "1C-MHS-METOPB",
    #     "2020",
    #     "08",
    #     "01",
    #     "1C.METOPB.MHS.XCAL2016-V.20200801-S102909-E121030.040841.V07A.HDF5",
    # )
    # )

    # ds = open_dataset(
    # start_time="2022-07-01T10:29:09",
    # end_time="2022-08-01T12:10:30",
    # product="1C-MHS-METOPB",
    # )

    ds = _open_granule(
        os.path.join(
            os.getcwd(),
            "gpm_api",
            "tests",
            "resources",
            "GPM",
            "RS",
            "V07",
            "PMW",
            "1C-MHS-METOPB",
            "2020",
            "08",
            "01",
            "1C.METOPB.MHS.XCAL2016-V.20200801-S102909-E121030.040841.V07A.HDF5",
        )
    )

    return ds


def pytest_configure():
    pytest.SaneEqualityArray = SaneEqualityArray
