import pytest
import h5py
import os


@pytest.fixture
def sample_dataset():
    """Return a sample dataset to use for testing

    Dataset is a 2A DPR file from 2022-07-05 that has been generated with
    test_granule_creation.py to maintain structure but remove data

    """

    return h5py.File(
        os.path.join(
            os.getcwd(),
            "gpm_api",
            "tests",
            "resources",
            "GPM",
            "RS",
            "V07",
            "RADAR",
            "2A-DPR",
            "2022",
            "07",
            "05",
            "2A.GPM.DPR.V9-20211125.20220705-S144632-E161905.047447.V07A.HDF5",
        )
    )
