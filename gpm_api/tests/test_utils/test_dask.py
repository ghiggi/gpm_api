#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:58:56 2024

@author: ghiggi
"""
import pytest
from dask.distributed import Client, LocalCluster
from gpm_api.utils.dask import trim_memory, clean_memory, get_client


@pytest.fixture(scope="module")
def dask_client():
    """Fixture for creating and closing a Dask client."""
    cluster = LocalCluster()
    client = Client(cluster)
    yield client
    # Teardown: close the client and cluster after tests are done
    client.close()
    cluster.close()


def test_trim_memory():
    """Test the trim_memory function actually interacts with libc and does not raise errors."""
    trim_memory()  # 1
    assert True


def test_clean_memory(dask_client):
    """Test clean_memory actually calls trim_memory on Dask workers without errors."""
    clean_memory(dask_client)


def test_get_client(dask_client):
    """Test get_client returns the current Dask client."""
    # Within the context of this test, the dask_client fixture is the active client
    assert get_client() == dask_client
