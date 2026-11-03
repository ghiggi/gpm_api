# -----------------------------------------------------------------------------.
# MIT License

# Copyright (c) 2024 GPM-API developers
#
# This file is part of GPM-API.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -----------------------------------------------------------------------------.
"""This module test the dask utilities."""

import logging
import os

import pytest
from dask.distributed import Client, LocalCluster

from gpm.utils.dask import (
    clean_memory,
    close_dask_cluster,
    get_client,
    initialize_dask_cluster,
    trim_memory,
)


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


@pytest.fixture(autouse=True)
def reset_dask_scheduler():
    """Force all tests to run with scheduler=None."""
    import dask

    with dask.config.set(scheduler=None):
        yield


class TestInitializeDaskCluster:
    """Test initialize_dask_cluster."""

    def test_initialize_and_close_cluster(self, reset_dask_scheduler):
        """Test basic initialization and closing of dask cluster."""
        cluster, client = initialize_dask_cluster()
        try:
            # Basic checks
            assert cluster.scheduler_address
            assert client.status == "running"
            assert os.environ.get("HDF5_USE_FILE_LOCKING") == "FALSE"
        finally:
            close_dask_cluster(cluster, client)
            # After closing, the client should not be running
            assert client.status != "running"

    def test_initialize_with_memory_constraint(self, reset_dask_scheduler):
        """Test setting an excessive memory enforce 1 worker."""
        # Request a huge memory requirement → should fallback to at least 1 worker
        cluster, client = initialize_dask_cluster(minimum_memory="10TB")
        try:
            assert len(cluster.workers) == 1
        finally:
            close_dask_cluster(cluster, client)

    def test_close_dask_cluster_restores_log_level(self, reset_dask_scheduler):
        """Test that closing dask cluster restores original log level."""
        cluster, client = initialize_dask_cluster()
        logger = logging.getLogger()
        original_level = logger.level
        close_dask_cluster(cluster, client)
        assert logger.level == original_level
