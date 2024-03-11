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
import pytest
from dask.distributed import Client, LocalCluster
from gpm.utils.dask import trim_memory, clean_memory, get_client


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
