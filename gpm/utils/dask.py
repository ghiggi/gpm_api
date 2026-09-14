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
"""This module contains utilities for Dask Distributed processing."""

import ctypes
import logging
import os
import platform


def trim_memory() -> int:
    os_name = platform.system()
    if os_name == "Linux":
        libc = ctypes.CDLL("libc.so.6")
        return libc.malloc_trim(0)
    # elif os_name == "Windows":
    #     # Windows does not have a direct equivalent
    #     pass
    # elif os_name == "Darwin":
    #     # macOS (Darwin) does not have a direct equivalent
    #     pass
    return -1  # Indicate no operation was performed


def clean_memory(client):
    """Call the garbage collector on each process.

    See https://distributed.dask.org/en/latest/worker-memory.html#manually-trim-memory
    """
    client.run(trim_memory)


def get_client():
    from dask.distributed import get_client

    return get_client()


def get_scheduler(get=None, collection=None):
    """Determine the dask scheduler that is being used.

    None is returned if no dask scheduler is active.

    See Also
    --------
    dask.base.get_scheduler

    """
    try:
        import dask
        from dask.base import get_scheduler

        actual_get = get_scheduler(get, collection)
    except ImportError:
        return None

    try:
        from dask.distributed import Client

        if isinstance(actual_get.__self__, Client):
            return "distributed"
    except (ImportError, AttributeError):
        pass

    try:
        if actual_get is dask.multiprocessing.get:
            return "multiprocessing"
    except AttributeError:
        pass

    return "threaded"


def initialize_dask_cluster(minimum_memory=None):
    """Initialize Dask Cluster."""
    import dask
    import psutil

    # Silence dask warnings
    # dask.config.set({'distributed.worker.multiprocessing-method': 'forkserver'})
    # dask.config.set({"distributed.worker.multiprocessing-method": "spawn"})
    # dask.config.set({"logging.distributed": "error"})
    # Import dask.distributed after setting the config
    from dask.distributed import Client, LocalCluster
    from dask.utils import parse_bytes

    # Set HDF5_USE_FILE_LOCKING to avoid going stuck with HDF
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    # Retrieve the number of processes to run
    # --> If DASK_NUM_WORKERS is not set, use all CPUs minus 2
    available_workers = os.cpu_count() - 2  # if not set, all CPUs minus 2
    num_workers = dask.config.get("num_workers", available_workers)

    # If memory limit specified, ensure correct amount of workers
    if minimum_memory is not None:
        # Compute available memory (in bytes)
        total_memory = psutil.virtual_memory().total
        # Get minimum memory per worker (in bytes)
        minimum_memory = parse_bytes(minimum_memory)
        # Determine number of workers constrained by memory
        maximum_workers_allowed = max(1, total_memory // minimum_memory)
        # Respect both CPU and memory requirements
        num_workers = min(maximum_workers_allowed, num_workers)

    # Create dask.distributed local cluster
    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=1,
        processes=True,
        memory_limit=0,  # this avoid flexible dask memory management
        silence_logs=logging.ERROR,
    )
    client = Client(cluster)
    return cluster, client


def close_dask_cluster(cluster, client):
    """Close Dask Cluster."""
    logger = logging.getLogger()
    # Backup current log level
    original_level = logger.level
    logger.setLevel(logging.CRITICAL + 1)  # Set level to suppress all logs
    # Close cluster
    # - Avoid log 'distributed.worker - ERROR - Failed to communicate with scheduler during heartbeat.'
    try:
        cluster.close()
        client.close()
    finally:
        # Restore the original log level
        logger.setLevel(original_level)
