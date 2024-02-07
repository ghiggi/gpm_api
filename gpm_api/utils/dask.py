#!/usr/bin/env python3
"""
Created on Wed Aug  9 10:30:36 2023

@author: ghiggi
"""
import ctypes
import platform


def trim_memory() -> int:
    os_name = platform.system()
    if os_name == "Linux":
        libc = ctypes.CDLL("libc.so.6")
        return libc.malloc_trim(0)
    elif os_name == "Windows":
        # Windows does not have a direct equivalent
        pass
    elif os_name == "Darwin":
        # macOS (Darwin) does not have a direct equivalent
        pass
    return -1  # Indicate no operation was performed


def clean_memory(client):
    """
    Call the garbage collector on each process.

    See https://distributed.dask.org/en/latest/worker-memory.html#manually-trim-memory
    """
    client.run(trim_memory)


def get_client():
    from dask.distributed import get_client

    return get_client()
