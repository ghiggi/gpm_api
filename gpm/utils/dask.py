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
