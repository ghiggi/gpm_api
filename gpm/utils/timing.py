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
"""This module contains decorators which measure the function time of execuution."""

import datetime
import time
from time import perf_counter


def print_elapsed_time(fn):
    def decorator(*args, **kwargs):
        start_time = time.perf_counter()
        results = fn(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        timedelta_str = str(datetime.timedelta(seconds=execution_time))
        print(f"Elapsed time: {timedelta_str} .", end="\n")
        return results

    return decorator


def print_task_elapsed_time(prefix=" - "):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = perf_counter()
            results = func(*args, **kwargs)
            end_time = perf_counter()
            execution_time = end_time - start_time
            timedelta_str = str(datetime.timedelta(seconds=execution_time))
            print(f"{prefix} Elapsed time: {timedelta_str} .", end="\n")
            return results

        return wrapper

    return decorator
