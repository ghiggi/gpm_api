#!/usr/bin/env python3
"""
Created on Tue Jul 25 19:25:07 2023

@author: ghiggi
"""
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
