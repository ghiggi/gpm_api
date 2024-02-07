#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:19:45 2024

@author: ghiggi
"""
import pytest
from gpm_api.utils.timing import print_elapsed_time, print_task_elapsed_time


# Sample function to decorate
def add(a, b):
    return a + b


# Test for print_elapsed_time
def test_print_elapsed_time(capsys):
    @print_elapsed_time
    def add_decorated(a, b):
        return add(a, b)

    result = add_decorated(2, 3)
    captured = capsys.readouterr()  # Capture the print output

    assert result == 5, "Function result incorrect"
    assert "Elapsed time: " in captured.out


# Test for print_task_elapsed_time with custom prefix
def test_print_task_elapsed_time(capsys):
    @print_task_elapsed_time(prefix="MY_CUSTOM_TASK")
    def add_decorated(a, b):
        return add(a, b)

    result = add_decorated(2, 3)
    captured = capsys.readouterr()  # Capture the print output

    assert result == 5, "Function result incorrect"
    assert "MY_CUSTOM_TASK Elapsed time: " in captured.out
