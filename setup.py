#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup

long_description = """A Python code for download and analyze GPM products."""

setup(
    name="gpm_api",
    description="xarray-based API to analyze GPM products",
    author="Gionata Ghiggi & Randy J. Chase ",
    author_email="gionata.ghiggi@epfl.ch, randy.chase12@gmail.com",
    url="https://github.com/ghiggi/gpm_api",
    packages=["gpm_api"],
    long_description=long_description,
    license="MIT",
    requires=["h5py", "xarray", "trollsift", "curl", "wget", "cartopy", "tqdm"],
)
