#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup

long_description = """A Python code for loading GPM-DPR files into xarray datasets
"""

setup(name='gpm_api',
      description='python package for all things GPM',
      author='Gionata Ghiggi & Randy J. Chase ',
      author_email='gionata.ghiggi@gmail.com,randy.chase12@gmail.com',
      url='https://github.com/ghiggi/gpm_api',
      packages=['gpm_api'],
      long_description = long_description,
      license = 'MIT',
      requires = ['h5py', 'xarray','yaml','numpy']
     )
