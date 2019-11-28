#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from setuptools import setup, find_packages

setup(
        name = 'geometric_clustering',
        version = '0.1',
        packages=['.'],
        install_requires=['numpy', 
                        'scipy', 
                         'networkx', 
                         'matplotlib', 
                         'cython', 
                         'POT'],
      )
