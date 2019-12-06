#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from setuptools import setup, find_packages
import numpy as np

setup(
        name = 'geometric_clustering',
        version = '1.0',
        include_dirs = ['utils',np.get_include()], #Add Include path of numpy
        packages=['.'],
        install_requires=['numpy', 
                          'scipy', 
                          'networkx', 
                          'matplotlib', 
                          'cython', 
                          'POT'],
      )
