#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from setuptools import setup, find_packages
import numpy as np

setup(
        name = 'geometric_clustering',
        version = '1.0',
        packages=['geometric_clustering'],
        install_requires=['numpy', 
                          'scipy', 
                          'networkx', 
                          'matplotlib', 
                          'cython', 
                          'POT'],
      )
