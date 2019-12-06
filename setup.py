#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from setuptools import setup, find_packages
import numpy as np

setup(
        name = 'geometric_clustering',
        version = '1.0',
        include_dirs = ['utils',np.get_include()], #Add Include path of numpy
        packages=['.'],
        scripts=['utils/curvature_utils.py', 
                 'utils/clustering_utils.py', 
                 'utils/embedding_utils.py'],
        install_requires=['numpy', 
                          'scipy', 
                          'networkx', 
                          'matplotlib', 
                          'cython', 
                          'POT'],
      )
