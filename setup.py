#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#import setuptools
from distutils.core import setup

setup(
        name = 'geometric_clustering',
        version = '1.0',
        packages=['geometric_clustering'],
        py_modules = ['geometric_clustering.utils.curvature_utils',
                      'geometric_clustering.utils.clustering_utils',
                      'geometric_clustering.utils.embedding_utils',
                      'geometric_clustering.utils.misc'],
        install_requires=['numpy', 
                          'scipy', 
                          'networkx', 
                          'matplotlib', 
                          'cython', 
                          'POT'],
      )
