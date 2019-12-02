#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from setuptools import setup, find_packages

setup(
        name = 'geometric_clustering',
<<<<<<< HEAD
        version = '1.0',
        include_dirs = ['utils',np.get_include()], #Add Include path of numpy
=======
        version = '0.1',
>>>>>>> b676661b99fc7fe8ab58d404c78cf04b2b80a5b8
        packages=['.'],
        install_requires=['numpy', 
                        'scipy', 
                         'networkx', 
                         'matplotlib', 
                         'cython', 
                         'POT'],
      )
