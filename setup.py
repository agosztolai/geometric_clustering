#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from setuptools import setup, find_packages

setup(
        name = 'geocluster',
        version = '1.0',
        install_requires=['numpy', 
                          'scipy', 
                          'networkx', 
                          'matplotlib', 
                          'cython', 
                          'POT'],
        packages = find_packages(exclude=('benchmark', 'scripts')),                       
      )