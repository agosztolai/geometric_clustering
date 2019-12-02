#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np


setup(
        name = 'geometric_clustering',
        version = '1.0',
        include_dirs = ['utils',np.get_include()], #Add Include path of numpy
        packages=['.'],
      )
