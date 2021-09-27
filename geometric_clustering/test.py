#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 13:14:29 2021

@author: gosztola
"""

import ot
import numpy as np
from sinkhorn_gpu import sinkhorn_knopp

# a=[.6, .4]
a=[[.6, .6,.6,.1],
    [.4,.4,.4,.9]]
b=[[.3, .5,.2,.1],
   [.3,.4,.4,.0],
   [.4,.1,.4,.9]]
M=[[0., 1., 1.], 
   [1., 0., 1.]]
W = sinkhorn_knopp(a, b, M, 0.1)
print(W[1])

W = ot.emd2([.6,.4], [.5,.4,.1], M)
print(W)