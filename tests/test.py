#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 18:16:36 2021

@author: gosztola
"""
import numpy as np
import networkx as nx
from geometric_clustering import compute_curvatures


times = np.logspace(-2, 0, 1)
G = nx.planted_partition_graph(2, 100, 0.8, 0.1)

kappas = compute_curvatures(G, times, sinkhorn_regularisation=10, use_gpu=True,n_workers=16)