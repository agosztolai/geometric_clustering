#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:43:49 2020

@author: adamgosztolai
"""
import networkx as nx
from geocluster import geocluster
G = nx.read_gpickle("clique_of_cliques_0_.gpickle")
gc = geocluster.GeoCluster(G)
gc.load_curvature()
gc.plot_edge_curvature()
