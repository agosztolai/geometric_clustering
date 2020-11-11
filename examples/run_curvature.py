"""example of how to compute curvature of edges"""
import os
import sys

import networkx as nx
import numpy as np

from geocluster import compute_curvatures

graph_name = sys.argv[-1]
graph = nx.read_gpickle(os.path.join("graphs", "graph_" + graph_name + ".gpickle"))

if not os.path.isdir(graph_name):
    os.mkdir(graph_name)
os.chdir(graph_name)

t_min = -2.0
t_max = 0.5
n_t = 100
times = np.logspace(t_min, t_max, n_t)

kappas = compute_curvatures(graph, times, n_workers=12)
