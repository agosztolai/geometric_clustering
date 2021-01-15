"""example of how to compute curvature of edges"""
import os

import networkx as nx
import numpy as np

from geocluster import compute_curvatures

m = 'jaccard'

graph = nx.read_gpickle(os.path.join("data","hox_gene_expression_" + m + ".gpickle"))

if not os.path.isdir(m):
    os.mkdir(m)
os.chdir(m)

t_min = -5
t_max = -0.5
n_t = 200
times = np.logspace(t_min, t_max, n_t)
kappas = compute_curvatures(graph, times, n_workers=12) #, weighted_curvature=True)
