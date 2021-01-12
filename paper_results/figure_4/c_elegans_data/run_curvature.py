"""example of how to compute curvature of edges"""
import os

import networkx as nx
import numpy as np

from geocluster import compute_curvatures

metrics = ['cosine', 'dice', 'jaccard', 'jensenshannon', 
           'kulsinski', 'matching', 'minkowski', 'rogerstanimoto', 
           'russellrao', 'sokalmichener', 'sokalsneath', 
           'yule']

m = 'jaccard'

print(m)

graph = nx.read_gpickle(os.path.join("data","hox_gene_expression_" + m + ".gpickle"))

if not os.path.isdir(m):
    os.mkdir(m)
os.chdir(m)

#jaccard
# t_min = -6
# t_max = 1
t_min = -3
t_max = 0.5
n_t = 100
times = np.logspace(t_min, t_max, n_t)

kappas = compute_curvatures(graph, times, n_workers=16)