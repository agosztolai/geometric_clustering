"""plot the computed edge curvature"""
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
from geocluster import load_curvature, plotting

# get the graph from terminal input
graph_name = 'jaccard' #sys.argv[-1]
graph = nx.read_gpickle(os.path.join("data", "hox_gene_expression_" + graph_name + ".gpickle"))

os.chdir(graph_name)

# Compute the OR curvatures
times, kappas = load_curvature()

# Save results for later analysis
plotting.plot_edge_curvatures(times, kappas)
plotting.plot_edge_curvature_variance(times, kappas)
plt.show()
