"""Example to compute the original Ollivier-Ricci curvature."""
import sys
import os
import yaml
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import geocluster as gc
from geocluster import plotting

graph_name = sys.argv[-1]
graph_params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_name]
graph = nx.read_gpickle(os.path.join("graphs", "graph_" + graph_name + ".gpickle"))
graph = nx.convert_node_labels_to_integers(graph)

if not os.path.isdir(graph_name):
    os.mkdir(graph_name)

params = yaml.full_load(open("params.yaml", "rb"))

os.chdir(graph_name)

kappas = gc.compute_OR_curvature(graph, params)

plotting.plot_graph(
    graph, edge_color=kappas, node_size=10, edge_width=3,
)

plt.savefig('original_OR_curvature.png', bbox_inches='tight')
plt.show()
