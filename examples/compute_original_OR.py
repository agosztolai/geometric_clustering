"""Example to compute the original Ollivier-Ricci curvature."""
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx

from geocluster import compute_OR_curvature
from geocluster.plotting import plot_graph

graph_name = sys.argv[-1]
graph = nx.read_gpickle(os.path.join("graphs", "graph_" + graph_name + ".gpickle"))

if not os.path.isdir(graph_name):
    os.mkdir(graph_name)
os.chdir(graph_name)

kappas = compute_OR_curvature(graph)

plot_graph(
    graph, edge_color=kappas, node_size=10, edge_width=3,
)

plt.savefig("original_OR_curvature.png", bbox_inches="tight")
plt.show()
