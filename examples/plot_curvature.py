"""plot the computed edge curvature"""
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx

from geocluster import load_curvature, plotting

# get the graph from terminal input
graph_name = sys.argv[-1]
graph = nx.read_gpickle(os.path.join("graphs", "graph_" + graph_name + ".gpickle"))

os.chdir(graph_name)

# Compute the OR curvatures
times, kappas = load_curvature()

# Save results for later analysis
plotting.plot_edge_curvatures(times, kappas, ext=".jpg")
plotting.plot_edge_curvature_variance(times, kappas, ext=".jpg")
plt.show()
plotting.plot_graph_snapshots(
    graph, times, kappas, folder="curvature_images", ext=".jpg", figsize=(12, 7)
)
