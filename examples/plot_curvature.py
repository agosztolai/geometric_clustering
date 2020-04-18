"""plot the computed edge curvature"""
import sys
import os
import yaml
import matplotlib.pyplot as plt
import networkx as nx

import geocluster as gc
from geocluster import plotting, io

# get the graph from terminal input
graph_name = sys.argv[-1]
graph = nx.read_gpickle(os.path.join('graphs', 'graph_' + graph_name + '.gpickle'))

os.chdir(graph_name)

# Compute the OR curvatures
times, kappas = io.load_curvature()

# Save results for later analysis
plotting.plot_edge_curvatures(times, kappas)
plotting.plot_edge_curvature_variance(times, kappas)
plt.show()
plotting.plot_graph_snapshots(
    graph, times, kappas, folder="curvature_images", ext=".png", figsize=(15, 7)
)
