"""example of how to cluster a graph based on edges curvatures"""
import sys
import os
import yaml
import networkx as nx
import matplotlib.pyplot as plt

import geocluster as gc
from geocluster import io

from pygenstability import plotting

graph_name = sys.argv[-1]

# load parameters
graph_params = yaml.full_load(open("graph_params.yaml", "rb"))[graph_name]
params = yaml.full_load(open("params.yaml"))
graph = nx.read_gpickle(os.path.join('graphs', 'graph_' + graph_name + '.gpickle'))

os.chdir(graph_name)

times, kappas = io.load_curvature()

cluster_results = gc.cluster(graph, times, kappas, params)

plotting.plot_scan(cluster_results, figure_name='figures/clustering_scan.svg', use_plotly=False)
plt.show()
plotting.plot_communities(graph, cluster_results)
