"""example of how to cluster a graph based on edges curvatures"""
import sys
import os
import yaml

import geocluster as gc
from geocluster import io
from graph_library import generate

from pygenstability import plotting

# get the graph from terminal input
whichgraph = sys.argv[-1]

# load parameters
graph_params = yaml.full_load(open("graph_params.yaml", "rb"))[whichgraph]
params = yaml.full_load(open("params.yaml"))

os.chdir(whichgraph)

# Load graph
graph = generate(whichgraph=whichgraph, params=graph_params)


# Compute the OR curvatures
times, kappas = io.load_curvature()

cluster_results = gc.cluster(graph, times, kappas, params)

plotting.plot_scan(cluster_results)
plotting.plot_communities(graph, cluster_results)
