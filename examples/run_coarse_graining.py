"""example of how to coarse grain a graph based on edge curvatures"""
import sys
import os
import yaml

import geocluster as gc
from geocluster import plotting, io
from graph_library import generate

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

print("Compute scales")
edge_scales = gc.compute_scales(times, kappas)
plotting.plot_scales(graph, edge_scales)

print("Coarse grain")
graphs_reduc = gc.coarse_grain(graph, edge_scales, times)

print("plot coarse grain")
plotting.plot_coarse_grain(graphs_reduc, node_size=20, edge_width=1)
