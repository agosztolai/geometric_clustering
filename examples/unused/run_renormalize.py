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

print("Renormalize")
graphs_reduc = gc.renormalize(graph, 10 ** (-1.0), params)
print(len(graphs_reduc), "renormalized graphs")
print("plot renormalized graphs")
plotting.plot_coarse_grain(graphs_reduc, node_size=20, edge_width=1)
