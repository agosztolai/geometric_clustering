"""example of how to compute curvature of edges"""
import sys
import os
import yaml

import numpy as np

import geocluster as gc
from geocluster import plotting
from graph_library import generate

# get the graph from terminal input
whichgraph = sys.argv[-1]

# load parameters
graph_params = yaml.full_load(open("graph_params.yaml", "rb"))[whichgraph]

if not os.path.isdir(whichgraph):
    os.mkdir(whichgraph)

params = yaml.full_load(open("params.yaml", "rb"))

os.chdir(whichgraph)

# Load graph
graph = generate(whichgraph=whichgraph, params=graph_params)

# Initialise the code with parameters and graph
times = np.logspace(graph_params["t_min"], graph_params["t_max"], graph_params["n_t"])

# Compute the OR curvatures
kappas = gc.compute_curvatures(graph, times, params)

plotting.plot_edge_curvatures(times, kappas, ylog=True, filename="edge_curvature")

plotting.plot_graph_snapshots(
    graph, times, kappas, folder="curvature_images", ext=".png"
)
