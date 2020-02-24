"""plot the computed edge curvature"""
import sys
import os
import yaml

import geocluster as gc
from geocluster import plotting, io
from graph_library import generate

# get the graph from terminal input
whichgraph = sys.argv[-1]

# load parameters
paramsfile = "graph_params.yaml"
params = yaml.load(open(paramsfile, "rb"), Loader=yaml.FullLoader)[whichgraph]

os.chdir(whichgraph)

# Load graph
graph = generate(whichgraph=whichgraph, params=params)

# Compute the OR curvatures
times, kappas = io.load_curvature()

# Save results for later analysis
plotting.plot_edge_curvatures(times, kappas, ylog=False)
edge_scales = gc.compute_scales(times, kappas)
plotting.plot_scales(graph, edge_scales)
plotting.plot_graph_snapshots(
    graph, times, kappas, folder="curvature_images", ext=".png", figsize=(15, 7)
)
